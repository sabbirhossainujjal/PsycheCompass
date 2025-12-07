
from utils.logger import setup_logger
from utils.llm import LLMOrchestrator
from pipelines.therapeutic_pipeline import TherapeuticPipeline
import pandas as pd
import yaml
import sys
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = setup_logger('eval_counselbench', 'logs/eval_counselbench.log')


class CounselBenchEvaluator:
    """
    Evaluator for therapeutic system using CounselBench dataset

    This evaluator includes a post-processing subsystem that adapts comprehensive
    therapeutic responses to brief counselor format for CounselBench compliance.
    """

    def __init__(self, config_path: str = 'config.yml'):
        """
        Initialize evaluator

        Args:
            config_path: Path to system configuration
        """
        logger.info("Initializing CounselBench Evaluator")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize LLM
        self.llm = LLMOrchestrator(self.config)

        # Initialize therapeutic pipeline
        self.therapeutic_pipeline = TherapeuticPipeline(self.llm, self.config)

        logger.info("✓ Evaluator initialized successfully")

    def load_counselbench(self, csv_path: str) -> pd.DataFrame:
        """
        Load CounselBench dataset

        Args:
            csv_path: Path to CounselBench CSV

        Returns:
            DataFrame with CounselBench data
        """
        logger.info(f"Loading CounselBench from: {csv_path}")

        df = pd.read_csv(csv_path)

        logger.info(f"✓ Loaded {len(df)} examples")
        logger.info(f"  Unique questions: {df['questionID'].nunique()}")
        logger.info(f"  Topics: {df['topic'].nunique()}")

        return df

    def run_assessment(self, question: str, topic: str) -> Dict:
        """
        Create a mock assessment result for therapeutic response generation

        For CounselBench, we don't have actual PHQ-8 scores, so we create
        a moderate-risk scenario to test therapeutic capabilities

        Args:
            question: User question
            topic: Question topic

        Returns:
            Mock assessment dictionary
        """
        # Detect if question contains crisis indicators
        crisis_keywords = ['suicide', 'kill myself', 'want to die', 'end it all',
                           'self harm', 'hurt myself']

        has_crisis = any(keyword in question.lower()
                         for keyword in crisis_keywords)

        if has_crisis:
            score = 18  # High risk
            classification = "Moderately Severe"
        else:
            score = 10  # Moderate risk (to test therapeutic agent)
            classification = "Moderate"

        # Create mock assessment
        assessment = {
            'user_id': 'counselbench_eval',
            'user_info': {},
            'scale_name': 'PHQ-8',
            'total_score': score,
            'classification': classification,
            'crisis_indicators': ['suicide'] if has_crisis else [],
            'topics': [
                {
                    'topic': 'Depressed Mood',
                    'score': 2,
                    'summary': f'User seeking help regarding {topic}',
                    'basis': question
                },
                {
                    'topic': 'Loss of Interest',
                    'score': 2,
                    'summary': 'Moderate symptoms',
                    'basis': 'Based on question context'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }

        return assessment

    def generate_therapeutic_response(self, question: str, topic: str) -> Dict:
        """
        Generate therapeutic response using PsycheCompass

        This method:
        1. Generates comprehensive therapeutic response (system's actual output)
        2. Condenses to brief format for CounselBench evaluation
        3. Returns both versions for transparency

        Args:
            question: User question
            topic: Question topic

        Returns:
            Response dictionary with both full and condensed versions
        """
        # Guard against missing/NaN question values
        if pd.isna(question) or not str(question).strip():
            logger.warning(
                f"Empty or missing question for topic '{topic}'. Skipping generation.")
            return {
                'response': None,
                'full_response': None,
                'original_length': 0,
                'condensed_length': 0,
                'word_count_original': 0,
                'word_count_condensed': 0,
                'agent_type': None,
                'risk_level': None,
                'success': False,
                'error': 'Empty or missing question'
            }

        logger.debug(
            f"Generating response for question: {str(question)[:100]}...")

        # Create mock assessment
        assessment = self.run_assessment(question, topic)

        # Generate comprehensive response (actual system output)
        try:
            result = self.therapeutic_pipeline.generate_response(assessment)

            full_response = result['response']
            agent_type = result['agent_type']
            risk_level = result['risk_level']

            logger.info(f"Full response generated: {len(full_response)} chars")

            # POST-PROCESSING: Condense to brief format for CounselBench
            brief_response = self._condense_to_brief_response(
                question=question,
                full_response=full_response,
                agent_type=agent_type,
                risk_level=risk_level
            )

            logger.info(f"Brief response created: {len(brief_response)} chars")
            logger.info(
                f"Condensation ratio: {len(full_response) / len(brief_response):.1f}x")

            return {
                'response': brief_response,  # For evaluation
                'full_response': full_response,  # For analysis
                'original_length': len(full_response),
                'condensed_length': len(brief_response),
                'word_count_original': len(full_response.split()),
                'word_count_condensed': len(brief_response.split()),
                'agent_type': agent_type,
                'risk_level': risk_level,
                'success': True,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': None,
                'full_response': None,
                'original_length': 0,
                'condensed_length': 0,
                'word_count_original': 0,
                'word_count_condensed': 0,
                'agent_type': None,
                'risk_level': None,
                'success': False,
                'error': str(e)
            }

    def _condense_to_brief_response(
        self,
        question: str,
        full_response: str,
        agent_type: str,
        risk_level: str
    ) -> str:
        """
        POST-PROCESSING SUBSYSTEM: Condense comprehensive therapeutic response 
        to brief counselor format for CounselBench compliance

        This is purely for format adaptation. The full response represents the
        actual system output and therapeutic capability.

        Args:
            question: Original user question
            full_response: Comprehensive therapeutic response from system
            agent_type: Type of agent that generated response
            risk_level: Risk level assessment

        Returns:
            Brief counselor-style response (2-3 sentences, 40-80 words)
        """
        logger.debug("POST-PROCESSING: Condensing to brief format")

        # Extract excerpt from full response (first 1000 chars for context)
        # This keeps the prompt manageable while providing enough context
        context_excerpt = full_response[:1000]

        # Check for crisis indicators
        crisis_indicators = ['suicide', '988', 'crisis', 'emergency', '911']
        has_crisis_content = any(
            indicator in full_response.lower()
            for indicator in crisis_indicators
        )

        # Build condensation prompt
        prompt = self._build_condensation_prompt(
            question=question,
            context_excerpt=context_excerpt,
            has_crisis=has_crisis_content
        )

        try:
            # Generate brief response
            brief_response = self.llm.generate(
                prompt,
                max_tokens=4056,  # Allow some buffer
                temperature=0.7,  # Some creativity for natural language
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
            ).strip()

            # Quality control: enforce length constraints
            brief_response = self._enforce_length_constraints(brief_response)

            # Validate output
            if not brief_response or len(brief_response) < 20:
                logger.warning("Brief response too short, using fallback")
                return self._fallback_brief_response(question, context_excerpt)

            logger.debug(
                f"Condensation successful: {len(brief_response)} chars, {len(brief_response.split())} words")

            return brief_response

        except Exception as e:
            logger.error(f"Error in condensation: {e}")
            return self._fallback_brief_response(question, context_excerpt)

    def _build_condensation_prompt(
        self,
        question: str,
        context_excerpt: str,
        has_crisis: bool
    ) -> str:
        """
        Build prompt for LLM-based condensation

        Args:
            question: Original question
            context_excerpt: Excerpt from full response
            has_crisis: Whether response contains crisis content

        Returns:
            Condensation prompt
        """
        crisis_instruction = ""
        if has_crisis:
            crisis_instruction = """
CRITICAL: The original response contains crisis/safety information. 
If crisis resources (like 988) are mentioned, keep one brief safety sentence."""

        prompt = f"""You are adapting a comprehensive therapeutic response to brief counselor format.

ORIGINAL QUESTION:
{question}

COMPREHENSIVE THERAPEUTIC RESPONSE (excerpt):
{context_excerpt}

YOUR TASK:
Create a brief, warm counselor response (2-3 sentences, 40-80 words maximum).

MUST INCLUDE:
1. Validation of the person's feelings or experience
2. One practical suggestion, insight, or perspective

MUST EXCLUDE:
- Long explanations or therapeutic protocols
- Bullet points or lists
- Resource directories (SAMHSA, Psychology Today, etc.)
- Homework assignments
- Multiple paragraphs
- Psychoeducation sections
{crisis_instruction}

STYLE:
- Warm and empathetic
- Conversational, not clinical
- Direct and clear
- 2-3 sentences ONLY

Write ONLY the brief response, no preamble or explanation:"""

        return prompt

    def _enforce_length_constraints(self, response: str) -> str:
        """
        Enforce length constraints on brief response

        Args:
            response: Generated brief response

        Returns:
            Length-constrained response
        """
        # Count words
        word_count = len(response.split())

        # If too long, truncate to first 3 sentences
        if word_count > 100:
            logger.debug(f"Response too long ({word_count} words), truncating")

            # Split into sentences (simple split on . ! ?)
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Take first 3 sentences
            truncated = '. '.join(sentences[:3])
            if not truncated.endswith('.'):
                truncated += '.'

            return truncated

        return response

    def _fallback_brief_response(self, question: str, context: str) -> str:
        """
        Fallback method if LLM condensation fails

        Uses simple extraction from the original response

        Args:
            question: Original question
            context: Context from full response

        Returns:
            Fallback brief response
        """
        logger.warning("Using fallback condensation method")

        # Split into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

        if not sentences:
            return "I understand this is challenging. Consider talking with someone you trust or a professional counselor for personalized support."

        # Try to extract validation + suggestion
        # Usually first sentence is validation, look for action words in others
        validation = sentences[0] if sentences else ""

        action_keywords = ['consider', 'try',
                           'might', 'could', 'help', 'suggest']
        suggestion = ""

        for sentence in sentences[1:]:
            if any(keyword in sentence.lower() for keyword in action_keywords):
                suggestion = sentence
                break

        if validation and suggestion:
            brief = f"{validation}. {suggestion}."
        elif validation:
            brief = f"{validation}. Consider seeking support from a professional counselor."
        else:
            brief = '. '.join(sentences[:2]) + '.'

        # Final length check
        if len(brief.split()) > 100:
            brief = '. '.join(sentences[:3]) + '.'

        return brief

    def evaluate_with_llm_judge(
        self,
        question: str,
        generated_response: str,
        reference_response: str = None
    ) -> Dict:
        """
        Use LLM as judge to evaluate response quality

        Evaluates:
        - Empathy (1-5)
        - Specificity (1-5)
        - Actionability (1-5)
        - Clinical appropriateness (1-5)
        - Overall quality (1-5)

        Args:
            question: Original question
            generated_response: Response from PsycheCompass (condensed version)
            reference_response: Optional reference response

        Returns:
            Dictionary with evaluation scores and reasoning
        """
        logger.debug("Running LLM-as-Judge evaluation")

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            question=question,
            generated_response=generated_response,
            reference_response=reference_response
        )

        try:
            # Generate evaluation
            response = self.llm.generate(
                prompt,
                max_tokens=2048,  # Increased for full evaluation
                temperature=0.7
            )

            # Parse JSON response
            evaluation = self._extract_and_parse_json(response)

            logger.debug("✓ LLM evaluation completed")
            return evaluation

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}", exc_info=True)
            return self._default_evaluation_scores(f"Error: {str(e)}")

    def _build_evaluation_prompt(
        self,
        question: str,
        generated_response: str,
        reference_response: str = None
    ) -> str:
        """
        Build prompt for LLM-as-Judge evaluation

        Args:
            question: Original question
            generated_response: Generated response to evaluate
            reference_response: Optional reference

        Returns:
            Evaluation prompt
        """
        reference_section = ""
        if reference_response:
            reference_section = f"""
**REFERENCE RESPONSE (for comparison):**
{reference_response}
"""

        prompt = f"""You are an expert evaluator of mental health therapeutic responses.

**USER QUESTION:**
{question}

**RESPONSE TO EVALUATE:**
{generated_response}
{reference_section}

**YOUR TASK:**
Evaluate this response on 5 dimensions. Each score is 1-5 where:
- 1 = Very poor
- 2 = Poor  
- 3 = Adequate
- 4 = Good
- 5 = Excellent

**EVALUATION DIMENSIONS:**

1. **EMPATHY (1-5):** Does the response show genuine understanding and validate the person's feelings?
   
2. **SPECIFICITY (1-5):** Does it provide concrete, specific guidance rather than vague platitudes?

3. **ACTIONABILITY (1-5):** Are there clear, implementable steps or suggestions the person can actually take?

4. **CLINICAL APPROPRIATENESS (1-5):** Is the response appropriate for the severity and type of concern expressed?

5. **OVERALL QUALITY (1-5):** Overall, how helpful and therapeutic is this response?

**CRITICAL: Respond with ONLY a JSON object. No text before or after.**

Use this EXACT format:
{{
  "empathy_score": 4,
  "empathy_reasoning": "Brief explanation here",
  "specificity_score": 3,
  "specificity_reasoning": "Brief explanation here",
  "actionability_score": 4,
  "actionability_reasoning": "Brief explanation here",
  "clinical_appropriateness_score": 5,
  "clinical_appropriateness_reasoning": "Brief explanation here",
  "overall_score": 4,
  "overall_reasoning": "Brief explanation here",
  "strengths": ["Strength 1", "Strength 2"],
  "weaknesses": ["Weakness 1", "Weakness 2"]
}}

JSON OUTPUT:"""

        return prompt

    def _extract_and_parse_json(self, response: str) -> Dict:
        """
        Robustly extract and parse JSON from LLM response

        Args:
            response: LLM response (may contain JSON in various formats)

        Returns:
            Parsed evaluation dictionary
        """
        json_candidates = []

        # Strategy 1: Extract from markdown code blocks
        if '```json' in response:
            try:
                json_str = response.split('```json')[1].split('```')[0].strip()
                json_candidates.append(json_str)
            except IndexError:
                pass

        if '```' in response and not json_candidates:
            try:
                json_str = response.split('```')[1].split('```')[0].strip()
                json_candidates.append(json_str)
            except IndexError:
                pass

        # Strategy 2: Find JSON object boundaries using regex
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        json_candidates.extend(matches)

        # Strategy 3: Try the entire response
        json_candidates.append(response.strip())

        # Try parsing each candidate
        for i, candidate in enumerate(json_candidates):
            try:
                parsed = json.loads(candidate)

                # Validate it has the expected keys
                required_keys = ['empathy_score', 'overall_score']
                if all(key in parsed for key in required_keys):
                    logger.debug(f"Successfully parsed JSON (strategy {i+1})")
                    return parsed

            except json.JSONDecodeError:
                continue

        # All parsing attempts failed
        logger.error("Failed to parse JSON from LLM evaluation")
        logger.error(f"Response preview: {response[:500]}...")

        return self._default_evaluation_scores("JSON parsing failed")

    def _default_evaluation_scores(self, error_msg: str) -> Dict:
        """
        Return default evaluation scores when parsing fails

        Args:
            error_msg: Error message to include

        Returns:
            Default evaluation dictionary
        """
        return {
            'empathy_score': 0,
            'empathy_reasoning': error_msg,
            'specificity_score': 0,
            'specificity_reasoning': error_msg,
            'actionability_score': 0,
            'actionability_reasoning': error_msg,
            'clinical_appropriateness_score': 0,
            'clinical_appropriateness_reasoning': error_msg,
            'overall_score': 0,
            'overall_reasoning': error_msg,
            'strengths': [],
            'weaknesses': ['Evaluation system error'],
            'error': error_msg
        }

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        num_samples: int = None,
        output_dir: str = 'counselbench_results'
    ) -> Dict:
        """
        Evaluate therapeutic system on CounselBench dataset

        Args:
            df: CounselBench DataFrame
            num_samples: Number of samples to evaluate (None = all)
            output_dir: Directory to save results

        Returns:
            Evaluation results dictionary
        """
        logger.info("="*70)
        logger.info("STARTING COUNSELBENCH EVALUATION")
        logger.info("="*70)

        # Sample if needed
        if num_samples and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)
            logger.info(f"Evaluating {num_samples} samples")
        else:
            logger.info(f"Evaluating all {len(df)} samples")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Create directory for individual result files (one JSON per question)
        individual_dir = os.path.join(output_dir, 'individual_result')
        os.makedirs(individual_dir, exist_ok=True)

        # Get unique questions (CounselBench has multiple responses per question)
        unique_questions = df.groupby('questionID').first().reset_index()

        logger.info(f"Evaluating {len(unique_questions)} unique questions")

        results = []
        failed_count = 0

        # Evaluate each question
        for idx, row in tqdm(unique_questions.iterrows(),
                             total=len(unique_questions),
                             desc="Evaluating"):

            question_id = row['questionID']
            question = row['questionText']
            topic = row['topic']
            reference_response = row['response']

            # Get ground truth scores (average across evaluators)
            question_evals = df[df['questionID'] == question_id]
            gt_empathy = question_evals['empathy_score'].mean()
            gt_specificity = question_evals['specificity_score'].mean()
            gt_overall = question_evals['overall_score'].mean()
            # Sanitize question_id for filesystem operations
            safe_qid = re.sub(r'[^A-Za-z0-9_.-]', '_', str(question_id))
            individual_path = os.path.join(individual_dir, f"{safe_qid}.json")

            # If an individual result file already exists, load it and skip evaluation
            if os.path.exists(individual_path):
                try:
                    with open(individual_path, 'r') as ind_f:
                        existing_result = json.load(ind_f)
                    results.append(existing_result)
                    logger.info(
                        f"Skipping {question_id}; loaded existing result.")
                    continue
                except Exception as e:
                    logger.warning(
                        f"Failed to load existing result for {question_id}, will re-run: {e}")

            # Generate response (includes condensation)
            gen_result = self.generate_therapeutic_response(question, topic)

            if not gen_result['success']:
                failed_count += 1
                logger.warning(
                    f"Failed to generate response for {question_id}")
                continue

            brief_response = gen_result['response']
            full_response = gen_result['full_response']
            agent_type = gen_result['agent_type']
            risk_level = gen_result['risk_level']

            # Evaluate with LLM judge
            evaluation = self.evaluate_with_llm_judge(
                question=question,
                generated_response=brief_response,
                reference_response=reference_response
            )

            # Compile result
            result = {
                'question_id': question_id,
                'topic': topic,
                'question': question,

                # Responses
                # Brief version (evaluated)
                'generated_response': brief_response,
                # Full version (actual system)
                'full_system_response': full_response,
                'reference_response': reference_response,

                # Response metadata
                'original_length_chars': gen_result['original_length'],
                'condensed_length_chars': gen_result['condensed_length'],
                'original_word_count': gen_result['word_count_original'],
                'condensed_word_count': gen_result['word_count_condensed'],
                'condensation_ratio': gen_result['original_length'] / max(gen_result['condensed_length'], 1),

                # System info
                'agent_type': agent_type,
                'risk_level': risk_level,

                # LLM-as-Judge scores
                'llm_empathy_score': evaluation.get('empathy_score', 1),
                'llm_empathy_reasoning': evaluation.get('empathy_reasoning', ''),
                'llm_specificity_score': evaluation.get('specificity_score', 1),
                'llm_specificity_reasoning': evaluation.get('specificity_reasoning', ''),
                'llm_actionability_score': evaluation.get('actionability_score', 1),
                'llm_actionability_reasoning': evaluation.get('actionability_reasoning', ''),
                'llm_clinical_score': evaluation.get('clinical_appropriateness_score', 1),
                'llm_clinical_reasoning': evaluation.get('clinical_appropriateness_reasoning', ''),
                'llm_overall_score': evaluation.get('overall_score', 1),
                'llm_overall_reasoning': evaluation.get('overall_reasoning', ''),
                'strengths': evaluation.get('strengths', []),
                'weaknesses': evaluation.get('weaknesses', []),

                # Ground truth scores (from CounselBench)
                'gt_empathy_score': gt_empathy,
                'gt_specificity_score': gt_specificity,
                'gt_overall_score': gt_overall,

                # Comparison
                'empathy_diff': evaluation.get('empathy_score', 0) - gt_empathy,
                'specificity_diff': evaluation.get('specificity_score', 0) - gt_specificity,
                'overall_diff': evaluation.get('overall_score', 0) - gt_overall,
            }

            # Save individual JSON file for this question
            try:
                # Sanitize question_id for filesystem safety
                safe_qid = re.sub(r'[^A-Za-z0-9_.-]', '_', str(question_id))
                individual_path = os.path.join(
                    individual_dir, f"{safe_qid}.json")
                with open(individual_path, 'w') as ind_f:
                    json.dump(result, ind_f, indent=2, default=str)
                logger.debug(f"Saved individual result: {individual_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to save individual result for {question_id}: {e}")

            results.append(result)

        logger.info("="*70)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Successful: {len(results)}/{len(unique_questions)}")
        logger.info(f"Failed: {failed_count}")
        logger.info("="*70)

        # Compute aggregate statistics
        results_df = pd.DataFrame(results)

        statistics = self._compute_statistics(results_df)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_path = f"{output_dir}/detailed_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✓ Detailed results saved: {results_path}")

        # Save statistics
        stats_path = f"{output_dir}/statistics_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)

        logger.info(f"✓ Statistics saved: {stats_path}")

        # Save CSV
        csv_path = f"{output_dir}/results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)

        logger.info(f"✓ CSV results saved: {csv_path}")

        return {
            'results': results,
            'statistics': statistics,
            'output_files': {
                'detailed': results_path,
                'statistics': stats_path,
                'csv': csv_path
            }
        }

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute aggregate statistics

        Args:
            df: Results DataFrame

        Returns:
            Statistics dictionary
        """
        logger.info("Computing statistics...")

        stats = {
            'total_evaluated': len(df),

            # Response length statistics
            'response_lengths': {
                'original_avg_chars': df['original_length_chars'].mean(),
                'original_avg_words': df['original_word_count'].mean(),
                'condensed_avg_chars': df['condensed_length_chars'].mean(),
                'condensed_avg_words': df['condensed_word_count'].mean(),
                'avg_condensation_ratio': df['condensation_ratio'].mean(),
            },

            # LLM-as-Judge scores
            'llm_scores': {
                'empathy': {
                    'mean': df['llm_empathy_score'].mean(),
                    'std': df['llm_empathy_score'].std(),
                    'median': df['llm_empathy_score'].median(),
                    'min': df['llm_empathy_score'].min(),
                    'max': df['llm_empathy_score'].max()
                },
                'specificity': {
                    'mean': df['llm_specificity_score'].mean(),
                    'std': df['llm_specificity_score'].std(),
                    'median': df['llm_specificity_score'].median(),
                    'min': df['llm_specificity_score'].min(),
                    'max': df['llm_specificity_score'].max()
                },
                'actionability': {
                    'mean': df['llm_actionability_score'].mean(),
                    'std': df['llm_actionability_score'].std(),
                    'median': df['llm_actionability_score'].median(),
                    'min': df['llm_actionability_score'].min(),
                    'max': df['llm_actionability_score'].max()
                },
                'clinical_appropriateness': {
                    'mean': df['llm_clinical_score'].mean(),
                    'std': df['llm_clinical_score'].std(),
                    'median': df['llm_clinical_score'].median(),
                    'min': df['llm_clinical_score'].min(),
                    'max': df['llm_clinical_score'].max()
                },
                'overall': {
                    'mean': df['llm_overall_score'].mean(),
                    'std': df['llm_overall_score'].std(),
                    'median': df['llm_overall_score'].median(),
                    'min': df['llm_overall_score'].min(),
                    'max': df['llm_overall_score'].max()
                }
            },

            # Comparison with ground truth
            'comparison_with_gt': {
                'empathy_mae': abs(df['empathy_diff']).mean(),
                'specificity_mae': abs(df['specificity_diff']).mean(),
                'overall_mae': abs(df['overall_diff']).mean(),
            },

            # Agent routing statistics
            'agent_routing': {
                'emotional_support': (df['agent_type'] == 'emotional_support').sum(),
                'therapeutic': (df['agent_type'] == 'therapeutic').sum(),
                'crisis': (df['agent_type'] == 'crisis').sum()
            },

            # Topic breakdown
            'by_topic': {}
        }

        # Compute per-topic statistics
        for topic in df['topic'].unique():
            topic_df = df[df['topic'] == topic]
            stats['by_topic'][topic] = {
                'count': len(topic_df),
                'llm_empathy_mean': topic_df['llm_empathy_score'].mean(),
                'llm_specificity_mean': topic_df['llm_specificity_score'].mean(),
                'llm_overall_mean': topic_df['llm_overall_score'].mean(),
                'avg_word_count_condensed': topic_df['condensed_word_count'].mean()
            }

        logger.info("✓ Statistics computed")

        return stats


def print_evaluation_summary(results: Dict):
    """
    Print evaluation summary to console

    Args:
        results: Evaluation results dictionary
    """
    stats = results['statistics']

    print("\n" + "="*70)
    print("COUNSELBENCH EVALUATION SUMMARY")
    print("="*70)

    print(f"\nTotal Questions Evaluated: {stats['total_evaluated']}")

    # Response length summary
    print("\n" + "-"*70)
    print("RESPONSE LENGTH ANALYSIS")
    print("-"*70)

    lengths = stats['response_lengths']
    print(
        f"\nOriginal (System Output):   {lengths['original_avg_words']:.0f} words avg ({lengths['original_avg_chars']:.0f} chars)")
    print(
        f"Condensed (For Evaluation): {lengths['condensed_avg_words']:.0f} words avg ({lengths['condensed_avg_chars']:.0f} chars)")
    print(
        f"Condensation Ratio:         {lengths['avg_condensation_ratio']:.1f}x")

    print("\n" + "-"*70)
    print("LLM-AS-JUDGE SCORES (1-5 scale)")
    print("-"*70)

    llm_scores = stats['llm_scores']

    print(
        f"\nEmpathy Score:              {llm_scores['empathy']['mean']:.2f} ± {llm_scores['empathy']['std']:.2f}")
    print(
        f"Specificity Score:          {llm_scores['specificity']['mean']:.2f} ± {llm_scores['specificity']['std']:.2f}")
    print(
        f"Actionability Score:        {llm_scores['actionability']['mean']:.2f} ± {llm_scores['actionability']['std']:.2f}")
    print(
        f"Clinical Appropriateness:   {llm_scores['clinical_appropriateness']['mean']:.2f} ± {llm_scores['clinical_appropriateness']['std']:.2f}")
    print(
        f"Overall Quality:            {llm_scores['overall']['mean']:.2f} ± {llm_scores['overall']['std']:.2f}")

    print("\n" + "-"*70)
    print("COMPARISON WITH COUNSELBENCH GROUND TRUTH")
    print("-"*70)

    comparison = stats['comparison_with_gt']

    print(f"\nEmpathy MAE:        {comparison['empathy_mae']:.2f}")
    print(f"Specificity MAE:    {comparison['specificity_mae']:.2f}")
    print(f"Overall MAE:        {comparison['overall_mae']:.2f}")

    print("\nNote: Ground truth scores reflect human ratings of reference responses,")
    print("not ratings of your system's responses.")

    print("\n" + "-"*70)
    print("AGENT ROUTING DISTRIBUTION")
    print("-"*70)

    routing = stats['agent_routing']
    total = sum(routing.values())

    if total > 0:
        print(
            f"\nEmotional Support:  {routing['emotional_support']:3d} ({routing['emotional_support']/total*100:.1f}%)")
        print(
            f"Therapeutic Agent:  {routing['therapeutic']:3d} ({routing['therapeutic']/total*100:.1f}%)")
        print(
            f"Crisis Support:     {routing['crisis']:3d} ({routing['crisis']/total*100:.1f}%)")

    print("\n" + "-"*70)
    print("TOP 5 TOPICS BY PERFORMANCE")
    print("-"*70)

    by_topic = stats['by_topic']
    sorted_topics = sorted(by_topic.items(),
                           key=lambda x: x[1]['llm_overall_mean'],
                           reverse=True)[:5]

    print(f"\n{'Topic':<30} {'Count':<8} {'Overall Score':<15} {'Avg Words':<12}")
    print("-"*70)

    for topic, topic_stats in sorted_topics:
        print(
            f"{topic:<30} {topic_stats['count']:<8} {topic_stats['llm_overall_mean']:.2f}              {topic_stats['avg_word_count_condensed']:.0f}")

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)

    files = results['output_files']
    print(f"\nDetailed Results:   {files['detailed']}")
    print(f"Statistics:         {files['statistics']}")
    print(f"CSV Export:         {files['csv']}")

    print("\n" + "="*70)
    print("\nNOTE: This evaluation uses post-processing adaptation to condense")
    print("comprehensive therapeutic responses to brief counselor format.")
    print("Both versions are saved for transparency and analysis.")
    print("="*70)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate PsycheCompass therapeutic system on CounselBench'
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/counselbench_eval.csv',
        help='Path to CounselBench CSV file'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to config file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/counselbench',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("COUNSELBENCH EVALUATION FOR PSYCHECOMPASS")
    print("="*70)
    print(f"\nDataset:        {args.csv_path}")
    print(f"Num Samples:    {args.num_samples if args.num_samples else 'All'}")
    print(f"Output Dir:     {args.output_dir}")
    print("="*70 + "\n")

    try:
        # Initialize evaluator
        evaluator = CounselBenchEvaluator(config_path=args.config)

        # Load dataset
        df = evaluator.load_counselbench(args.csv_path)

        # Run evaluation
        results = evaluator.evaluate_dataset(
            df=df,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )

        # Print summary
        print_evaluation_summary(results)

        print("\n✓ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
