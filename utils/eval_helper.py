import numpy as np
from typing import Dict, List, Any
import re


def generate_therapy_response(
    llm: Any,
    patient_input: str,
    assessment_results: Dict = None,
    conversation_history: List[Dict] = None,
    question: str = None
) -> str:
    """
    Generate a therapy/counseling response

    Works for both:
    - Psych8k: Uses assessment results + conversation history
    - CounselBench: Uses question only

    Args:
        llm: LLM orchestrator
        patient_input: Patient description or question
        assessment_results: Optional assessment findings (Psych8k)
        conversation_history: Optional Q&A interactions (Psych8k)
        question: Optional direct question (CounselBench)

    Returns:
        Generated therapy response
    """

    if question:
        # CounselBench mode - direct question answering
        prompt = f"""You are an experienced, empathetic mental health counselor.

USER QUESTION:
{question}

Provide a compassionate, professional counseling response that:
1. Demonstrates empathy and validates feelings
2. Provides specific, actionable guidance
3. Includes practical steps or coping strategies
4. Maintains appropriate professional boundaries
5. Addresses safety concerns if present

Write a helpful counseling response (2-4 paragraphs).

COUNSELING RESPONSE:"""

    else:
        # Psych8k mode - assessment-based response
        conversation_summary = ""
        if conversation_history:
            conversation_summary = "\n\nKey points from assessment:\n"
            for interaction in conversation_history[:5]:
                conversation_summary += f"- Q: {interaction['question'][:100]}...\n"
                conversation_summary += f"  A: {interaction['llm_response'][:100]}...\n"

        topic_summary = ""
        if assessment_results and assessment_results.get('topics'):
            topic_summary = "\n\nAssessment findings:\n"
            for topic in assessment_results['topics']:
                topic_summary += f"- {topic['topic_name']}: Score {topic.get('final_score', 0)}/3\n"
                if topic.get('summary'):
                    topic_summary += f"  {topic['summary'][:150]}...\n"

        score = assessment_results.get(
            'total_score', 10) if assessment_results else 10
        classification = assessment_results.get(
            'classification', 'Moderate') if assessment_results else 'Moderate'

        prompt = f"""You are an experienced, empathetic mental health counselor. Based on your assessment, provide an appropriate counseling response.

PATIENT'S DESCRIPTION:
{patient_input}

{conversation_summary}

ASSESSMENT RESULTS:
- Total PHQ-8 Score: {score}/24
- Classification: {classification}
{topic_summary}

Provide a compassionate, professional counseling response that:
1. Acknowledges and validates the patient's experiences
2. Shows empathy and understanding
3. Addresses the key concerns from assessment
4. Provides appropriate guidance or next steps
5. Maintains a supportive, non-judgmental tone

Write a counseling response (2-4 paragraphs).

COUNSELING RESPONSE:"""

    try:
        response = llm.generate(prompt, max_tokens=5000, temperature=0.7)
        return response.strip()
    except Exception as e:
        print(f"Error generating therapy response: {e}")
        return "I appreciate you sharing your experiences with me. I'd like to work together to explore strategies that might help."


def calculate_similarity_metrics(generated: str, expected: str) -> Dict[str, float]:
    """
    Calculate similarity metrics between generated and expected responses

    Args:
        generated: Generated therapy response
        expected: Expected therapy response

    Returns:
        Dictionary with similarity scores
    """

    metrics = {}

    # Basic metrics
    metrics['length_generated'] = len(generated)
    metrics['length_expected'] = len(expected)
    metrics['length_ratio'] = len(generated) / max(len(expected), 1)

    # Word-level
    words_generated = set(generated.lower().split())
    words_expected = set(expected.lower().split())

    intersection = words_generated & words_expected
    union = words_generated | words_expected

    metrics['word_overlap'] = len(intersection) / max(len(union), 1)
    metrics['jaccard_similarity'] = len(intersection) / max(len(union), 1)

    # N-gram overlap
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    bigrams_gen = get_ngrams(generated, 2)
    bigrams_exp = get_ngrams(expected, 2)

    if bigrams_exp:
        metrics['bigram_overlap'] = len(
            bigrams_gen & bigrams_exp) / len(bigrams_exp)
    else:
        metrics['bigram_overlap'] = 0.0

    # Semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        emb_gen = model.encode(generated)
        emb_exp = model.encode(expected)

        cosine_sim = np.dot(emb_gen, emb_exp) / \
            (np.linalg.norm(emb_gen) * np.linalg.norm(emb_exp))
        metrics['semantic_similarity'] = float(cosine_sim)

    except ImportError:
        metrics['semantic_similarity'] = metrics['word_overlap']
    except Exception as e:
        print(f"Warning: Could not compute semantic similarity: {e}")
        metrics['semantic_similarity'] = metrics['word_overlap']

    return metrics


def evaluate_therapy_quality(
    llm: Any,
    generated_response: str,
    expected_response: str = None,
    patient_input: str = None,
    assessment_results: Dict = None,
    question: str = None,
    ground_truth_scores: Dict = None
) -> Dict[str, Any]:
    """
    Unified LLM-as-judge evaluation using 0-5 scale

    SCALE (consistent across datasets):
    0 = Not Acceptable
    1 = Poor
    2 = Fair
    3 = Good
    4 = Very Good
    5 = Excellent

    Args:
        llm: LLM orchestrator
        generated_response: Generated therapy response
        expected_response: Expected response (optional)
        patient_input: Patient description or question
        assessment_results: Assessment results (Psych8k only)
        question: Direct question (CounselBench only)
        ground_truth_scores: Ground truth scores if available (CounselBench)

    Returns:
        Dictionary with quality scores (0-5 scale) and analysis
    """

    # Build context
    context = ""
    if question:
        context = f"USER QUESTION:\n{question}\n"
    elif patient_input:
        context = f"PATIENT'S SITUATION:\n{patient_input}\n"

    if assessment_results:
        context += f"\nASSESSMENT RESULTS:\n"
        context += f"- PHQ-8 Score: {assessment_results.get('total_score', 'N/A')}/24\n"
        context += f"- Classification: {assessment_results.get('classification', 'N/A')}\n"

    prompt = f"""You are an expert evaluator of mental health counseling responses.

{context}

GENERATED COUNSELING RESPONSE:
{generated_response}
"""

    if expected_response:
        prompt += f"""
EXPECTED COUNSELING RESPONSE (for reference):
{expected_response}
"""

    prompt += """
Evaluate the GENERATED response using a 0-5 scale:
- 0 = Not Acceptable
- 1 = Poor
- 2 = Fair
- 3 = Good
- 4 = Very Good
- 5 = Excellent

Rate on the following dimensions (0-5 for each):

1. EMPATHY: Understanding and emotional support
2. SPECIFICITY: Concrete, specific guidance
3. ACTIONABILITY: Clear, implementable steps
4. APPROPRIATENESS: Suitable for the situation
5. HELPFULNESS: Provides genuine value
6. OVERALL QUALITY: Overall therapeutic value

Also provide:
7. STRENGTHS: List 2-3 key strengths
8. WEAKNESSES: List 1-2 areas for improvement
9. COMPARISON: How does it compare to expected response (if provided)?

Respond ONLY in JSON format:
{
    "empathy": <0-5>,
    "specificity": <0-5>,
    "actionability": <0-5>,
    "appropriateness": <0-5>,
    "helpfulness": <0-5>,
    "overall_quality": <0-5>,
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1"],
    "comparison": "Brief comparison if expected response provided",
    "reasoning": "Brief explanation"
}

JSON RESPONSE:"""

    try:
        response = llm.generate(prompt, max_tokens=8000, temperature=0.3)

        # Extract JSON
        import json

        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith('```'):
                    if in_json:
                        break
                    in_json = True
                    continue
                if in_json:
                    json_lines.append(line)
            response = '\n'.join(json_lines)

        evaluation = json.loads(response)

        # Ensure all required fields
        required = ['empathy', 'specificity', 'actionability',
                    'appropriateness', 'helpfulness', 'overall_quality']
        for field in required:
            if field not in evaluation:
                evaluation[field] = 2.5  # Default middle score
            # Ensure scores are in 0-5 range
            evaluation[field] = max(0, min(5, float(evaluation[field])))

        if 'strengths' not in evaluation:
            evaluation['strengths'] = []
        if 'weaknesses' not in evaluation:
            evaluation['weaknesses'] = []
        if 'comparison' not in evaluation:
            evaluation['comparison'] = "No comparison available"
        if 'reasoning' not in evaluation:
            evaluation['reasoning'] = "No reasoning provided"

        # Add ground truth comparison if available
        if ground_truth_scores:
            evaluation['ground_truth_comparison'] = {
                'empathy_diff': evaluation['empathy'] - ground_truth_scores.get('empathy', 0),
                'specificity_diff': evaluation['specificity'] - ground_truth_scores.get('specificity', 0),
                'overall_diff': evaluation['overall_quality'] - ground_truth_scores.get('overall', 0)
            }

        return evaluation

    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        print(
            f"Raw response: {response[:500] if 'response' in locals() else 'N/A'}...")

        # Return default scores (middle of 0-5 range)
        return {
            'empathy': 2.5,
            'specificity': 2.5,
            'actionability': 2.5,
            'appropriateness': 2.5,
            'helpfulness': 2.5,
            'overall_quality': 2.5,
            'strengths': ["Unable to evaluate"],
            'weaknesses': ["Evaluation failed"],
            'comparison': "Evaluation failed",
            'reasoning': f"Error: {str(e)}",
            'error': str(e)
        }


def convert_scale_to_05(score_10: float) -> float:
    """
    Convert 1-10 scale to 0-5 scale

    Args:
        score_10: Score on 1-10 scale

    Returns:
        Score on 0-5 scale
    """
    # Linear mapping: 1-10 -> 0-5
    # 1 -> 0, 10 -> 5
    return (score_10 - 1) * (5 / 9)


def convert_scale_to_10(score_05: float) -> float:
    """
    Convert 0-5 scale to 1-10 scale

    Args:
        score_05: Score on 0-5 scale

    Returns:
        Score on 1-10 scale
    """
    # Linear mapping: 0-5 -> 1-10
    # 0 -> 1, 5 -> 10
    return 1 + (score_05 * 9 / 5)


def get_scale_label(score: float) -> str:
    """
    Get descriptive label for 0-5 score

    Args:
        score: Score on 0-5 scale

    Returns:
        Descriptive label
    """
    if score < 0.5:
        return "Not Acceptable"
    elif score < 1.5:
        return "Poor"
    elif score < 2.5:
        return "Fair"
    elif score < 3.5:
        return "Good"
    elif score < 4.5:
        return "Very Good"
    else:
        return "Excellent"


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across multiple evaluations

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with aggregate statistics
    """

    if not results:
        return {}

    successful = [r for r in results if r.get('status') == 'success']

    if not successful:
        return {'error': 'No successful evaluations'}

    aggregates = {
        'total_evaluated': len(results),
        'successful': len(successful),
        'failed': len(results) - len(successful)
    }

    # Therapy quality metrics (0-5 scale)
    quality_scores = [r['therapy_results']['quality_evaluation']['overall_quality']
                      for r in successful
                      if 'therapy_results' in r]

    if quality_scores:
        aggregates['therapy'] = {
            'mean_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'mean_label': get_scale_label(np.mean(quality_scores))
        }

    # Assessment metrics if available
    assessment_scores = [r['assessment_results']['total_score']
                         for r in successful
                         if 'assessment_results' in r]

    if assessment_scores:
        aggregates['assessment'] = {
            'mean_score': np.mean(assessment_scores),
            'std_score': np.std(assessment_scores),
            'min_score': np.min(assessment_scores),
            'max_score': np.max(assessment_scores)
        }

    return aggregates


def generate_unified_report(results: List[Dict], output_path: str, dataset_name: str):
    """
    Generate unified evaluation report

    Args:
        results: List of evaluation results
        output_path: Path to save report
        dataset_name: Name of dataset (Psych8k, CounselBench, etc.)
    """

    aggregates = compute_aggregate_metrics(results)

    report = []
    report.append("="*70)
    report.append(f"{dataset_name.upper()} EVALUATION REPORT")
    report.append("="*70)
    report.append("")

    report.append(f"Total Evaluations: {aggregates['total_evaluated']}")
    report.append(f"Successful: {aggregates['successful']}")
    report.append(f"Failed: {aggregates['failed']}")
    report.append("")

    if 'therapy' in aggregates:
        report.append("-"*70)
        report.append("THERAPY QUALITY METRICS (0-5 Scale)")
        report.append("-"*70)
        report.append(
            f"Mean Overall Quality: {aggregates['therapy']['mean_quality']:.2f}/5")
        report.append(f"  ({aggregates['therapy']['mean_label']})")
        report.append(f"Std Dev: {aggregates['therapy']['std_quality']:.2f}")
        report.append(
            f"Range: {aggregates['therapy']['min_quality']:.2f} - {aggregates['therapy']['max_quality']:.2f}")
        report.append("")

        report.append("Scale Reference:")
        report.append("  0 = Not Acceptable")
        report.append("  1 = Poor")
        report.append("  2 = Fair")
        report.append("  3 = Good")
        report.append("  4 = Very Good")
        report.append("  5 = Excellent")
        report.append("")

    if 'assessment' in aggregates:
        report.append("-"*70)
        report.append("ASSESSMENT METRICS")
        report.append("-"*70)
        report.append(
            f"Mean PHQ-8 Score: {aggregates['assessment']['mean_score']:.2f}/24")
        report.append(f"Std Dev: {aggregates['assessment']['std_score']:.2f}")
        report.append(
            f"Range: {aggregates['assessment']['min_score']:.0f} - {aggregates['assessment']['max_score']:.0f}")
        report.append("")

    report.append("="*70)

    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)

    return report_text
