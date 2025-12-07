import pandas as pd
import yaml
import sys
import os
import json
import random
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pipelines.assessment_pipeline import AssessmentPipeline
from utils.llm import LLMOrchestrator
from utils.logger import setup_logger
from utils.eval_helper import (
    generate_therapy_response,
    evaluate_therapy_quality,
    calculate_similarity_metrics
)

logger = None

def setup_logging(log_file='logs/eval_psych8k_complete.log'):
    """Setup logging"""
    global logger
    os.makedirs('logs', exist_ok=True)
    if 'setup_logger' in dir():
        logger = setup_logger('eval_psych8k_complete', log_file)
    else:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger('eval_psych8k_complete')


class Psych8kSimulator:
    """Simulator that uses the patient input from psych8k.csv"""
    
    def __init__(self, llm, patient_input: str, case_id: str):
        self.llm = llm
        self.patient_input = patient_input
        self.case_id = case_id
        self.all_interactions = []
        
        if logger:
            logger.info(f"Initialized simulator for case {case_id}")
    
    def create_response_function(self):
        """Create response function that logs everything"""
        
        def simulate_response(question: str) -> str:
            """Simulate response and log"""
            turn = len(self.all_interactions) + 1
            timestamp = datetime.now().isoformat()
            
            if logger:
                logger.info(f"Turn {turn}: Generating response for case {self.case_id}")
            
            # Build prompt
            prompt = self._build_simulation_prompt(question)
            
            # Generate response
            try:
                response = self.llm.generate(prompt, max_tokens=8296, temperature=0.7)
                response = response.strip()
                
                # Log this interaction
                interaction = {
                    'turn': turn,
                    'timestamp': timestamp,
                    'question': question,
                    'llm_response': response,
                    'prompt_to_llm': prompt,
                    'llm_model': self.llm.model_name,
                    'llm_provider': self.llm.provider.value
                }
                
                self.all_interactions.append(interaction)
                
                if logger:
                    logger.info(f"Turn {turn}: Response logged")
                return response
                
            except Exception as e:
                if logger:
                    logger.error(f"Error in turn {turn}: {e}")
                fallback = "I'm not sure how to answer that."
                
                interaction = {
                    'turn': turn,
                    'timestamp': timestamp,
                    'question': question,
                    'llm_response': fallback,
                    'error': str(e)
                }
                
                self.all_interactions.append(interaction)
                return fallback
        
        return simulate_response
    
    def _build_simulation_prompt(self, question: str) -> str:
        """Build prompt for LLM based on patient input"""
        recent_history = ""
        if self.all_interactions:
            recent_history = "\n\nRecent conversation:\n"
            for interaction in self.all_interactions[-3:]:
                recent_history += f"Q: {interaction['question']}\n"
                recent_history += f"A: {interaction['llm_response']}\n"
        
        prompt = f"""You are simulating a patient in a counseling session.

PATIENT BACKGROUND:
{self.patient_input}

Respond to the counselor's questions naturally and consistently based on the background above.
Stay in character and be consistent with the symptoms and experiences described.
{recent_history}

COUNSELOR'S QUESTION:
{question}

Respond naturally and authentically (1-3 sentences unless more detail is requested).

YOUR RESPONSE:"""
        
        return prompt
    
    def get_all_interactions(self) -> list:
        """Get complete interaction log"""
        return self.all_interactions


class DetailedAssessmentPipeline(AssessmentPipeline):
    """
    Extended pipeline that captures evaluation and scoring decisions
    """
    
    def __init__(self, llm, config):
        super().__init__(llm, config)
        self.detailed_log = []  # Detailed conversation log
    
    def assess_topic_detailed(self, topic_config: dict, user_response_fn: callable) -> dict:
        """
        Assess a topic with detailed logging of all agent decisions
        
        Returns dict with:
        - topic_name
        - conversations: list of Q&A with evaluation
        - final_score
        - summary
        - basis
        """
        if self.memory is None:
            raise RuntimeError("Session not started")
        
        topic_name = topic_config['name']
        if logger:
            logger.info(f"Assessing topic: {topic_name} (detailed)")
        
        self.memory.add_topic_node(topic_name)
        
        # Generate initial question
        question = self.question_agent.generate_initial_question(
            topic_config=topic_config,
            memory=self.memory
        )
        
        topic_log = {
            'topic_name': topic_name,
            'topic_config': {
                'description': topic_config['description'],
                'rating_criteria': topic_config['rating_criteria']
            },
            'conversations': [],
            'final_score': None,
            'summary': None,
            'basis': None
        }
        
        follow_up_count = 0
        
        # Multi-turn conversation loop
        while follow_up_count < self.max_follow_ups:
            if logger:
                logger.info(f"Turn {follow_up_count + 1} for topic '{topic_name}'")
            
            # Get user response
            answer = user_response_fn(question)
            
            # Add to memory
            self.memory.add_qa_pair(topic_name, question, answer)
            statement = self.memory.extract_information(answer)
            self.memory.add_statement(topic_name, statement)
            
            # Evaluate adequacy
            necessity_score = self.evaluation_agent.evaluate_adequacy(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=self.memory
            )
            
            # Log this conversation turn with evaluation
            turn_log = {
                'turn': follow_up_count + 1,
                'question_type': 'initial' if follow_up_count == 0 else 'follow_up',
                'question': question,
                'answer': answer,
                'extracted_info': statement.to_dict() if hasattr(statement, 'to_dict') else str(statement),
                'evaluation': {
                    'necessity_score': necessity_score,
                    'threshold': self.necessity_threshold,
                    'decision': 'sufficient' if necessity_score < self.necessity_threshold else 'need_follow_up'
                }
            }
            
            topic_log['conversations'].append(turn_log)
            
            # Check if we need follow-up
            if necessity_score < self.necessity_threshold:
                if logger:
                    logger.info(f"Adequate information collected (necessity: {necessity_score})")
                break
            
            follow_up_count += 1
            
            if follow_up_count >= self.max_follow_ups:
                if logger:
                    logger.info(f"Max follow-ups reached ({self.max_follow_ups})")
                break
            
            # Generate follow-up question
            question = self.question_agent.generate_followup_question(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=self.memory
            )
        
        # Score the topic
        score, summary, basis = self.scoring_agent.score_topic(
            topic_config=topic_config,
            topic_name=topic_name,
            memory=self.memory
        )
        
        self.memory.update_topic_score(topic_name, score, summary, basis)
        
        # Add final scoring to log
        topic_log['final_score'] = score
        topic_log['summary'] = summary
        topic_log['basis'] = basis
        topic_log['total_turns'] = len(topic_log['conversations'])
        
        self.detailed_log.append(topic_log)
        
        return topic_log


def evaluate_complete_sample(
    case_data: dict,
    llm: any,
    config: dict,
    output_dir: str
) -> dict:
    """
    Complete evaluation: Assessment + Therapy
    
    Args:
        case_data: Dict with case_id, input, output, instruction
        llm: LLM orchestrator
        config: Configuration dict
        output_dir: Output directory
        
    Returns:
        Complete results with both assessment and therapy evaluation
    """
    
    case_id = case_data['case_id']
    patient_input = case_data['input']
    expected_output = case_data['output']
    instruction = case_data['instruction']
    
    print(f"\n{'='*70}")
    print(f"Evaluating Case ID: {case_id}")
    print(f"{'='*70}\n")
    
    # ========== PART 1: ASSESSMENT ==========
    print("PART 1: Assessment Evaluation")
    print("-" * 70)
    
    # Initialize pipeline
    pipeline = DetailedAssessmentPipeline(llm, config)
    
    # Create simulator
    simulator = Psych8kSimulator(
        llm=llm,
        patient_input=patient_input,
        case_id=str(case_id)
    )
    response_function = simulator.create_response_function()
    
    # Start session
    pipeline.start_session(
        user_id=str(case_id),
        user_info={}
    )
    
    start_time = datetime.now()
    
    # Assess each topic
    all_topics = []
    total_score = 0
    
    for i, topic_config in enumerate(pipeline.topics, 1):
        print(f"  [{i}/{len(pipeline.topics)}] Assessing: {topic_config['name']}")
        
        topic_result = pipeline.assess_topic_detailed(
            topic_config=topic_config,
            user_response_fn=response_function
        )
        
        all_topics.append(topic_result)
        total_score += topic_result.get('final_score', 0)
        
        print(f"    Score: {topic_result.get('final_score', 0)}")
    
    assessment_duration = (datetime.now() - start_time).total_seconds()
    
    # Determine classification
    if total_score <= 4:
        classification = "None/Minimal"
    elif total_score <= 9:
        classification = "Mild"
    elif total_score <= 14:
        classification = "Moderate"
    elif total_score <= 19:
        classification = "Moderately Severe"
    else:
        classification = "Severe"
    
    print(f"\n  Assessment Complete!")
    print(f"  Total Score: {total_score}")
    print(f"  Classification: {classification}")
    print(f"  Duration: {assessment_duration:.1f}s\n")
    
    # ========== PART 2: THERAPY RESPONSE ==========
    print("PART 2: Therapy Response Generation")
    print("-" * 70)
    
    therapy_start = datetime.now()
    
    # Generate therapy response
    print("  Generating counseling response...")
    generated_response = generate_therapy_response(
        llm=llm,
        patient_input=patient_input,
        assessment_results={
            'total_score': total_score,
            'classification': classification,
            'topics': all_topics
        },
        conversation_history=simulator.get_all_interactions()
    )
    
    print(f"  Generated response ({len(generated_response)} chars)")
    
    # ========== PART 3: THERAPY EVALUATION ==========
    print("\nPART 3: Therapy Quality Evaluation")
    print("-" * 70)
    
    # Calculate similarity metrics
    print("  Computing similarity metrics...")
    similarity_metrics = calculate_similarity_metrics(
        generated=generated_response,
        expected=expected_output
    )
    
    # LLM-based quality evaluation
    print("  Evaluating therapy quality with LLM judge...")
    quality_evaluation = evaluate_therapy_quality(
        llm=llm,
        generated_response=generated_response,
        expected_response=expected_output,
        patient_input=patient_input,
        assessment_results={
            'total_score': total_score,
            'classification': classification
        }
    )
    
    therapy_duration = (datetime.now() - therapy_start).total_seconds()
    total_duration = assessment_duration + therapy_duration
    
    print(f"\n  Therapy Evaluation Complete!")
    print(f"  Overall Quality: {quality_evaluation.get('overall_quality', 'N/A')}/10")
    print(f"  Duration: {therapy_duration:.1f}s\n")
    
    # ========== BUILD COMPLETE RESULT ==========
    result = {
        'metadata': {
            'case_id': case_id,
            'evaluation_timestamp': datetime.now().isoformat(),
            'duration_seconds': {
                'assessment': assessment_duration,
                'therapy': therapy_duration,
                'total': total_duration
            },
            'llm_model': llm.model_name,
            'llm_provider': llm.provider.value,
            'assessment_scale': config['assessment_scale']['name']
        },
        
        'case_data': {
            'case_id': case_id,
            'patient_input': patient_input,
            'expected_counselor_output': expected_output,
            'instruction': instruction
        },
        
        # Assessment Results
        'assessment_results': {
            'total_score': total_score,
            'classification': classification,
            'topics': all_topics,
            'total_turns': len(simulator.get_all_interactions()),
            'conversation_history': simulator.get_all_interactions()
        },
        
        # Therapy Results
        'therapy_results': {
            'generated_response': generated_response,
            'expected_response': expected_output,
            
            'similarity_metrics': similarity_metrics,
            
            'quality_evaluation': quality_evaluation
        },
        
        # Overall Performance
        'overall_performance': {
            'assessment_score': total_score,
            'assessment_classification': classification,
            'therapy_overall_quality': quality_evaluation.get('overall_quality', 0),
            'therapy_similarity': similarity_metrics.get('semantic_similarity', 0)
        }
    }
    
    # Save individual result
    individual_dir = os.path.join(output_dir, 'individual_result')
    os.makedirs(individual_dir, exist_ok=True)
    
    output_path = os.path.join(individual_dir, f"{case_id}.json")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"{'='*70}")
    print(f"✓ Complete evaluation saved: {output_path}")
    print(f"{'='*70}\n")
    
    return result


def evaluate_random_samples(
    csv_path: str,
    config_path: str,
    output_dir: str,
    num_samples: int,
    random_seed: int = None
):
    """
    Evaluate random samples from psych8k.csv
    
    Args:
        csv_path: Path to psych8k.csv
        config_path: Path to config.yml
        output_dir: Output directory
        num_samples: Number of random samples to evaluate
        random_seed: Random seed for reproducibility
    """
    
    print("\n" + "="*70)
    print("Psych8k Complete Evaluation (Assessment + Therapy)")
    print("="*70)
    print(f"CSV: {csv_path}")
    print(f"Samples: {num_samples} (randomly selected)")
    print(f"Output: {output_dir}/individual_result/")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
    print("="*70 + "\n")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"✓ Random seed set to {random_seed}\n")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize LLM
    llm = LLMOrchestrator(config)
    print("✓ LLM initialized\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    total_cases = len(df)
    print(f"✓ Loaded {total_cases} cases from CSV\n")
    
    # Validate num_samples
    if num_samples > total_cases:
        print(f"⚠ Warning: Requested {num_samples} samples but only {total_cases} available.")
        num_samples = total_cases
        print(f"  Adjusted to {num_samples} samples.\n")
    
    # Randomly select samples
    print(f"Randomly selecting {num_samples} samples...")
    selected_indices = random.sample(range(total_cases), num_samples)
    selected_indices.sort()  # Sort for easier tracking
    
    print(f"✓ Selected case IDs: {[df.iloc[idx]['case_id'] for idx in selected_indices[:10]]}" + 
          ("..." if num_samples > 10 else ""))
    print()
    
    # Process each selected sample
    results_summary = []
    
    for i, idx in enumerate(selected_indices, 1):
        row = df.iloc[idx]
        
        case_data = {
            'case_id': int(row['case_id']),
            'input': str(row['input']),
            'output': str(row['output']),
            'instruction': str(row['instruction'])
        }
        
        print(f"\n{'#'*70}")
        print(f"# Sample {i}/{num_samples} - Case ID: {case_data['case_id']}")
        print(f"{'#'*70}")
        
        try:
            result = evaluate_complete_sample(
                case_data=case_data,
                llm=llm,
                config=config,
                output_dir=output_dir
            )
            
            results_summary.append({
                'case_id': case_data['case_id'],
                'status': 'success',
                'assessment': {
                    'total_score': result['assessment_results']['total_score'],
                    'classification': result['assessment_results']['classification']
                },
                'therapy': {
                    'overall_quality': result['therapy_results']['quality_evaluation'].get('overall_quality', 0),
                    'semantic_similarity': result['therapy_results']['similarity_metrics'].get('semantic_similarity', 0)
                }
            })
            
        except Exception as e:
            print(f"\n✗ Error processing case {case_data['case_id']}: {e}\n")
            if logger:
                logger.error(f"Error on case {case_data['case_id']}: {e}", exc_info=True)
            
            results_summary.append({
                'case_id': case_data['case_id'],
                'status': 'error',
                'error': str(e)
            })
    
    # Save summary
    summary = {
        'evaluation_info': {
            'total_cases_in_dataset': total_cases,
            'num_samples_evaluated': num_samples,
            'selected_indices': selected_indices,
            'random_seed': random_seed,
            'timestamp': datetime.now().isoformat()
        },
        'results': results_summary
    }
    
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    successful = [r for r in results_summary if r['status'] == 'success']
    failed = [r for r in results_summary if r['status'] == 'error']
    
    print(f"Total Evaluated: {num_samples}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_quality = sum(r['therapy']['overall_quality'] for r in successful) / len(successful)
        avg_similarity = sum(r['therapy']['semantic_similarity'] for r in successful) / len(successful)
        
        print(f"\nAverage Therapy Quality: {avg_quality:.2f}/10")
        print(f"Average Semantic Similarity: {avg_similarity:.3f}")
    
    print(f"\nResults:")
    print(f"  Individual files: {output_dir}/individual_result/")
    print(f"  Summary: {summary_path}")
    print("="*70 + "\n")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='Complete Psych8k Evaluation: Assessment + Therapy'
    )
    parser.add_argument('--csv-path', type=str,
                       default='data/psych8k.csv',
                       help='Path to psych8k.csv')
    parser.add_argument('--config', type=str, 
                       default='config.yml',
                       help='Path to config.yml')
    parser.add_argument('--output-dir', type=str, 
                       default='results/psych8k',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, 
                       required=True,
                       help='Number of random samples to evaluate')
    parser.add_argument('--random-seed', type=int,
                       default=421,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        evaluate_random_samples(
            csv_path=args.csv_path,
            config_path=args.config,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            random_seed=args.random_seed
        )
        
        print("\n✓ Complete evaluation finished successfully!\n")
        
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