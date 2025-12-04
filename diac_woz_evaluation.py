
import pandas as pd
import yaml
import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.assessment_pipeline import AssessmentPipeline
from utils.llm import LLMOrchestrator
from utils.logger import setup_logger
from utils.memory import TreeMemory

logger = setup_logger('assessment_evaluation_full', 'logs/assessment_evaluation_full.log')


class DetailedJSONSimulator:
    """
    Simulator that tracks everything for JSON export
    """
    
    def __init__(self, llm: LLMOrchestrator, interaction_context: str, participant_id: str):
        self.llm = llm
        self.interaction_context = interaction_context
        self.participant_id = participant_id
        self.all_interactions = []  # Complete log
        
        logger.info(f"Initialized detailed JSON simulator for participant {participant_id}")
    
    def create_response_function(self):
        """Create response function that logs everything"""
        
        def simulate_response(question: str) -> str:
            """Simulate response and log"""
            turn = len(self.all_interactions) + 1
            timestamp = datetime.now().isoformat()
            
            logger.info(f"Turn {turn}: Generating response")
            
            # Build prompt
            prompt = self._build_simulation_prompt(question)
            
            # Generate response
            try:
                response = self.llm.generate(prompt, max_tokens=12288, temperature=0.7)
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
                
                logger.info(f"Turn {turn}: Response logged")
                return response
                
            except Exception as e:
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
        """Build prompt for LLM"""
        recent_history = ""
        if self.all_interactions:
            recent_history = "\n\nRecent conversation:\n"
            for interaction in self.all_interactions[-3:]:
                recent_history += f"Q: {interaction['question']}\n"
                recent_history += f"A: {interaction['llm_response']}\n"
        
        prompt = f"""You are simulating a person being interviewed for a mental health assessment.

IMPORTANT: Base your responses on the context from your previous interview. Be consistent.

===== YOUR PREVIOUS INTERVIEW CONTEXT =====
{self.interaction_context}
===========================================
{recent_history}

NEW ASSESSMENT QUESTION:
{question}

Respond naturally and consistently (1-3 sentences unless more detail requested).

YOUR RESPONSE:"""
        
        return prompt
    
    def get_all_interactions(self) -> list:
        """Get complete interaction log"""
        return self.all_interactions


def load_participant_from_csv(csv_path: str, participant_id: str) -> dict:
    """Load participant data from CSV"""
    logger.info(f"Loading participant {participant_id}")
    
    df = pd.read_csv(csv_path)
    participant_row = df[df['Participant_ID'].astype(str) == str(participant_id)]
    
    if participant_row.empty:
        raise ValueError(f"Participant {participant_id} not found")
    
    row = participant_row.iloc[0]
    
    participant_data = {
        'participant_id': str(row['Participant_ID']),
        'phq8_score': int(row['PHQ8_Score']),
        'phq8_binary': int(row['PHQ8_Binary']),
        'depression_label': 'Distressed' if row['PHQ8_Binary'] == 1 else 'Normal',
        'gender': row['Gender'] if pd.notna(row['Gender']) else 'Unknown',
        'interaction_context': str(row['interaction']),
        'topic_scores': {
            'Loss of Interest': int(row['PHQ8_NoInterest']) if pd.notna(row['PHQ8_NoInterest']) else None,
            'Depressed Mood': int(row['PHQ8_Depressed']) if pd.notna(row['PHQ8_Depressed']) else None,
            'Sleep Problems': int(row['PHQ8_Sleep']) if pd.notna(row['PHQ8_Sleep']) else None,
            'Fatigue or Low Energy': int(row['PHQ8_Tired']) if pd.notna(row['PHQ8_Tired']) else None,
            'Appetite or Weight Changes': int(row['PHQ8_Appetite']) if pd.notna(row['PHQ8_Appetite']) else None,
            'Low Self-Worth': int(row['PHQ8_Failure']) if pd.notna(row['PHQ8_Failure']) else None,
            'Concentration Difficulties': int(row['PHQ8_Concentrating']) if pd.notna(row['PHQ8_Concentrating']) else None,
            'Psychomotor Changes': int(row['PHQ8_Moving']) if pd.notna(row['PHQ8_Moving']) else None,
        }
    }
    
    logger.info(f"Loaded participant {participant_id}: Score={participant_data['phq8_score']}")
    return participant_data


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
                'extracted_info': statement.to_dict(),
                'evaluation': {
                    'necessity_score': necessity_score,
                    'threshold': self.necessity_threshold,
                    'decision': 'sufficient' if necessity_score < self.necessity_threshold else 'need_follow_up'
                }
            }
            
            topic_log['conversations'].append(turn_log)
            
            # Check if we need follow-up
            if necessity_score < self.necessity_threshold:
                logger.info(f"Adequate information collected (necessity: {necessity_score})")
                break
            
            follow_up_count += 1
            
            if follow_up_count >= self.max_follow_ups:
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


def evaluate_single_participant_detailed(
    participant_id: str,
    csv_path: str = 'data/diac_woz/dev_split_all.csv',
    config_path: str = 'config.yml',
    output_dir: str = 'data/results'
):
    """
    Evaluate participant with detailed JSON logging
    """
    
    print("\n" + "="*70)
    print("Detailed JSON Evaluation")
    print("="*70)
    print(f"Participant ID: {participant_id}")
    print(f"Output: {output_dir}/{participant_id}.json")
    print("="*70 + "\n")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize LLM
    llm = LLMOrchestrator(config)
    print("✓ LLM initialized")
    
    # Load participant
    participant_data = load_participant_from_csv(csv_path, participant_id)
    print(f"✓ Participant loaded: True Score = {participant_data['phq8_score']}")
    print()
    
    # Initialize detailed pipeline
    pipeline = DetailedAssessmentPipeline(llm, config)
    print("✓ Detailed assessment pipeline initialized")
    print()
    
    # Create detailed simulator
    simulator = DetailedJSONSimulator(
        llm=llm,
        interaction_context=participant_data['interaction_context'],
        participant_id=participant_id
    )
    response_function = simulator.create_response_function()
    print("✓ Simulator ready")
    print()
    
    # Start session
    pipeline.start_session(
        user_id=participant_id,
        user_info={'gender': participant_data['gender']}
    )
    
    print("-"*70)
    print("Running Assessment...")
    print("-"*70)
    print()
    
    start_time = datetime.now()
    
    # Assess each topic with detailed logging
    all_topics = []
    total_score = 0
    
    for i, topic_config in enumerate(pipeline.topics, 1):
        print(f"[{i}/{len(pipeline.topics)}] Assessing: {topic_config['name']}")
        
        topic_result = pipeline.assess_topic_detailed(
            topic_config=topic_config,
            user_response_fn=response_function
        )
        
        all_topics.append(topic_result)
        total_score += topic_result['final_score']
        
        print(f"  Score: {topic_result['final_score']}, Turns: {topic_result['total_turns']}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print()
    print("-"*70)
    print("Assessment Complete!")
    print("-"*70)
    print()
    
    # Calculate classification
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
    
    # Build complete JSON structure
    complete_data = {
        'metadata': {
            'participant_id': participant_id,
            'evaluation_timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'llm_model': llm.model_name,
            'llm_provider': llm.provider.value,
            'assessment_scale': config['assessment_scale']['name']
        },
        
        'ground_truth': {
            'phq8_total_score': participant_data['phq8_score'],
            'phq8_binary': participant_data['phq8_binary'],
            'depression_label': participant_data['depression_label'],
            'gender': participant_data['gender'],
            'per_topic_scores': participant_data['topic_scores'],
            'daic_woz_interview': participant_data['interaction_context']
        },
        
        'predictions': {
            'phq8_total_score': total_score,
            'classification': classification,
            'per_topic_results': all_topics
        },
        
        'evaluation_metrics': {
            'total_error': total_score - participant_data['phq8_score'],
            'absolute_error': abs(total_score - participant_data['phq8_score']),
            'classification_correct': (total_score >= 10) == (participant_data['phq8_score'] >= 10),
            'per_topic_errors': []
        },
        
        'simulator_interactions': simulator.get_all_interactions(),
        
        'detailed_conversation_flow': all_topics
    }
    
    # Calculate per-topic errors
    for topic_result in all_topics:
        topic_name = topic_result['topic_name']
        predicted = topic_result['final_score']
        true = participant_data['topic_scores'].get(topic_name)
        
        if true is not None:
            complete_data['evaluation_metrics']['per_topic_errors'].append({
                'topic': topic_name,
                'predicted': predicted,
                'true': true,
                'error': predicted - true,
                'absolute_error': abs(predicted - true)
            })
    
    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{participant_id}.json")
    
    with open(output_path, 'w') as f:
        json.dump(complete_data, f, indent=2, default=str)
    
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Predicted Score: {total_score}/24")
    print(f"True Score:      {participant_data['phq8_score']}/24")
    print(f"Error:           {total_score - participant_data['phq8_score']:+d}")
    print(f"Absolute Error:  {abs(total_score - participant_data['phq8_score'])}")
    print(f"Duration:        {duration:.1f}s")
    print(f"Total Turns:     {len(simulator.get_all_interactions())}")
    print("="*70)
    print()
    print(f"✓ Detailed JSON saved: {output_path}")
    print()
    
    return complete_data, output_path


def main():
    """Main function"""
    
    diac_woz_data = 'data/diac_woz_dev_split_all.csv'
    import pandas as pd
    df = pd.read_csv(diac_woz_data)
    participant_ids = list(df['Participant_ID'])
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate participant with detailed JSON logging'
    )
    # parser.add_argument('--participant-id', type=str, required=True)
    parser.add_argument('--csv-path', type=str,
                       default=diac_woz_data)
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--output-dir', type=str, default='./results/diac_woz')
    
    args = parser.parse_args()
    
    for participant_id in participant_ids:
        try:
            data, output_path = evaluate_single_participant_detailed(
                participant_id=participant_id,
                csv_path=args.csv_path,
                config_path=args.config,
                output_dir=args.output_dir
            )
            
            print("✓ Evaluation completed successfully!")
            print(f"\nJSON file: {output_path}")
            print("\nTo view:")
            print(f"  cat {output_path} | jq '.'")
            print()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Evaluation interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()