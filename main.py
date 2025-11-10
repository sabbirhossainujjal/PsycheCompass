"""
AgentMental Framework - Main Entry Point

This module provides the main interface for running mental health assessments
using the multi-agent framework with AutoGen.
"""

import yaml
import json
import os
from datetime import datetime
from typing import Dict, Optional

from utils.logger import setup_logger
from utils.memory import TreeMemory
from utils.llm import LLMOrchestrator
from utils.agents import (
    QuestionGeneratorAgent,
    EvaluationAgent,
    ScoringAgent,
    UpdatingAgent
)

# Setup logger
logger = setup_logger('main', 'logs/main.log')


class AgentMental:
    """Main multi-agent framework for mental health assessment"""
    
    def __init__(self, config_path: str = "config.yml"):
        logger.info("Initializing AgentMental framework")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize LLM Orchestrator
        self.llm = LLMOrchestrator(self.config)
        
        # Load assessment scale
        self.scale = self.config['assessment_scale']
        self.scale_name = self.scale['name']
        self.topics = self.scale['topics']
        
        # Agent parameters
        self.max_follow_ups = self.config['agent_params']['max_follow_ups']
        self.necessity_threshold = self.config['agent_params']['necessity_threshold']
        
        # Initialize agents
        logger.info("Initializing agents")
        self._initialize_agents()
        
        # Memory
        self.memory: Optional[TreeMemory] = None
        
        logger.info("AgentMental framework initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all four agents"""
        self.question_agent = QuestionGeneratorAgent(
            llm=self.llm,
            config=self.config
        )
        
        self.evaluation_agent = EvaluationAgent(
            llm=self.llm,
            config=self.config
        )
        
        self.scoring_agent = ScoringAgent(
            llm=self.llm,
            config=self.config
        )
        
        self.updating_agent = UpdatingAgent(
            llm=self.llm,
            config=self.config
        )
        
        logger.info("All agents initialized")
    
    def start_assessment(self, user_id: str, user_info: dict = None):
        """Start a new assessment session"""
        logger.info(f"Starting assessment for user: {user_id}")
        
        self.memory = TreeMemory(user_id)
        
        if user_info:
            self.memory.user.age = user_info.get('age')
            self.memory.user.occupation = user_info.get('occupation')
            self.memory.user.gender = user_info.get('gender')
            logger.info(f"User info set: {user_info}")
        
        print(f"\n{'='*60}")
        print(f"Starting {self.scale_name} Assessment")
        print(f"User ID: {user_id}")
        print(f"{'='*60}\n")
    
    def assess_topic(self, topic_config: dict, simulate_user: bool = True) -> int:
        """Assess a single topic through multi-turn dialogue"""
        topic_name = topic_config['name']
        logger.info(f"Starting assessment for topic: {topic_name}")
        
        self.memory.add_topic_node(topic_name)
        
        print(f"\n--- Topic: {topic_name} ---")
        
        # Generate initial question using QuestionGeneratorAgent
        logger.info(f"Generating initial question for {topic_name}")
        question = self.question_agent.generate_initial_question(
            topic_config=topic_config,
            memory=self.memory
        )
        
        print(f"Agent: {question}")
        
        follow_up_count = 0
        
        while follow_up_count < self.max_follow_ups:
            # Get user response
            if simulate_user:
                answer = self._simulate_user_response(topic_config, question, follow_up_count)
            else:
                answer = input("User: ")
            
            print(f"User: {answer}")
            logger.info(f"User response received (follow-up {follow_up_count})")
            
            # Store Q&A in memory
            self.memory.add_qa_pair(topic_name, question, answer)
            
            # Extract information and create statement node
            statement = self.memory.extract_information(answer)
            self.memory.add_statement(topic_name, statement)
            logger.info(f"Statement extracted and stored: {statement.to_dict()}")
            
            # Evaluate response adequacy using EvaluationAgent
            logger.info(f"Evaluating response adequacy for {topic_name}")
            necessity_score = self.evaluation_agent.evaluate_adequacy(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=self.memory
            )
            
            logger.info(f"Necessity score: {necessity_score}")
            
            # Check if follow-up needed
            if necessity_score < self.necessity_threshold:
                print(f"[Evaluation: Adequate information collected]\n")
                logger.info(f"Adequate information collected for {topic_name}")
                break
            
            follow_up_count += 1
            
            # Check if we've reached max follow-ups
            if follow_up_count >= self.max_follow_ups:
                logger.info(f"Max follow-ups reached for {topic_name}")
                break
            
            # Generate follow-up question using QuestionGeneratorAgent
            logger.info(f"Generating follow-up question {follow_up_count} for {topic_name}")
            question = self.question_agent.generate_followup_question(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=self.memory
            )
            
            print(f"\nAgent: {question}")
        
        # Score the topic using ScoringAgent
        logger.info(f"Scoring topic: {topic_name}")
        score, summary, basis = self.scoring_agent.score_topic(
            topic_config=topic_config,
            topic_name=topic_name,
            memory=self.memory
        )
        
        self.memory.update_topic_score(topic_name, score, summary, basis)
        logger.info(f"Topic '{topic_name}' scored: {score}")
        
        print(f"\n[Topic '{topic_name}' scored: {score}]")
        print(f"Basis: {basis[:150]}...\n")
        
        return score
    
    def _simulate_user_response(self, topic_config: dict, question: str, turn: int) -> str:
        """Simulate user responses for demonstration"""
        logger.debug(f"Simulating user response for turn {turn}")
        
        # Simple simulation based on topic and turn
        responses = {
            0: [
                "Yes, I've been experiencing that.",
                "I have noticed some issues with this.",
                "It's been affecting me recently."
            ],
            1: [
                "It happens quite often, maybe several times a week.",
                "Fairly frequently, it's hard to manage.",
                "More than I'd like, it's becoming a concern."
            ],
            2: [
                "It really impacts my daily life and work performance.",
                "Yes, it makes things difficult and exhausting.",
                "It's been making everything harder to handle."
            ]
        }
        
        import random
        return random.choice(responses.get(turn, ["I'm not sure how to describe it further."]))
    
    def run_full_assessment(
        self, 
        user_id: str, 
        user_info: dict = None, 
        simulate: bool = True
    ) -> Dict:
        """Run complete assessment across all topics"""
        logger.info(f"Starting full assessment for user {user_id}")
        
        self.start_assessment(user_id, user_info)
        
        total_score = 0
        
        # Assess each topic
        for i, topic_config in enumerate(self.topics, 1):
            logger.info(f"Assessing topic {i}/{len(self.topics)}: {topic_config['name']}")
            score = self.assess_topic(topic_config, simulate_user=simulate)
            total_score += score
        
        # Generate final report using UpdatingAgent
        logger.info("Generating final report")
        print(f"\n{'='*60}")
        print(f"ASSESSMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Total Score: {total_score}")
        
        # Determine classification
        threshold = self.scale.get('threshold', 10)
        classification = "Depression" if total_score >= threshold else "No Depression"
        print(f"Classification: {classification}\n")
        
        logger.info(f"Assessment complete. Total score: {total_score}, Classification: {classification}")
        
        final_report = self.updating_agent.generate_report(
            memory=self.memory,
            total_score=total_score,
            scale_name=self.scale_name
        )
        
        print("Final Report:")
        print(final_report)
        
        results = {
            'user_id': user_id,
            'total_score': total_score,
            'classification': classification,
            'topics': {name: node.to_dict() for name, node in self.memory.topics.items()},
            'report': final_report,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Full assessment completed successfully")
        
        # Save results if configured
        if self.config.get('output', {}).get('save_results', False):
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save assessment results to file"""
        output_dir = self.config.get('output', {}).get('output_directory', './assessments')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/assessment_{results['user_id']}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
            print(f"\n✓ Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"\n✗ Failed to save results: {e}")


def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("AgentMental Framework Starting")
    logger.info("="*60)
    
    try:
        # Initialize the framework
        agent_mental = AgentMental("config.yml")
        
        # Run assessment with simulated user
        user_info = {
            'age': 35,
            'occupation': 'Software Engineer',
            'gender': 'Male'
        }
        
        results = agent_mental.run_full_assessment(
            user_id="user_001",
            user_info=user_info,
            simulate=True
        )
        
        print("\n" + "="*60)
        print("Assessment Results Summary:")
        print(f"User ID: {results['user_id']}")
        print(f"Total Score: {results['total_score']}")
        print(f"Classification: {results['classification']}")
        print("="*60)
        
        logger.info("AgentMental Framework completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
