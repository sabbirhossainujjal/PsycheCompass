import yaml
import json
from typing import Dict, List, Optional, Tuple

from utils.memory import TreeMemory, StatementNode
from utils.llm import LLMInterface
from utils.logger import get_logger

# ============================================================================ 
# Agent System
# ============================================================================ 

from dotenv import load_dotenv

class AgentMental:
    """Main multi-agent framework for mental health assessment"""
    
    def __init__(self, config_path: str = "config.yml"):
        load_dotenv()
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM
        self.llm = LLMInterface(self.config)
        
        # Load assessment scale
        self.scale = self.config['assessment_scale']
        self.scale_name = self.scale['name']
        self.topics = self.scale['topics']
        
        # Agent parameters
        self.max_follow_ups = self.config['agent_params']['max_follow_ups']
        self.necessity_threshold = self.config['agent_params']['necessity_threshold']
        
        # Memory
        self.memory: Optional[TreeMemory] = None

        # Loggers
        self.q_logger = get_logger("question_generator", "logs/question_generator.log")
        self.eval_logger = get_logger("evaluation", "logs/evaluation.log")
        self.scoring_logger = get_logger("scoring", "logs/scoring.log")
        self.updating_logger = get_logger("updating", "logs/updating.log")
    
    def start_assessment(self, user_id: str, user_info: dict = None):
        """Start a new assessment session"""
        self.memory = TreeMemory(user_id)
        
        if user_info:
            self.memory.user.age = user_info.get('age', None)
            self.memory.user.occupation = user_info.get('occupation', None)
            self.memory.user.gender = user_info.get('gender', None)
        
        print(f"\n{'='*60}")
        print(f"Starting {self.scale_name} Assessment")
        print(f"User ID: {user_id}")
        print(f"{'='*60}\n")
    
    def assess_topic(self, topic_config: dict, simulate_user: bool = True) -> int:
        """Assess a single topic through multi-turn dialogue"""
        topic_name = topic_config['name']
        self.memory.add_topic_node(topic_name)
        
        print(f"\n--- Topic: {topic_name} ---")
        
        # Generate initial question (AGq)
        question = self._agent_question_generator(topic_config, is_initial=True)
        print(f"Agent: {question}")
        
        follow_up_count = 0
        
        while follow_up_count < self.max_follow_ups:
            # Get user response
            if simulate_user:
                answer = self._simulate_user_response(topic_config, question, follow_up_count)
            else:
                answer = input("User: ")
            
            print(f"User: {answer}")
            
            # Store Q&A
            self.memory.add_qa_pair(topic_name, question, answer)
            
            # Extract information and create statement node
            statement = self._extract_information(answer)
            self.memory.add_statement(topic_name, statement)
            
            # Evaluate response adequacy (AGev)
            necessity_score = self._agent_evaluation(topic_config, topic_name)
            
            # Check if follow-up needed
            if necessity_score < self.necessity_threshold:
                print(f"[Evaluation: Adequate information collected]\n")
                break
            
            # Generate follow-up question (AGq)
            question = self._agent_question_generator(topic_config, is_initial=False)
            print(f"\nAgent: {question}")
            follow_up_count += 1
        
        # Score the topic (AGs)
        score, summary, basis = self._agent_scoring(topic_config, topic_name)
        self.memory.update_topic_score(topic_name, score, summary, basis)
        
        print(f"\n[Topic '{topic_name}' scored: {score}]")
        print(f"Basis: {basis}\n")
        
        return score
    
    def _agent_question_generator(self, topic_config: dict, is_initial: bool) -> str:
        """AGq: Question Generator Agent"""
        prompt_template = self.config['agent_prompts']['question_generator']
        
        if is_initial:
            prompt = prompt_template['initial'].format(
                topic=topic_config['name'],
                description=topic_config['description'],
                context=self.memory.get_context_summary(topic_config['name'])
            )
        else:
            prompt = prompt_template['followup'].format(
                topic=topic_config['name'],
                conversation=self.memory.get_current_topic_context(topic_config['name']),
                context=self.memory.get_context_summary(topic_config['name'])
            )
        
        question = self.llm.generate(prompt)
        self.q_logger.info(f"Generated question for topic {topic_config['name']}: {question}")
        return question
    
    def _agent_evaluation(self, topic_config: dict, topic_name: str) -> float:
        """AGev: Evaluation Agent - Returns necessity score (0-2)"""
        prompt_template = self.config['agent_prompts']['evaluation']
        
        prompt = prompt_template.format(
            topic=topic_config['name'],
            conversation=self.memory.get_current_topic_context(topic_name),
            rating_criteria=json.dumps(topic_config['rating_criteria'], indent=2)
        )
        
        response = self.llm.generate(prompt)
        self.eval_logger.info(f"Evaluation for topic {topic_name}: {response}")
        
        # Parse necessity score from response
        try:
            # Extract number from response
            for word in response.split():
                if word.replace('.', '').isdigit():
                    score = float(word)
                    if 0 <= score <= 2:
                        return score
            return 1.0  # Default if parsing fails
        except:
            return 1.0
    
    def _agent_scoring(self, topic_config: dict, topic_name: str) -> Tuple[int, str, str]:
        """AGs: Scoring Agent"""
        prompt_template = self.config['agent_prompts']['scoring']
        
        prompt = prompt_template.format(
            topic=topic_config['name'],
            conversation=self.memory.get_current_topic_context(topic_name),
            rating_criteria=json.dumps(topic_config['rating_criteria'], indent=2)
        )
        
        response = self.llm.generate(prompt)
        self.scoring_logger.info(f"Scoring for topic {topic_name}: {response}")
        
        # Parse score, summary, and basis
        score = self._parse_score(response, topic_config)
        summary = self._parse_summary(response)
        basis = response  # Full response as basis
        
        return score, summary, basis
    
    def _agent_updating(self, total_score: int) -> str:
        """AGu: Updating Agent - Generate final report"""
        prompt_template = self.config['agent_prompts']['updating']
        
        # Compile all topic information
        topics_summary = ""
        for topic_name, topic_node in self.memory.topics.items():
            topics_summary += f"\n{topic_name}:\n"
            topics_summary += f"  Score: {topic_node.score}\n"
            topics_summary += f"  Summary: {topic_node.summary}\n"
        
        prompt = prompt_template.format(
            scale_name=self.scale_name,
            topics_summary=topics_summary,
            total_score=total_score,
            user_info=json.dumps(self.memory.user.to_dict(), indent=2)
        )
        
        report = self.llm.generate(prompt)
        self.updating_logger.info(f"Generated final report: {report}")
        return report
    
    def _extract_information(self, answer: str) -> StatementNode:
        """Extract key information from user answer"""
        # Simple keyword-based extraction (can be enhanced with LLM)
        statement = StatementNode()
        
        answer_lower = answer.lower()
        
        # Emotion keywords
        if any(word in answer_lower for word in ['sad', 'depressed', 'down', 'hopeless']):
            statement.emotion = "negative"
        elif any(word in answer_lower for word in ['anxious', 'worried', 'nervous']):
            statement.emotion = "anxious"
        
        # Frequency keywords
        if any(word in answer_lower for word in ['always', 'constantly', 'every day']):
            statement.frequency = "high"
        elif any(word in answer_lower for word in ['often', 'frequently', 'regularly']):
            statement.frequency = "medium"
        elif any(word in answer_lower for word in ['sometimes', 'occasionally']):
            statement.frequency = "low"
        
        # Duration keywords
        if any(word in answer_lower for word in ['weeks', 'months', 'long time']):
            statement.duration = "extended"
        elif any(word in answer_lower for word in ['days', 'recently']):
            statement.duration = "short"
        
        # Impact keywords
        if any(word in answer_lower for word in ['difficult', 'hard', 'struggle', 'affect']):
            statement.impact = "significant"
        
        return statement
    
    def _parse_score(self, response: str, topic_config: dict) -> int:
        """Parse score from agent response"""
        # Look for score in response
        for i in range(len(topic_config['rating_criteria'])):
            if str(i) in response[:50]:  # Check first 50 chars
                return i
        return 0
    
    def _parse_summary(self, response: str) -> str:
        """Parse summary from response"""
        # Take first 200 chars as summary
        return response[:200].strip()
    
    def _simulate_user_response(self, topic_config: dict, question: str, turn: int) -> str:
        """Simulate user responses for demonstration"""
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
    
    def run_full_assessment(self, user_id: str, user_info: dict = None, simulate: bool = True):
        """Run complete assessment across all topics"""
        self.start_assessment(user_id, user_info)
        
        total_score = 0
        
        # Assess each topic
        for topic_config in self.topics:
            score = self.assess_topic(topic_config, simulate_user=simulate)
            total_score += score
        
        # Generate final report (AGu)
        print(f"\n{'='*60}")
        print(f"ASSESSMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Total Score: {total_score}")
        
        # Determine classification
        threshold = self.scale.get('threshold', 10)
        classification = "Depression" if total_score >= threshold else "No Depression"
        print(f"Classification: {classification}\n")
        
        final_report = self._agent_updating(total_score)
        print("Final Report:")
        print(final_report)
        
        return {
            'total_score': total_score,
            'classification': classification,
            'topics': {name: node.to_dict() for name, node in self.memory.topics.items()},
            'report': final_report
        }
