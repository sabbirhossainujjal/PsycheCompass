import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from utils.logger import setup_logger
from utils.memory import TreeMemory
from utils.llm import LLMOrchestrator
from agents.assessment.question_generator import QuestionGeneratorAgent
from agents.assessment.evaluation_agent import EvaluationAgent
from agents.assessment.scoring_agent import ScoringAgent
from agents.assessment.updating_agent import UpdatingAgent


logger = setup_logger('assessment_pipeline', 'logs/assessment_pipeline.log')


class AssessmentPipeline:
    """
    Assessment Pipeline - Orchestrates PHQ-8 mental health assessment

    This pipeline:
    1. Generates initial questions for each topic
    2. Evaluates user responses for adequacy
    3. Generates follow-up questions when needed
    4. Scores each topic based on conversation
    5. Produces structured assessment output
    """

    def __init__(self, llm: LLMOrchestrator, config: Dict):
        logger.info("="*70)
        logger.info("Initializing Assessment Pipeline")
        logger.info("="*70)

        self.llm = llm
        self.config = config

        # Load configuration
        self.scale = config['assessment_scale']
        self.scale_name = self.scale['name']
        self.topics = self.scale['topics']

        agent_params = config['agent_params']
        self.max_follow_ups = agent_params['max_follow_ups']
        self.necessity_threshold = agent_params['necessity_threshold']

        logger.info("Initializing assessment agents...")
        self._initialize_agents()

        
        self.memory: Optional[TreeMemory] = None # Memory for each session

        logger.info(f"Assessment Pipeline initialized successfully")
        logger.info(f"Scale: {self.scale_name}")
        logger.info(f"Topics: {len(self.topics)}")
        logger.info(f"Max follow-ups per topic: {self.max_follow_ups}")
        logger.info("="*70)

    def _initialize_agents(self):
        """Initialize all assessment agents"""
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

        logger.info("All assessment agents initialized")

    def start_session(self, user_id: str, user_info: Optional[Dict] = None) -> TreeMemory:
        """
        Start a new assessment session

        Args:
            user_id: Unique user identifier
            user_info: Optional user information (age, occupation, gender)

        Returns:
            Initialized TreeMemory instance
        """
        logger.info("="*70)
        logger.info(f"Starting new assessment session")
        logger.info(f"User ID: {user_id}")
        logger.info("="*70)

        
        self.memory = TreeMemory(user_id) # new memory for this session

        if user_info:
            self.memory.user.age = user_info.get('age')
            self.memory.user.occupation = user_info.get('occupation')
            self.memory.user.gender = user_info.get('gender')
            logger.info(f"User info set: {user_info}")

        return self.memory

    def assess_topic(
        self,
        topic_config: Dict,
        user_response_fn: callable
    ) -> Tuple[int, str, str]:
        """
        Assess a single topic through multi-turn dialogue

        Args:
            topic_config: Configuration for the topic to assess
            user_response_fn: Function that takes a question and returns user's answer
                             Signature: fn(question: str) -> str

        Returns:
            Tuple of (score, summary, basis)
        """
        if self.memory is None:
            raise RuntimeError(
                "Session not started. Call start_session() first.")

        topic_name = topic_config['name']
        logger.info("="*60)
        logger.info(f"Assessing topic: {topic_name}")
        logger.info("="*60)

        # update memory
        self.memory.add_topic_node(topic_name)

        logger.info("Generating initial question...")
        question = self.question_agent.generate_initial_question(
            topic_config=topic_config,
            memory=self.memory
        )
        logger.info(f"Initial question: {question[:100]}...")

        follow_up_count = 0

        # topic wise multi-turn conversation loop
        while follow_up_count < self.max_follow_ups:
            
            logger.info(
                f"Waiting for user response (turn {follow_up_count + 1})...")
            answer = user_response_fn(question)
            logger.info(f"User response received: {answer[:100]}...")

            
            self.memory.add_qa_pair(topic_name, question, answer)

            
            statement = self.memory.extract_information(answer)
            self.memory.add_statement(topic_name, statement)
            logger.info(f"Extracted statement: {statement.to_dict()}")

            
            logger.info("Evaluating response adequacy...")
            necessity_score = self.evaluation_agent.evaluate_adequacy(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=self.memory
            )
            logger.info(f"Necessity score: {necessity_score}")

            if necessity_score < self.necessity_threshold:
                logger.info("✓ Adequate information collected")
                break

            follow_up_count += 1

            if follow_up_count >= self.max_follow_ups:
                logger.info(
                    f"⚠ Max follow-ups ({self.max_follow_ups}) reached")
                break

            logger.info(f"Generating follow-up question {follow_up_count}...")
            question = self.question_agent.generate_followup_question(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=self.memory
            )
            logger.info(f"Follow-up question: {question[:100]}...")

        logger.info("Scoring topic...")
        score, summary, basis = self.scoring_agent.score_topic(
            topic_config=topic_config,
            topic_name=topic_name,
            memory=self.memory
        )

        self.memory.update_topic_score(topic_name, score, summary, basis)

        logger.info(f"✓ Topic '{topic_name}' scored: {score}")
        logger.info(f"  Summary: {summary[:100]}...")
        logger.info("="*60)

        return score, summary, basis

    def run_full_assessment(
        self,
        user_id: str,
        user_info: Optional[Dict] = None,
        user_response_fn: callable = None
    ) -> Dict:
        """
        Run complete PHQ-8 assessment across all topics

        Args:
            user_id: Unique user identifier
            user_info: Optional user information
            user_response_fn: Function to get user responses
                            If None, uses simulated responses

        Returns:
            Complete assessment results dictionary
        """
        logger.info("="*70)
        logger.info("STARTING FULL ASSESSMENT")
        logger.info("="*70)


        self.start_session(user_id, user_info)

        if user_response_fn is None:
            logger.info("No user_response_fn provided, using simulator")
            user_response_fn = self._create_simulator()

        total_score = 0
        topic_results = []

        for i, topic_config in enumerate(self.topics, 1):
            logger.info(
                f"\n--- Topic {i}/{len(self.topics)}: {topic_config['name']} ---")

            score, summary, basis = self.assess_topic(
                topic_config, user_response_fn)

            total_score += score
            topic_results.append({
                'topic': topic_config['name'],
                'score': score,
                'summary': summary,
                'basis': basis
            })

        logger.info("="*70)
        logger.info("ASSESSMENT COMPLETE")
        logger.info(f"Total Score: {total_score}")
        logger.info("="*70)

        classification = self._classify_risk(total_score)

        crisis_indicators = self._detect_crisis_indicators()

        assessment_results = {
            'user_id': user_id,
            'user_info': user_info or {},
            'scale_name': self.scale_name,
            'total_score': total_score,
            'classification': classification,
            'crisis_indicators': crisis_indicators,
            'topics': topic_results,
            'topic_details': {
                name: node.to_dict()
                for name, node in self.memory.topics.items()
            },
            'timestamp': datetime.now().isoformat(),
            'memory_stats': self.memory.get_memory_stats()
        }

        logger.info("✓ Assessment results compiled")

        return assessment_results

    def generate_assessment_report(self, assessment_results: Dict) -> str:
        """
        Generate final assessment report

        Args:
            assessment_results: Results from run_full_assessment()

        Returns:
            Formatted assessment report string
        """
        logger.info("Generating assessment report...")

        if self.memory is None:
            raise RuntimeError("No active session. Run assessment first.")

        report = self.updating_agent.generate_report(
            memory=self.memory,
            total_score=assessment_results['total_score'],
            scale_name=self.scale_name
        )

        logger.info("✓ Assessment report generated")

        return report

    def _classify_risk(self, total_score: int) -> str:
        """
        Classify risk level based on PHQ-8 score

        PHQ-8 Score Ranges:
        - 0-4: None or minimal depression
        - 5-9: Mild depression
        - 10-14: Moderate depression
        - 15-19: Moderately severe depression
        - 20-24: Severe depression

        Args:
            total_score: Total PHQ-8 score

        Returns:
            Risk classification string
        """
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

        logger.info(
            f"Risk classification: {classification} (score: {total_score})")

        return classification

    def _detect_crisis_indicators(self) -> List[str]:
        """
        Detect crisis keywords in conversation history

        Returns:
            List of detected crisis keywords
        """
        crisis_keywords = [
            'want to die', 'kill myself', 'end it all', 'no point living',
            'suicide', 'suicidal', 'self harm', 'hurt myself',
            'better off dead', 'end my life', 'not worth living'
        ]

        crisis_indicators = []

        if self.memory is None:
            return crisis_indicators

        for topic_name, topic_node in self.memory.topics.items():
            for question, answer in topic_node.qa_pairs:
                answer_lower = answer.lower()
                for keyword in crisis_keywords:
                    if keyword in answer_lower:
                        if keyword not in crisis_indicators:
                            crisis_indicators.append(keyword)
                            logger.warning(
                                f"⚠ Crisis keyword detected: '{keyword}' in topic '{topic_name}'")

        if crisis_indicators:
            logger.critical(
                f"⚠⚠⚠ {len(crisis_indicators)} crisis indicators detected!")

        return crisis_indicators

    def _create_simulator(self):
        """
        Create a simple response simulator for testing

        Returns:
            Simulator function
        """
        import random

        turn_counter = {'count': 0}

        def simulator(question: str) -> str:
            """Simulate user responses"""
            turn = turn_counter['count'] % 3
            turn_counter['count'] += 1

            responses = {
                0: [
                    "Yes, I've been experiencing that.",
                    "I have noticed some issues with this.",
                    "It's been affecting me recently.",
                    "Yes, quite a bit actually."
                ],
                1: [
                    "It happens quite often, maybe several times a week.",
                    "Fairly frequently, it's hard to manage.",
                    "More than I'd like, it's becoming a concern.",
                    "Almost every day for the past two weeks."
                ],
                2: [
                    "It really impacts my daily life and work performance.",
                    "Yes, it makes things difficult and exhausting.",
                    "It's been making everything harder to handle.",
                    "It affects my relationships and productivity."
                ]
            }

            return random.choice(responses.get(turn, ["I'm not sure how to describe it further."]))

        return simulator

    def get_memory(self) -> Optional[TreeMemory]:
        """
        Get current memory instance

        Returns:
            Current TreeMemory instance or None
        """
        return self.memory

    def save_results(self, assessment_results: Dict, output_dir: str = './assessments'):
        """
        Save assessment results to JSON file

        Args:
            assessment_results: Results dictionary
            output_dir: Directory to save results
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/assessment_{assessment_results['user_id']}_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(assessment_results, f, indent=2, default=str)
            logger.info(f"✓ Results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"✗ Failed to save results: {e}")
            raise


def main():
    """Example usage of AssessmentPipeline"""
    import yaml

    print("\n" + "="*70)
    print("Assessment Pipeline - Standalone Demo")
    print("="*70 + "\n")

    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    from utils.llm import LLMOrchestrator
    llm = LLMOrchestrator(config)

    pipeline = AssessmentPipeline(llm, config)

    user_info = {
        'age': 28,
        'occupation': 'Teacher',
        'gender': 'Female'
    }

    results = pipeline.run_full_assessment(
        user_id="demo_user_001",
        user_info=user_info,
        user_response_fn=None  # Uses simulator
    )

    report = pipeline.generate_assessment_report(results)

    print("\n" + "="*70)
    print("ASSESSMENT RESULTS")
    print("="*70)
    print(f"User ID: {results['user_id']}")
    print(f"Total Score: {results['total_score']}")
    print(f"Classification: {results['classification']}")
    print(f"Crisis Indicators: {len(results['crisis_indicators'])}")
    print("\n" + "-"*70)
    print("ASSESSMENT REPORT")
    print("-"*70)
    print(report)
    print("="*70 + "\n")

    pipeline.save_results(results)


if __name__ == "__main__":
    main()
