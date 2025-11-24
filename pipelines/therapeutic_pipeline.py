"""
Therapeutic Pipeline

Orchestrates the therapeutic subsystem agents:
- Therapeutic Router
- Emotional Support Agent
- Therapeutic Agent
- Crisis Support Agent
- Knowledge Retrieval Agent
- Clinical Validator

This module provides a clean interface for generating therapeutic responses
based on assessment results.
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from utils.logger import setup_logger
from utils.llm import LLMOrchestrator

# Import therapeutic agents
from therapeutic_agents.router import TherapeuticRouter
from therapeutic_agents.emotional_support_agent import EmotionalSupportAgent
from therapeutic_agents.therapeutic_agent import TherapeuticAgent
from therapeutic_agents.crisis_support_agent import CrisisSupportAgent
from therapeutic_agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from therapeutic_agents.clinical_validator import ClinicalValidator

logger = setup_logger('therapeutic_pipeline', 'logs/therapeutic_pipeline.log')


class TherapeuticPipeline:
    """
    Therapeutic Pipeline - Orchestrates therapeutic support

    This pipeline:
    1. Receives assessment results
    2. Routes to appropriate therapeutic agent (based on risk level)
    3. Generates therapeutic response
    4. Validates response for safety and clinical appropriateness
    5. Returns validated therapeutic support
    """

    def __init__(self, llm: LLMOrchestrator, config: Dict):
        """
        Initialize therapeutic pipeline

        Args:
            llm: LLM orchestrator instance
            config: Configuration dictionary
        """
        logger.info("="*70)
        logger.info("Initializing Therapeutic Pipeline")
        logger.info("="*70)

        self.llm = llm
        self.config = config

        # Check if therapeutic support is enabled
        therapeutic_config = config.get('therapeutic_support', {})
        self.enabled = therapeutic_config.get('enabled', True)

        if not self.enabled:
            logger.warning("⚠ Therapeutic support is disabled in config")
            return

        # Initialize agents
        logger.info("Initializing therapeutic agents...")
        self._initialize_agents()

        # Session tracking
        self.session_history: List[Dict] = []

        logger.info("✓ Therapeutic Pipeline initialized successfully")
        logger.info("="*70)

    def _initialize_agents(self):
        """Initialize all therapeutic agents"""

        # Router
        self.router = TherapeuticRouter(self.config)
        logger.info("✓ Therapeutic Router initialized")

        # Knowledge Retrieval Agent (shared by all agents)
        self.knowledge_agent = KnowledgeRetrievalAgent(
            llm=self.llm,
            config=self.config
        )
        logger.info("✓ Knowledge Retrieval Agent initialized")

        # Emotional Support Agent (low risk)
        self.emotional_support_agent = EmotionalSupportAgent(
            llm=self.llm,
            config=self.config,
            knowledge_agent=self.knowledge_agent
        )
        logger.info("✓ Emotional Support Agent initialized")

        # Therapeutic Agent (moderate risk)
        self.therapeutic_agent = TherapeuticAgent(
            llm=self.llm,
            config=self.config,
            knowledge_agent=self.knowledge_agent
        )
        logger.info("✓ Therapeutic Agent initialized")

        # Crisis Support Agent (high risk)
        self.crisis_support_agent = CrisisSupportAgent(
            llm=self.llm,
            config=self.config,
            knowledge_agent=self.knowledge_agent
        )
        logger.info("✓ Crisis Support Agent initialized")

        # Clinical Validator
        self.validator = ClinicalValidator(
            llm=self.llm,
            config=self.config
        )
        logger.info("✓ Clinical Validator initialized")

    def prepare_therapeutic_input(self, assessment_results: Dict) -> Dict:
        """
        Prepare therapeutic input from assessment results

        Args:
            assessment_results: Results from assessment pipeline

        Returns:
            Prepared therapeutic input dictionary
        """
        logger.info("Preparing therapeutic input from assessment results...")

        # Extract key information
        total_score = assessment_results.get('total_score', 0)
        classification = assessment_results.get('classification', 'Unknown')
        crisis_indicators = assessment_results.get('crisis_indicators', [])
        topics = assessment_results.get('topics', [])
        user_info = assessment_results.get('user_info', {})

        # Identify key symptoms (highest scoring topics)
        key_symptoms = []
        for topic in topics:
            if topic['score'] >= 2:  # Moderate or higher
                key_symptoms.append({
                    'symptom': topic['topic'],
                    'score': topic['score'],
                    'severity': self._score_to_severity(topic['score']),
                    'summary': topic['summary']
                })

        # Sort by score (descending)
        key_symptoms.sort(key=lambda x: x['score'], reverse=True)

        # Create assessment summary
        assessment_summary = self._create_assessment_summary(
            assessment_results,
            key_symptoms
        )

        # Prepare therapeutic input
        therapeutic_input = {
            'phq8_score': total_score,
            'classification': classification,
            'risk_level': self._determine_risk_level(total_score, crisis_indicators),
            'crisis_indicators': crisis_indicators,
            'key_symptoms': key_symptoms,
            'assessment_summary': assessment_summary,
            'user_info': user_info,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Therapeutic input prepared")
        logger.info(f"  PHQ-8 Score: {total_score}")
        logger.info(f"  Risk Level: {therapeutic_input['risk_level']}")
        logger.info(f"  Key Symptoms: {len(key_symptoms)}")
        logger.info(f"  Crisis Indicators: {len(crisis_indicators)}")

        return therapeutic_input

    def generate_response(
        self,
        assessment_results: Dict,
        user_message: Optional[str] = None
    ) -> Dict:
        """
        Generate therapeutic response based on assessment

        Args:
            assessment_results: Results from assessment pipeline
            user_message: Optional user message for follow-up

        Returns:
            Therapeutic response dictionary
        """
        if not self.enabled:
            logger.error("Therapeutic support is disabled")
            return {
                'error': 'Therapeutic support is disabled in configuration',
                'response': None
            }

        logger.info("="*70)
        logger.info("GENERATING THERAPEUTIC RESPONSE")
        logger.info("="*70)

        # Prepare input
        therapeutic_input = self.prepare_therapeutic_input(assessment_results)

        # Add user message if provided
        if user_message:
            therapeutic_input['user_message'] = user_message

        # Route to appropriate agent
        logger.info("Routing to appropriate therapeutic agent...")
        agent_type = self.router.route(therapeutic_input)
        logger.info(f"→ Routed to: {agent_type.upper()} AGENT")

        # Generate response based on routing
        logger.info("Generating therapeutic response...")

        if agent_type == 'crisis':
            response = self.crisis_support_agent.generate_response(
                therapeutic_input)

        elif agent_type == 'therapeutic':
            response = self.therapeutic_agent.generate_response(
                therapeutic_input)

        elif agent_type == 'emotional_support':
            response = self.emotional_support_agent.generate_response(
                therapeutic_input)

        else:
            logger.error(f"Unknown agent type: {agent_type}")
            response = "I apologize, but I'm unable to provide a response at this time."

        logger.info("✓ Therapeutic response generated")

        # Validate response
        logger.info("Validating therapeutic response...")
        validated_response = self.validator.validate(
            therapeutic_response=response,
            risk_level=therapeutic_input['risk_level'],
            assessment_results=therapeutic_input
        )
        logger.info("✓ Response validated")

        # Create result
        result = {
            'agent_type': agent_type,
            'risk_level': therapeutic_input['risk_level'],
            'phq8_score': therapeutic_input['phq8_score'],
            'crisis_indicators': therapeutic_input['crisis_indicators'],
            'response': validated_response,
            'routing_explanation': self.router.get_routing_explanation(therapeutic_input),
            'timestamp': datetime.now().isoformat()
        }

        # Add to session history
        self.session_history.append({
            'type': 'therapeutic_response',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })

        logger.info("="*70)
        logger.info("THERAPEUTIC RESPONSE COMPLETE")
        logger.info("="*70)

        return result

    def continue_conversation(
        self,
        user_message: str,
        assessment_results: Dict,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Continue therapeutic conversation (multi-turn)

        Args:
            user_message: User's latest message
            assessment_results: Original assessment results
            conversation_history: Previous conversation turns

        Returns:
            Therapeutic response dictionary
        """
        logger.info("Continuing therapeutic conversation...")

        if not self.enabled:
            return {
                'error': 'Therapeutic support is disabled',
                'response': None
            }

        # Prepare input
        therapeutic_input = self.prepare_therapeutic_input(assessment_results)
        therapeutic_input['user_message'] = user_message

        # Route (may change if crisis detected)
        agent_type = self.router.route(therapeutic_input)

        # Use conversation history from session or provided
        if conversation_history is None:
            conversation_history = [
                item for item in self.session_history
                if item['type'] in ['therapeutic_response', 'user_message']
            ]

        logger.info(f"Routing to {agent_type} agent for follow-up...")

        # Generate follow-up response
        if agent_type == 'crisis':
            response = self.crisis_support_agent.continue_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                assessment_results=therapeutic_input
            )

        elif agent_type == 'therapeutic':
            response = self.therapeutic_agent.continue_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                assessment_results=therapeutic_input
            )

        elif agent_type == 'emotional_support':
            response = self.emotional_support_agent.continue_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                assessment_results=therapeutic_input
            )

        # Validate
        validated_response = self.validator.validate(
            therapeutic_response=response,
            risk_level=therapeutic_input['risk_level'],
            assessment_results=therapeutic_input
        )

        result = {
            'agent_type': agent_type,
            'response': validated_response,
            'timestamp': datetime.now().isoformat()
        }

        # Update history
        self.session_history.append({
            'type': 'user_message',
            'message': user_message,
            'timestamp': datetime.now().isoformat()
        })

        self.session_history.append({
            'type': 'therapeutic_response',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })

        return result

    def _score_to_severity(self, score: int) -> str:
        """Convert score to severity label"""
        severity_map = {
            0: "Not at all",
            1: "Several days",
            2: "More than half the days",
            3: "Nearly every day"
        }
        return severity_map.get(score, "Unknown")

    def _determine_risk_level(self, score: int, crisis_indicators: List[str]) -> str:
        """
        Determine risk level for routing

        Args:
            score: PHQ-8 total score
            crisis_indicators: List of crisis keywords detected

        Returns:
            Risk level: 'low', 'moderate', 'high', or 'crisis'
        """
        # Crisis indicators override everything
        if crisis_indicators:
            return 'crisis'

        # Score-based classification
        if score >= 15:
            return 'high'
        elif score >= 5:
            return 'moderate'
        else:
            return 'low'

    def _create_assessment_summary(
        self,
        assessment_results: Dict,
        key_symptoms: List[Dict]
    ) -> str:
        """
        Create human-readable assessment summary

        Args:
            assessment_results: Assessment results dictionary
            key_symptoms: List of key symptoms

        Returns:
            Formatted assessment summary
        """
        summary = f"""PHQ-8 Depression Assessment Results:

Total Score: {assessment_results['total_score']}/24
Classification: {assessment_results['classification']}
Assessment Scale: {assessment_results['scale_name']}

"""

        if key_symptoms:
            summary += "Primary Symptoms Identified:\n"
            for i, symptom in enumerate(key_symptoms[:5], 1):  # Top 5
                summary += f"{i}. {symptom['symptom']}: {symptom['severity']} (Score: {symptom['score']})\n"
                summary += f"   {symptom['summary']}\n"

        if assessment_results.get('crisis_indicators'):
            summary += f"\n⚠ CRISIS INDICATORS DETECTED: {len(assessment_results['crisis_indicators'])} keywords\n"

        return summary

    def save_session(self, output_dir: str = './therapy_sessions'):
        """
        Save therapeutic session to file

        Args:
            output_dir: Directory to save session
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/session_{timestamp}.json"

        session_data = {
            'session_history': self.session_history,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            logger.info(f"✓ Session saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"✗ Failed to save session: {e}")
            raise

    def get_session_history(self) -> List[Dict]:
        """Get current session history"""
        return self.session_history

    def clear_session(self):
        """Clear current session history"""
        logger.info("Clearing session history")
        self.session_history = []


def main():
    """Example usage of TherapeuticPipeline"""
    import yaml

    print("\n" + "="*70)
    print("Therapeutic Pipeline - Standalone Demo")
    print("="*70 + "\n")

    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize LLM
    from utils.llm import LLMOrchestrator
    llm = LLMOrchestrator(config)

    # Create pipeline
    pipeline = TherapeuticPipeline(llm, config)

    # Simulate assessment results (moderate depression)
    mock_assessment = {
        'user_id': 'demo_user_001',
        'user_info': {'age': 28, 'occupation': 'Teacher', 'gender': 'Female'},
        'scale_name': 'PHQ-8',
        'total_score': 12,  # Moderate depression
        'classification': 'Moderate',
        'crisis_indicators': [],
        'topics': [
            {'topic': 'Depressed Mood', 'score': 2,
                'summary': 'Feeling down more than half the days'},
            {'topic': 'Loss of Interest', 'score': 2,
                'summary': 'Reduced interest in activities'},
            {'topic': 'Sleep Problems', 'score': 2,
                'summary': 'Difficulty sleeping most nights'},
            {'topic': 'Fatigue or Low Energy', 'score': 2,
                'summary': 'Low energy frequently'},
            {'topic': 'Concentration Difficulties', 'score': 2,
                'summary': 'Hard to focus on tasks'},
            {'topic': 'Appetite or Weight Changes', 'score': 1,
                'summary': 'Some changes in appetite'},
            {'topic': 'Low Self-Worth', 'score': 1,
                'summary': 'Occasional negative thoughts'},
            {'topic': 'Psychomotor Changes', 'score': 0,
                'summary': 'No significant changes'}
        ]
    }

    # Generate therapeutic response
    result = pipeline.generate_response(mock_assessment)

    # Display results
    print("\n" + "="*70)
    print("THERAPEUTIC RESPONSE")
    print("="*70)
    print(f"Agent Type: {result['agent_type'].upper()}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"PHQ-8 Score: {result['phq8_score']}")
    print("\n" + "-"*70)
    print("RESPONSE:")
    print("-"*70)
    print(result['response'])
    print("\n" + "-"*70)
    print("ROUTING EXPLANATION:")
    print("-"*70)
    print(result['routing_explanation'])
    print("="*70 + "\n")

    # Save session
    pipeline.save_session()


if __name__ == "__main__":
    main()
