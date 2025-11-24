"""
PsycheCompass - Main Orchestrator

This is the main entry point that connects:
1. Assessment Pipeline (PHQ-8 mental health assessment)
2. Therapeutic Pipeline (Risk-based therapeutic support)

Clean modular architecture for research and production use.
"""

import yaml
import json
import os
from datetime import datetime
from typing import Dict, Optional, List

from utils.logger import setup_logger, log_session_start, log_session_end
from utils.llm import LLMOrchestrator

# Import pipelines
from pipelines.assessment_pipeline import AssessmentPipeline
from pipelines.therapeutic_pipeline import TherapeuticPipeline

# Setup logger
logger = setup_logger('psychecompass', 'logs/psychecompass.log')


class PsycheCompass:
    """
    PsycheCompass - Adaptive Multi-Agent Mental Health System
    
    Main orchestrator that coordinates:
    - Assessment subsystem (PHQ-8 screening)
    - Therapeutic subsystem (Risk-based support)
    - Clinical validation (Safety checks)
    
    This implements the architecture from the research paper:
    "PsycheCompass: An Adaptive Multi-Agent system for Mental Health 
     Assessment & Therapeutic Support"
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize PsycheCompass system
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("="*70)
        logger.info("INITIALIZING PSYCHECOMPASS")
        logger.info("Adaptive Multi-Agent Mental Health System")
        logger.info("="*70)
        
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM Orchestrator
        logger.info("Initializing LLM Orchestrator...")
        self.llm = LLMOrchestrator(self.config)
        
        # Test LLM connection
        if self.llm.test_connection():
            logger.info("✓ LLM connection successful")
        else:
            logger.warning("⚠ LLM connection test failed")
        
        # Initialize pipelines
        logger.info("\nInitializing subsystems...")
        self._initialize_pipelines()
        
        # Session tracking
        self.current_session_id: Optional[str] = None
        self.sessions: Dict[str, Dict] = {}
        
        logger.info("="*70)
        logger.info("✓ PSYCHECOMPASS INITIALIZED SUCCESSFULLY")
        logger.info("="*70)
    
    def _initialize_pipelines(self):
        """Initialize both assessment and therapeutic pipelines"""
        
        # Assessment Pipeline
        logger.info("→ Initializing Assessment Pipeline...")
        self.assessment_pipeline = AssessmentPipeline(
            llm=self.llm,
            config=self.config
        )
        logger.info("  ✓ Assessment Pipeline ready")
        
        # Therapeutic Pipeline
        logger.info("→ Initializing Therapeutic Pipeline...")
        self.therapeutic_pipeline = TherapeuticPipeline(
            llm=self.llm,
            config=self.config
        )
        logger.info("  ✓ Therapeutic Pipeline ready")
    
    def run_full_session(
        self,
        user_id: str,
        user_info: Optional[Dict] = None,
        user_response_fn: Optional[callable] = None,
        interactive: bool = False
    ) -> Dict:
        """
        Run complete end-to-end session:
        1. PHQ-8 Assessment
        2. Therapeutic Response
        3. (Optional) Multi-turn conversation
        
        Args:
            user_id: Unique user identifier
            user_info: Optional user information
            user_response_fn: Function to get user responses for assessment
            interactive: If True, allows multi-turn conversation
            
        Returns:
            Complete session results
        """
        # Generate session ID
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session_id = session_id
        
        log_session_start(logger, session_id)
        
        logger.info("="*70)
        logger.info("STARTING FULL SESSION")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"User ID: {user_id}")
        logger.info("="*70)
        
        # =====================================================================
        # PHASE 1: ASSESSMENT
        # =====================================================================
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: MENTAL HEALTH ASSESSMENT")
        logger.info("="*70)
        
        assessment_results = self.assessment_pipeline.run_full_assessment(
            user_id=user_id,
            user_info=user_info,
            user_response_fn=user_response_fn
        )
        
        logger.info("\n✓ Assessment Phase Complete")
        logger.info(f"  Total Score: {assessment_results['total_score']}")
        logger.info(f"  Classification: {assessment_results['classification']}")
        logger.info(f"  Crisis Indicators: {len(assessment_results['crisis_indicators'])}")
        
        # Generate assessment report
        assessment_report = self.assessment_pipeline.generate_assessment_report(
            assessment_results
        )
        
        # =====================================================================
        # PHASE 2: THERAPEUTIC RESPONSE
        # =====================================================================
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: THERAPEUTIC SUPPORT")
        logger.info("="*70)
        
        therapeutic_result = self.therapeutic_pipeline.generate_response(
            assessment_results=assessment_results
        )
        
        logger.info("\n✓ Therapeutic Phase Complete")
        logger.info(f"  Agent Type: {therapeutic_result['agent_type']}")
        logger.info(f"  Risk Level: {therapeutic_result['risk_level']}")
        
        # =====================================================================
        # PHASE 3: INTERACTIVE CONVERSATION (Optional)
        # =====================================================================
        conversation_history = []
        
        if interactive:
            logger.info("\n" + "="*70)
            logger.info("PHASE 3: INTERACTIVE CONVERSATION")
            logger.info("="*70)
            
            conversation_history = self._interactive_conversation(
                assessment_results,
                therapeutic_result
            )
        
        # =====================================================================
        # COMPILE COMPLETE SESSION RESULTS
        # =====================================================================
        session_results = {
            'session_id': session_id,
            'user_id': user_id,
            'user_info': user_info or {},
            'timestamp': datetime.now().isoformat(),
            
            # Assessment results
            'assessment': {
                'total_score': assessment_results['total_score'],
                'classification': assessment_results['classification'],
                'crisis_indicators': assessment_results['crisis_indicators'],
                'topics': assessment_results['topics'],
                'report': assessment_report
            },
            
            # Therapeutic results
            'therapeutic': {
                'agent_type': therapeutic_result['agent_type'],
                'risk_level': therapeutic_result['risk_level'],
                'response': therapeutic_result['response'],
                'routing_explanation': therapeutic_result['routing_explanation']
            },
            
            # Conversation history (if interactive)
            'conversation_history': conversation_history,
            
            # Full details
            'full_assessment_results': assessment_results,
            'full_therapeutic_result': therapeutic_result
        }
        
        # Store session
        self.sessions[session_id] = session_results
        
        # Save results
        if self.config.get('output', {}).get('save_results', False):
            self._save_session(session_results)
        
        log_session_end(logger, session_id, "completed")
        
        return session_results
    
    def run_assessment_only(
        self,
        user_id: str,
        user_info: Optional[Dict] = None,
        user_response_fn: Optional[callable] = None
    ) -> Dict:
        """
        Run only the assessment phase (no therapeutic response)
        
        Useful for:
        - Research evaluation of assessment subsystem
        - Batch processing
        - Data collection
        
        Args:
            user_id: Unique user identifier
            user_info: Optional user information
            user_response_fn: Function to get user responses
            
        Returns:
            Assessment results
        """
        logger.info("Running assessment-only mode")
        
        assessment_results = self.assessment_pipeline.run_full_assessment(
            user_id=user_id,
            user_info=user_info,
            user_response_fn=user_response_fn
        )
        
        # Generate report
        assessment_report = self.assessment_pipeline.generate_assessment_report(
            assessment_results
        )
        
        assessment_results['report'] = assessment_report
        
        return assessment_results
    
    def run_therapeutic_only(self, assessment_results: Dict) -> Dict:
        """
        Run only therapeutic response (given assessment results)
        
        Useful for:
        - Testing therapeutic agents
        - Trying different therapeutic approaches
        - Research evaluation
        
        Args:
            assessment_results: Pre-existing assessment results
            
        Returns:
            Therapeutic results
        """
        logger.info("Running therapeutic-only mode")
        
        therapeutic_result = self.therapeutic_pipeline.generate_response(
            assessment_results=assessment_results
        )
        
        return therapeutic_result
    
    def continue_conversation(
        self,
        session_id: str,
        user_message: str
    ) -> Dict:
        """
        Continue an existing therapeutic conversation
        
        Args:
            session_id: Session ID to continue
            user_message: User's message
            
        Returns:
            Therapeutic response
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        assessment_results = session['full_assessment_results']
        
        # Get response from therapeutic pipeline
        response = self.therapeutic_pipeline.continue_conversation(
            user_message=user_message,
            assessment_results=assessment_results
        )
        
        # Update session history
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        session['conversation_history'].append({
            'turn': len(session['conversation_history']) + 1,
            'user_message': user_message,
            'agent_response': response['response'],
            'agent_type': response['agent_type'],
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _interactive_conversation(
        self,
        assessment_results: Dict,
        initial_therapeutic_result: Dict
    ) -> List[Dict]:
        """
        Run interactive multi-turn conversation
        
        Args:
            assessment_results: Assessment results
            initial_therapeutic_result: Initial therapeutic response
            
        Returns:
            Conversation history
        """
        logger.info("Starting interactive conversation mode")
        print("\n" + "="*70)
        print("INTERACTIVE CONVERSATION MODE")
        print("Type 'exit' to end conversation")
        print("="*70 + "\n")
        
        conversation_history = []
        
        # Show initial therapeutic response
        print(f"Agent: {initial_therapeutic_result['response']}\n")
        
        turn = 0
        max_turns = 10
        
        while turn < max_turns:
            turn += 1
            
            # Get user input
            user_message = input(f"\nYou (turn {turn}): ").strip()
            
            if user_message.lower() in ['exit', 'quit', 'bye']:
                logger.info("User ended conversation")
                print("\nThank you for the conversation. Take care!")
                break
            
            if not user_message:
                continue
            
            # Get therapeutic response
            response = self.therapeutic_pipeline.continue_conversation(
                user_message=user_message,
                assessment_results=assessment_results
            )
            
            print(f"\nAgent ({response['agent_type']}): {response['response']}\n")
            
            # Store in history
            conversation_history.append({
                'turn': turn,
                'user_message': user_message,
                'agent_response': response['response'],
                'agent_type': response['agent_type'],
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"Interactive conversation ended after {turn} turns")
        
        return conversation_history
    
    def _save_session(self, session_results: Dict):
        """Save complete session results"""
        # Save assessment results
        if self.config.get('output', {}).get('save_results', True):
            output_dir = self.config.get('output', {}).get('output_directory', './assessments')
            self.assessment_pipeline.save_results(
                session_results['full_assessment_results'],
                output_dir
            )
        
        # Save therapeutic session
        if self.config.get('output', {}).get('save_therapeutic_sessions', True):
            self.therapeutic_pipeline.save_session()
        
        # Save complete session
        output_dir = './sessions'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/session_{session_results['session_id']}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(session_results, f, indent=2, default=str)
            logger.info(f"✓ Complete session saved to: {filename}")
        except Exception as e:
            logger.error(f"✗ Failed to save session: {e}")
    
    def display_session_summary(self, session_results: Dict):
        """Display formatted session summary"""
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Session ID: {session_results['session_id']}")
        print(f"User ID: {session_results['user_id']}")
        print(f"Timestamp: {session_results['timestamp']}")
        
        print("\n" + "-"*70)
        print("ASSESSMENT RESULTS")
        print("-"*70)
        print(f"PHQ-8 Score: {session_results['assessment']['total_score']}/24")
        print(f"Classification: {session_results['assessment']['classification']}")
        print(f"Crisis Indicators: {len(session_results['assessment']['crisis_indicators'])}")
        
        print("\n" + "-"*70)
        print("THERAPEUTIC SUPPORT")
        print("-"*70)
        print(f"Agent Type: {session_results['therapeutic']['agent_type'].upper()}")
        print(f"Risk Level: {session_results['therapeutic']['risk_level'].upper()}")
        
        print("\n" + "-"*70)
        print("ASSESSMENT REPORT")
        print("-"*70)
        print(session_results['assessment']['report'])
        
        print("\n" + "-"*70)
        print("THERAPEUTIC RESPONSE")
        print("-"*70)
        print(session_results['therapeutic']['response'])
        
        if session_results['conversation_history']:
            print("\n" + "-"*70)
            print(f"CONVERSATION: {len(session_results['conversation_history'])} turns")
            print("-"*70)
        
        print("="*70 + "\n")
    
    def get_system_info(self) -> Dict:
        """Get system information and configuration"""
        return {
            'system': 'PsycheCompass',
            'version': '1.0.0',
            'llm_provider': self.llm.provider.value,
            'llm_model': self.llm.model_name,
            'assessment_scale': self.config['assessment_scale']['name'],
            'therapeutic_enabled': self.config.get('therapeutic_support', {}).get('enabled', True),
            'active_sessions': len(self.sessions)
        }


def main():
    """Example usage of PsycheCompass"""
    print("\n" + "="*70)
    print("PSYCHECOMPASS - Adaptive Multi-Agent Mental Health System")
    print("="*70 + "\n")
    
    # Initialize system
    psyche = PsycheCompass("config.yml")
    
    # Display system info
    info = psyche.get_system_info()
    print(f"System: {info['system']} v{info['version']}")
    print(f"LLM: {info['llm_model']} ({info['llm_provider']})")
    print(f"Assessment: {info['assessment_scale']}")
    print(f"Therapeutic Support: {'Enabled' if info['therapeutic_enabled'] else 'Disabled'}")
    print()
    
    # User information
    user_info = {
        'age': 28,
        'occupation': 'Software Engineer',
        'gender': 'Male'
    }
    
    # Run full session with simulated user
    print("Running full session with simulated user...\n")
    
    session_results = psyche.run_full_session(
        user_id="demo_user_001",
        user_info=user_info,
        user_response_fn=None,  # Uses simulator
        interactive=False  # Set to True for interactive mode
    )
    
    # Display results
    psyche.display_session_summary(session_results)
    
    print("\n✓ Session complete!")
    print(f"✓ Results saved to ./sessions/ and ./assessments/")


if __name__ == "__main__":
    main()