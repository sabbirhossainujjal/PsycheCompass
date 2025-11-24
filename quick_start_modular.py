"""
PsycheCompass Modular Architecture - Quick Start Guide
======================================================

This guide helps you understand and use the new modular architecture.
"""

# ARCHITECTURE OVERVIEW
# ====================

"""
Old Architecture (main.py):
---------------------------
main.py (monolithic)
  └── Everything in one file
      - Assessment logic
      - No therapeutic support
      - Hard to extend
      - Hard to test parts independently


New Architecture (Modular):
---------------------------
psychecompass_main.py (thin orchestrator)
  ├── assessment_pipeline.py (Assessment Subsystem)
  │   ├── Question Generator Agent
  │   ├── Evaluation Agent
  │   ├── Scoring Agent
  │   └── Memory Updater Agent
  │
  └── therapeutic_pipeline.py (Therapeutic Subsystem)
      ├── Therapeutic Router
      ├── Emotional Support Agent (low risk)
      ├── Therapeutic Agent (moderate risk)
      ├── Crisis Support Agent (high risk)
      ├── Knowledge Retrieval Agent
      └── Clinical Validator


BENEFITS:
---------
✓ Modular - Each subsystem independent
✓ Testable - Test each part separately
✓ Extensible - Add features without breaking code
✓ Research-friendly - Evaluate subsystems independently
✓ Team-friendly - No merge conflicts
"""


# USAGE EXAMPLES
# ==============

def example_1_full_session():
    """
    Example 1: Run complete end-to-end session
    
    This runs both assessment and therapeutic support
    """
    from psychecompass_main import PsycheCompass
    
    # Initialize system
    psyche = PsycheCompass("config.yml")
    
    # Run full session
    results = psyche.run_full_session(
        user_id="user_001",
        user_info={'age': 28, 'occupation': 'Teacher'},
        interactive=False  # Set True for multi-turn conversation
    )
    
    # Display
    psyche.display_session_summary(results)
    
    return results


def example_2_assessment_only():
    """
    Example 2: Run only assessment (research evaluation)
    
    Useful for:
    - Testing assessment agents on DAIC-WOZ
    - Batch processing assessments
    - Collecting PHQ-8 scores
    """
    from psychecompass_main import PsycheCompass
    
    psyche = PsycheCompass("config.yml")
    
    # Assessment only
    assessment = psyche.run_assessment_only(
        user_id="user_002",
        user_info={'age': 35}
    )
    
    print(f"Score: {assessment['total_score']}")
    print(f"Report: {assessment['report']}")
    
    return assessment


def example_3_therapeutic_only():
    """
    Example 3: Run only therapeutic (given assessment)
    
    Useful for:
    - Testing different therapeutic approaches
    - Comparing agent responses
    - Trying different routing thresholds
    """
    from psychecompass_main import PsycheCompass
    
    psyche = PsycheCompass("config.yml")
    
    # First get assessment (or load from file)
    assessment = psyche.run_assessment_only(
        user_id="user_003",
        user_info={'age': 42}
    )
    
    # Then run therapeutic only
    therapeutic = psyche.run_therapeutic_only(assessment)
    
    print(f"Agent: {therapeutic['agent_type']}")
    print(f"Response: {therapeutic['response']}")
    
    return therapeutic


def example_4_standalone_assessment_pipeline():
    """
    Example 4: Use assessment pipeline directly
    
    Maximum control over assessment process
    """
    from assessment_pipeline import AssessmentPipeline
    from utils.llm import LLMOrchestrator
    import yaml
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize
    llm = LLMOrchestrator(config)
    pipeline = AssessmentPipeline(llm, config)
    
    # Start session
    memory = pipeline.start_session(
        user_id="user_004",
        user_info={'age': 30}
    )
    
    # Assess individual topics with custom response function
    def my_response_function(question):
        # Your custom logic here
        return "User's answer to: " + question
    
    score, summary, basis = pipeline.assess_topic(
        topic_config=config['assessment_scale']['topics'][0],
        user_response_fn=my_response_function
    )
    
    return score, summary


def example_5_standalone_therapeutic_pipeline():
    """
    Example 5: Use therapeutic pipeline directly
    
    Maximum control over therapeutic responses
    """
    from therapeutic_pipeline import TherapeuticPipeline
    from utils.llm import LLMOrchestrator
    import yaml
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize
    llm = LLMOrchestrator(config)
    pipeline = TherapeuticPipeline(llm, config)
    
    # Create mock assessment
    assessment = {
        'user_id': 'user_005',
        'total_score': 15,  # High risk
        'classification': 'Moderately Severe',
        'crisis_indicators': [],
        'topics': [
            {'topic': 'Depressed Mood', 'score': 2, 'summary': '...'},
            # ... more topics
        ],
        'user_info': {}
    }
    
    # Generate response
    result = pipeline.generate_response(assessment)
    
    print(f"Routed to: {result['agent_type']}")
    print(f"Response: {result['response']}")
    
    return result


def example_6_multi_turn_conversation():
    """
    Example 6: Multi-turn conversation
    
    Interactive therapeutic conversation
    """
    from psychecompass_main import PsycheCompass
    
    psyche = PsycheCompass("config.yml")
    
    # Run full session
    session = psyche.run_full_session(
        user_id="user_006",
        user_info={'age': 26},
        interactive=False  # Get session first
    )
    
    # Continue conversation
    response1 = psyche.continue_conversation(
        session_id=session['session_id'],
        user_message="Can you explain more about CBT?"
    )
    
    response2 = psyche.continue_conversation(
        session_id=session['session_id'],
        user_message="How do I get started with that?"
    )
    
    return response1, response2


def example_7_batch_processing():
    """
    Example 7: Batch process multiple users
    
    Process dataset of assessments
    """
    from assessment_pipeline import AssessmentPipeline
    from utils.llm import LLMOrchestrator
    import yaml
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    llm = LLMOrchestrator(config)
    pipeline = AssessmentPipeline(llm, config)
    
    # Batch process
    users = [
        {'id': 'user_007', 'age': 25},
        {'id': 'user_008', 'age': 30},
        {'id': 'user_009', 'age': 35},
    ]
    
    results = []
    for user in users:
        result = pipeline.run_full_assessment(
            user_id=user['id'],
            user_info={'age': user['age']}
        )
        results.append(result)
        
        # Save
        pipeline.save_results(result)
    
    return results


def example_8_custom_agent():
    """
    Example 8: Add custom therapeutic agent
    
    Extend the system with new agents
    """
    from therapeutic_pipeline import TherapeuticPipeline
    from utils.llm import LLMOrchestrator
    import yaml
    
    class MyCustomAgent:
        """Custom therapeutic agent"""
        
        def __init__(self, llm, config):
            self.llm = llm
            self.config = config
        
        def generate_response(self, assessment_results):
            # Your custom logic
            response = "Custom therapeutic response based on..."
            return response
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    llm = LLMOrchestrator(config)
    pipeline = TherapeuticPipeline(llm, config)
    
    # Add custom agent
    pipeline.my_custom_agent = MyCustomAgent(llm, config)
    
    # Use it
    # ...


# TESTING
# =======

def run_tests():
    """Run all tests"""
    import subprocess
    subprocess.run(['python', 'test_modular_architecture.py'])


# FILE MAPPING
# ============

"""
What each file does:

psychecompass_main.py
  → Main orchestrator
  → Connects assessment + therapeutic
  → Manages sessions
  → USE: Full end-to-end workflows

assessment_pipeline.py
  → PHQ-8 assessment subsystem
  → Multi-turn questioning
  → Scoring and reporting
  → USE: Research on assessment, batch processing

therapeutic_pipeline.py
  → Therapeutic support subsystem
  → Risk-based routing
  → Multi-agent responses
  → Clinical validation
  → USE: Testing therapeutic approaches

config.yml
  → All configuration
  → Agent prompts
  → Routing thresholds
  → LLM settings
  → USE: Tune system behavior

utils/agents.py
  → Assessment agents implementation
  → QuestionGenerator, Evaluator, Scorer, Updater

therapeutic_agents/*.py
  → Therapeutic agents implementation
  → Router, EmotionalSupport, Therapeutic, Crisis, etc.

test_modular_architecture.py
  → Comprehensive test suite
  → Tests each component
  → USE: Verify system works
"""


# COMMON WORKFLOWS
# ================

"""
Research Evaluation:
-------------------
1. Evaluate assessment on DAIC-WOZ:
   - Use assessment_pipeline.py standalone
   - Calculate MAE, F1 score
   
2. Evaluate therapeutic on Psych8k:
   - Use therapeutic_pipeline.py standalone
   - Calculate BLEU, empathy scores

3. End-to-end evaluation:
   - Use psychecompass_main.py
   - Test full pipeline


Development:
-----------
1. Testing new assessment prompts:
   - Edit config.yml agent_prompts section
   - Run: python assessment_pipeline.py
   
2. Testing new therapeutic agents:
   - Modify therapeutic_agents/*.py
   - Run: python therapeutic_pipeline.py
   
3. Changing routing thresholds:
   - Edit config.yml therapeutic_support.routing
   - Run: python psychecompass_main.py


Production:
----------
1. Deploy full system:
   - Import PsycheCompass
   - Run full sessions
   
2. Custom user interface:
   - Call assessment_pipeline for assessment
   - Display questions to user
   - Call therapeutic_pipeline for support
   
3. Batch processing:
   - Use assessment_pipeline for many users
   - Save results to database
"""


# QUICK COMMANDS
# ==============

"""
# Test entire system
python test_modular_architecture.py

# Test assessment only
python assessment_pipeline.py

# Test therapeutic only
python therapeutic_pipeline.py

# Run full system demo
python psychecompass_main.py

# Check configuration
cat config.yml

# View logs
tail -f logs/psychecompass.log
tail -f logs/assessment_pipeline.log
tail -f logs/therapeutic_pipeline.log
"""


if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("Available Examples:")
    print("="*70)
    print("1. example_1_full_session() - Complete end-to-end")
    print("2. example_2_assessment_only() - Assessment subsystem")
    print("3. example_3_therapeutic_only() - Therapeutic subsystem")
    print("4. example_4_standalone_assessment_pipeline() - Direct pipeline use")
    print("5. example_5_standalone_therapeutic_pipeline() - Direct pipeline use")
    print("6. example_6_multi_turn_conversation() - Interactive mode")
    print("7. example_7_batch_processing() - Batch processing")
    print("8. example_8_custom_agent() - Extending the system")
    print("="*70)
    
    print("\nRun any example:")
    print("  from quick_start_modular import example_1_full_session")
    print("  result = example_1_full_session()")
    
    print("\nOr run tests:")
    print("  python test_modular_architecture.py")
