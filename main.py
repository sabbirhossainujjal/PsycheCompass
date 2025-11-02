import json
from utils.agents import AgentMental
from utils.logger import get_logger

# ============================================================================ 
# Main Execution
# ============================================================================ 

if __name__ == "__main__":
    main_logger = get_logger("main", "logs/main.log")
    main_logger.info("Starting new assessment.")

    # Initialize the framework
    agent_mental = AgentMental("confiig.yml")
    
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
    
    main_logger.info(f"Assessment complete. Total score: {results['total_score']}, Classification: {results['classification']}")
    print("\n" + "="*60)
    print("Assessment Results Summary:")
    print(json.dumps(results, indent=2, default=str))
    