#!/usr/bin/env python3
"""
AgentMental Framework - Quick Start Script

This script helps you get started with the AgentMental framework quickly.
It will check your environment, help configure the system, and run a demo.
"""

import os
import sys
import yaml


def print_header():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    AgentMental Framework                          ║
    ║                      Quick Start Guide                            ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n[Step 1] Checking dependencies...")
    
    required = ['yaml']
    optional = {
        'google.generativeai': 'Gemini API support',
        'vllm': 'Local LLM support (vLLM)',
        'openai': 'OpenAI-compatible API support'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required
    for module in required:
        try:
            __import__(module)
            print(f"  ✓ {module} is installed")
        except ImportError:
            missing_required.append(module)
            print(f"  ✗ {module} is NOT installed")
    
    # Check optional
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {module} is installed ({description})")
        except ImportError:
            missing_optional.append((module, description))
    
    if missing_required:
        print("\n⚠ Missing required dependencies!")
        print("Install with: pip install pyyaml")
        return False
    
    if missing_optional:
        print("\n📝 Optional dependencies not installed:")
        for module, desc in missing_optional:
            print(f"  - {module}: {desc}")
        print("\nYou can install them based on your needs:")
        print("  For Gemini: pip install google-generativeai")
        print("  For local LLM: pip install vllm")
    
    return True


def setup_config():
    """Guide user through configuration setup"""
    print("\n[Step 2] Configuration Setup...")
    
    if os.path.exists('config.yml'):
        print("  ℹ config.yml already exists")
        choice = input("  Do you want to reconfigure? (y/n): ").lower()
        if choice != 'y':
            return True
    
    print("\n  Which LLM provider do you want to use?")
    print("  1. Gemini API (recommended for getting started)")
    print("  2. Local LLM (requires local inference server)")
    
    choice = input("\n  Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        return setup_gemini_config()
    elif choice == '2':
        return setup_local_config()
    else:
        print("  Invalid choice. Please run again.")
        return False


def setup_gemini_config():
    """Setup Gemini API configuration"""
    print("\n  Configuring Gemini API...")
    
    # Check if API key is in environment
    api_key = os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        print("\n  Gemini API key not found in environment.")
        print("  You can get a free API key at: https://makersuite.google.com/app/apikey")
        
        choice = input("\n  Do you want to enter API key now? (y/n): ").lower()
        if choice == 'y':
            api_key = input("  Enter your Gemini API key: ").strip()
            os.environ['GOOGLE_API_KEY'] = api_key
            print("  ✓ API key set for this session")
            print("  💡 Tip: Add 'export GOOGLE_API_KEY=your_key' to ~/.bashrc for permanent setup")
        else:
            print("\n  ⚠ You'll need to set GOOGLE_API_KEY before running the framework")
            print("  Run: export GOOGLE_API_KEY='your-api-key-here'")
    else:
        print(f"  ✓ Gemini API key found: {api_key[:10]}...")
    
    print("\n  ✓ Gemini configuration ready")
    return True


def setup_local_config():
    """Setup local LLM configuration"""
    print("\n  Configuring Local LLM...")
    print("\n  For local LLM, you need a running inference server.")
    print("  Common options:")
    print("    - vLLM: python -m vllm.entrypoints.openai.api_server --model MODEL_NAME")
    print("    - Ollama: ollama serve")
    print("    - llama.cpp: ./server -m model.gguf")
    
    base_url = input("\n  Enter base URL (default: http://localhost:8000/v1): ").strip()
    if not base_url:
        base_url = "http://localhost:8000/v1"
    
    model_name = input("  Enter model name (e.g., Qwen/Qwen2.5-14B-Instruct): ").strip()
    if not model_name:
        model_name = "Qwen/Qwen2.5-14B-Instruct"
    
    # Update config
    try:
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        config['llm']['provider'] = 'local'
        config['llm']['base_url'] = base_url
        config['llm']['model_name'] = model_name
        
        with open('config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n  ✓ Local LLM configured")
        print(f"    Base URL: {base_url}")
        print(f"    Model: {model_name}")
        return True
    
    except Exception as e:
        print(f"\n  ✗ Error updating config: {e}")
        return False


def run_demo():
    """Run a quick demo assessment"""
    print("\n[Step 3] Running Demo Assessment...")
    print("\nThis will run a simulated mental health assessment using the framework.")
    
    choice = input("\nProceed with demo? (y/n): ").lower()
    if choice != 'y':
        print("Skipping demo.")
        return
    
    try:
        from main import AgentMental
        
        print("\nInitializing framework...")
        agent = AgentMental("config.yml")
        
        print("\nRunning assessment with simulated user...")
        print("="*70)
        
        user_info = {
            'age': 30,
            'occupation': 'Software Developer',
            'gender': 'Male'
        }
        
        # Run just first 2 topics for quick demo
        agent.start_assessment("demo_user", user_info)
        
        total_score = 0
        for i, topic in enumerate(agent.topics[:2]):  # Just first 2 topics
            score = agent.assess_topic(topic, simulate_user=True)
            total_score += score
            
            if i < 1:  # Not last topic
                input("\nPress Enter to continue to next topic...")
        
        print(f"\n{'='*70}")
        print(f"Demo Assessment Complete!")
        print(f"Partial Score (2 topics): {total_score}")
        print(f"{'='*70}")
        
        print("\n✓ Demo completed successfully!")
        print("\nNext steps:")
        print("  - Run 'python agent_mental.py' for full assessment")
        print("  - Run 'python examples.py' to see more usage examples")
        print("  - Edit 'config.yml' to customize the framework")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that config.yml is properly configured")
        print("  2. For Gemini: Verify GOOGLE_API_KEY is set")
        print("  3. For Local LLM: Ensure inference server is running")
        print("  4. Check the error message above for details")


def show_next_steps():
    """Show next steps after setup"""
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    
    print("\nYou're ready to use AgentMental! Here's what you can do:")
    
    print("\n📖 Read the Documentation:")
    print("  - Check README.md for detailed usage guide")
    print("  - Review config.yml to understand configuration options")
    
    print("\n🚀 Run Examples:")
    print("  - python examples.py        # Interactive examples menu")
    print("  - python agent_mental.py    # Run full assessment")
    
    print("\n🔧 Customize:")
    print("  - Edit config.yml to change assessment scales")
    print("  - Modify agent prompts for different questioning styles")
    print("  - Adjust parameters like max_follow_ups and threshold")
    
    print("\n📚 Learn More:")
    print("  - Read the original paper: arXiv:2508.11567")
    print("  - Explore the code to understand agent interactions")
    
    print("\n" + "="*70)


def main():
    print_header()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install required dependencies first.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 2: Setup configuration
    if not setup_config():
        print("\n❌ Configuration setup failed.")
        sys.exit(1)
    
    # Step 3: Run demo
    run_demo()
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
