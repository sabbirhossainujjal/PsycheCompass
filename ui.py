import sys
import subprocess
import os


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    missing = []

    # Check for gradio
    try:
        import gradio
        print(f"  ✓ gradio {gradio.__version__}")
    except ImportError:
        missing.append("gradio")
        print("  ✗ gradio not found")

    # Check for yaml
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError:
        missing.append("pyyaml")
        print("  ✗ pyyaml not found")

    # Check for LLM provider
    has_llm = False
    try:
        import google.generativeai
        print(f"  ✓ google-generativeai (Gemini)")
        has_llm = True
    except ImportError:
        print("  ℹ google-generativeai not found")

    try:
        import openai
        print(f"  ✓ openai")
        has_llm = True
    except ImportError:
        print("  ℹ openai not found")

    if not has_llm:
        print("  ⚠ No LLM provider found (need Gemini or OpenAI)")
        missing.append("google-generativeai or openai")

    if missing:
        print("\n❌ Missing dependencies!")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all UI requirements:")
        print("  pip install -r requirements_ui.txt")
        return False

    print("\n✓ All dependencies satisfied!\n")
    return True


def check_config():
    """Check if config.yml exists and is valid"""
    print("📋 Checking configuration...")

    if not os.path.exists('config.yml'):
        print("  ✗ config.yml not found")
        print("\n⚠ Please ensure config.yml is in the current directory")
        return False

    try:
        import yaml
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)

        # Check for required sections
        if 'llm' not in config:
            print("  ✗ 'llm' section missing from config.yml")
            return False

        if 'assessment_scale' not in config:
            print("  ✗ 'assessment_scale' section missing from config.yml")
            return False

        print(f"  ✓ config.yml loaded")
        print(f"  ✓ LLM Provider: {config['llm']['provider']}")
        print(f"  ✓ Model: {config['llm']['model_name']}")

    except Exception as e:
        print(f"  ✗ Error loading config.yml: {e}")
        return False

    print()
    return True


def check_api_keys():
    """Check if API keys are set"""
    print("🔑 Checking API keys...")

    gemini_key = os.environ.get('GEMINI_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    if gemini_key:
        print(f"  ✓ GEMINI_API_KEY set ({gemini_key[:10]}...)")
    else:
        print("  ℹ GEMINI_API_KEY not set")

    if openai_key:
        print(f"  ✓ OPENAI_API_KEY set ({openai_key[:10]}...)")
    else:
        print("  ℹ OPENAI_API_KEY not set")

    if not gemini_key and not openai_key:
        print("\n⚠ No API keys found")
        print("Set one with:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nOr configure local LLM in config.yml")

    print()
    return True


def check_project_files():
    """Check if required project files exist"""
    print("📁 Checking project files...")

    required_files = [
        'app.py',
        'psychecompass_main.py',
        'assessment_pipeline.py',
        'therapeutic_pipeline.py',
        'utils/llm.py',
        'utils/agents.py',
        'utils/memory.py',
        'therapeutic_agents/router.py'
    ]

    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            missing.append(file)

    if missing:
        print("\n❌ Missing project files!")
        print("Ensure you're in the PsycheCompass project directory")
        return False

    print("\n✓ All project files found!\n")
    return True


def launch_app():
    """Launch the Gradio app"""
    print("="*70)
    print("🚀 Launching PsycheCompass UI...")
    print("="*70)
    print()
    print("The UI will open in your browser at: http://localhost:7860")
    print()
    print("Press Ctrl+C to stop the server")
    print("="*70)
    print()

    try:
        # Launch the app
        import app
        # The app.py file has its own launch code
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("Thank you for using PsycheCompass!")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nCheck logs for details:")
        print("  tail -f logs/psychecompass.log")
        return False

    return True


def main():
    """Main launcher function"""
    print("\n" + "="*70)
    print("🧠 PsycheCompass UI Launcher")
    print("="*70 + "\n")

    # Run all checks
    checks = [
        ("dependencies", check_dependencies),
        ("project files", check_project_files),
        ("configuration", check_config),
        ("API keys", check_api_keys),
    ]

    for check_name, check_func in checks:
        if not check_func():
            print(f"\n❌ {check_name.title()} check failed!")
            print("\nPlease fix the issues above and try again.")
            print("\nFor help, see:")
            print("  - UI_README.md")
            print("  - SETUP_CHECKLIST.md")
            sys.exit(1)

    print("="*70)
    print("✅ All checks passed! Ready to launch.")
    print("="*70 + "\n")

    # Launch
    if launch_app():
        print("\n✓ Server stopped successfully")
    else:
        print("\n✗ Server encountered an error")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Launcher interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
