import os
from enum import Enum
from typing import Optional, Dict, Any
from utils.logger import setup_logger
from dotenv import load_dotenv
load_dotenv()
logger = setup_logger('llm', 'logs/llm.log')


class LLMProvider(Enum):
    """Supported LLM providers"""
    LOCAL = "local"
    GEMINI = "gemini"
    OPENAI = "openai"


class LLMOrchestrator:
    """
    Unified LLM orchestration interface.
    Manages interactions with different LLM providers.
    """

    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing LLM Orchestrator")

        self.provider = LLMProvider(config['llm']['provider'])
        self.model_name = config['llm']['model_name']
        self.temperature = config['llm'].get('temperature', 0.0)
        self.max_tokens = config['llm'].get('max_tokens', 1024)

        logger.info(f"Provider: {self.provider.value}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Temperature: {self.temperature}")

        # Initialize provider
        if self.provider == LLMProvider.GEMINI:
            try:
                self._init_gemini(api_key=os.getenv("GEMINI_API_KEY"))
            except:
                self._init_gemini(config['llm'].get('api_key'))
        elif self.provider == LLMProvider.OPENAI:
            self._init_openai(config['llm'].get('api_key'))
        else:
            self._init_local(config['llm'])

        logger.info("LLM Orchestrator initialized successfully")

    def _init_gemini(self, api_key: Optional[str]):
        """Initialize Gemini API"""
        logger.info("Initializing Gemini API")

        try:
            import google.generativeai as genai

            # Get API key from environment if not provided
            if not api_key:
                api_key = os.environ.get('GEMINI_API_KEY')

            if not api_key:
                logger.error("Gemini API key not found")
                raise ValueError(
                    "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                    "or provide api_key in config.yml"
                )

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)

            logger.info("Gemini API initialized successfully")

        except ImportError:
            logger.error("google-generativeai package not installed")
            raise ImportError(
                "Please install google-generativeai: pip install google-generativeai"
            )

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI API"""
        logger.info("Initializing OpenAI API")

        try:
            import openai

            # Get API key from environment if not provided
            if not api_key:
                api_key = os.environ.get('OPENAI_API_KEY')

            if not api_key:
                logger.error("OpenAI API key not found")
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or provide api_key in config.yml"
                )

            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI API initialized successfully")

        except ImportError:
            logger.error("openai package not installed")
            raise ImportError("Please install openai: pip install openai")

    def _init_local(self, llm_config: Dict[str, Any]):
        """Initialize local LLM"""
        logger.info("Initializing Local LLM")

        self.base_url = llm_config.get('base_url', 'http://localhost:8000/v1')
        self.local_type = llm_config.get('local_type', 'openai_compatible')

        logger.info(f"Local LLM base URL: {self.base_url}")
        logger.info(f"Local LLM type: {self.local_type}")

        if self.local_type == 'openai_compatible':
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key="dummy"  # Many local servers don't need real keys
                )
                logger.info("OpenAI-compatible local LLM initialized")
            except ImportError:
                logger.error("openai package not installed for local LLM")
                raise ImportError(
                    "For OpenAI-compatible local LLMs, install: pip install openai"
                )
        else:
            logger.warning(
                f"Local LLM type '{self.local_type}' - using mock mode")
            logger.warning(
                "Implement actual integration for your local LLM setup")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        logger.debug(
            f"Generating response (prompt length: {len(prompt)} chars)")

        try:
            if self.provider == LLMProvider.GEMINI:
                response = self._generate_gemini(prompt, **kwargs)
            elif self.provider == LLMProvider.OPENAI:
                response = self._generate_openai(prompt, **kwargs)
            else:
                response = self._generate_local(prompt, **kwargs)

            logger.debug(f"Response generated (length: {len(response)} chars)")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise

    def _generate_gemini(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini API"""
        logger.debug("Using Gemini API for generation")

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens
        }

        logger.debug(f"Generation config: {generation_config}")

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
        )

        if response.parts:
            return response.text
        else:
            logger.warning(
                f"Gemini response blocked. Full response: {response}")
            return "Response blocked by API."

    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API"""
        logger.debug("Using OpenAI API for generation")

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate response using local LLM"""
        logger.debug("Using Local LLM for generation")

        if self.local_type == 'openai_compatible':
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Local LLM generation failed: {e}")
                logger.warning("Falling back to mock response")
                return self._mock_local_response(prompt)
        else:
            return self._mock_local_response(prompt)

    def _mock_local_response(self, prompt: str) -> str:
        """
        Mock response for demonstration/testing.
        Replace with actual local LLM integration.
        """
        logger.warning(
            "Using mock response - implement actual local LLM integration")

        # Simple mock based on prompt content
        if "question" in prompt.lower():
            if "initial" in prompt.lower():
                return "Over the past two weeks, have you been experiencing any challenges in this area?"
            else:
                return "Can you tell me more about how often this occurs and how it affects you?"

        elif "evaluate" in prompt.lower() or "adequacy" in prompt.lower():
            return "1\nSome information is present, but more details about frequency and impact would be helpful."

        elif "score" in prompt.lower():
            return "Score: 2\n\nSummary: User reports moderate symptoms affecting daily life.\n\nBasis: Based on the conversation, the user indicated experiencing the symptom more than half the days, with noticeable impact on daily activities."

        elif "report" in prompt.lower():
            return """
OVERALL ASSESSMENT:
The assessment indicates moderate symptoms requiring attention.

DETAILED FINDINGS:
Multiple symptoms identified with varying severity levels.

RECOMMENDATIONS:
Consider professional consultation for comprehensive evaluation.
Follow-up assessment recommended in 2-4 weeks.
"""

        return "[Mock Response] Please implement actual local LLM integration"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            'provider': self.provider.value,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

        if self.provider == LLMProvider.LOCAL:
            info['base_url'] = self.base_url
            info['local_type'] = self.local_type

        logger.debug(f"Model info: {info}")
        return info

    def test_connection(self) -> bool:
        """Test LLM connection"""
        logger.info("Testing LLM connection")

        try:
            response = self.generate(
                "Hello, this is a test. Please respond with 'OK'.")
            logger.info(
                f"Connection test successful. Response: {response[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
