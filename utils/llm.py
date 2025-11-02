from enum import Enum
from typing import Optional
import os

# ============================================================================ 
# LLM Interface
# ============================================================================ 

class LLMProvider(Enum):
    LOCAL = "local"
    GEMINI = "gemini"

class LLMInterface:
    """Unified interface for different LLM providers"""
    
    def __init__(self, config: dict):
        self.provider = LLMProvider(config['llm']['provider'])
        self.model_name = config['llm']['model_name']
        self.temperature = config['llm'].get('temperature', 0.0)
        
        if self.provider == LLMProvider.GEMINI:
            api_key = config['llm'].get('api_key') or os.getenv('GEMINI_API_KEY')
            self._init_gemini(api_key)
        else:
            self._init_local(config['llm'])
    
    def _init_gemini(self, api_key: Optional[str]):
        """Initialize Gemini API"""
        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def _init_local(self, llm_config: dict):
        """Initialize local LLM (placeholder for local model integration)"""
        # This is a placeholder - in practice, you'd integrate with:
        # - vLLM, llama.cpp, or other local inference engines
        # - Example: from vllm import LLM
        self.base_url = llm_config.get('base_url', 'http://localhost:8000/v1')
        print(f"Local LLM mode - Base URL: {self.base_url}")
        print("Note: Implement actual local LLM integration based on your setup")
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        if self.provider == LLMProvider.GEMINI:
            response = self.model.generate_content(
                prompt,
                generation_config={'temperature': self.temperature}
            )
            return response.text
        else:
            # Placeholder for local LLM
            # In practice, implement API call to local model
            return self._mock_local_response(prompt)
    
    def _mock_local_response(self, prompt: str) -> str:
        """Mock response for demonstration"""
        return "[Local LLM Response] - Implement actual local model integration"
