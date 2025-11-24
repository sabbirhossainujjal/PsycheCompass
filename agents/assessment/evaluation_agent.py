"""
Evaluation Agent

Evaluates adequacy of user responses and determines if follow-up questions are needed.
"""

import json
from typing import Dict, Any
from abc import ABC, abstractmethod

from utils.logger import setup_logger
from utils.memory import TreeMemory
from utils.llm import LLMOrchestrator


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, llm: LLMOrchestrator, config: Dict[str, Any], name: str):
        self.llm = llm
        self.config = config
        self.name = name
        self.logger = setup_logger(f'agent_{name}', f'logs/agent_{name}.log')
        self.logger.info(f"{name} initialized")
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process method to be implemented by subclasses"""
        pass


class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent
    
    Responsibilities:
    - Evaluate adequacy of user responses
    - Determine if follow-up questions are needed
    - Return necessity score (0-2)
    """
    
    def __init__(self, llm: LLMOrchestrator, config: Dict[str, Any]):
        super().__init__(llm, config, "evaluation")
        self.prompt_template = config['agent_prompts']['evaluation']
    
    def evaluate_adequacy(
        self,
        topic_config: Dict[str, Any],
        topic_name: str,
        memory: TreeMemory
    ) -> float:
        """
        Evaluate if user responses are adequate for scoring.
        
        Args:
            topic_config: Configuration for the current topic
            topic_name: Name of the topic
            memory: Tree-structured memory with conversation context
            
        Returns:
            Necessity score (0.0 = adequate, 1.0 = needs clarification, 2.0 = insufficient)
        """
        self.logger.info(f"Evaluating response adequacy for topic: {topic_name}")
        
        conversation = memory.get_current_topic_context(topic_name)
        
        prompt = self.prompt_template.format(
            topic=topic_config['name'],
            conversation=conversation,
            rating_criteria=json.dumps(topic_config['rating_criteria'], indent=2)
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        response = self.llm.generate(prompt)
        
        self.logger.debug(f"Evaluation response: {response[:200]}...")
        
        necessity_score = self._parse_necessity_score(response)
        
        self.logger.info(f"Necessity score: {necessity_score}")
        
        return necessity_score
    
    def _parse_necessity_score(self, response: str) -> float:
        """
        Parse necessity score from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Necessity score (0.0, 1.0, or 2.0)
        """
        lines = response.strip().split('\n')
        first_line = lines[0] if lines else response
        
        for word in first_line.split():
            cleaned = word.strip('.:,;!?')
            if cleaned.replace('.', '').isdigit():
                score = float(cleaned)
                if 0 <= score <= 2:
                    self.logger.debug(f"Parsed necessity score: {score}")
                    return score
        
        for word in response.split():
            cleaned = word.strip('.:,;!?')
            if cleaned.replace('.', '').isdigit():
                score = float(cleaned)
                if 0 <= score <= 2:
                    self.logger.debug(f"Parsed necessity score (fallback): {score}")
                    return score
        
        self.logger.warning("Failed to parse necessity score, defaulting to 1.0")
        return 1.0
    
    def process(self, *args, **kwargs):
        """Process wrapper for compatibility"""
        return self.evaluate_adequacy(
            kwargs['topic_config'],
            kwargs['topic_name'],
            kwargs['memory']
        )
