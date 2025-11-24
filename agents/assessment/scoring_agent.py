"""
Scoring Agent

Analyzes conversation history and assigns quantitative scores based on rating criteria.
"""

import json
from typing import Dict, Any, Tuple
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


class ScoringAgent(BaseAgent):
    """
    Scoring Agent
    
    Responsibilities:
    - Analyze conversation history
    - Assign quantitative scores based on rating criteria
    - Generate summary and basis for the score
    """
    
    def __init__(self, llm: LLMOrchestrator, config: Dict[str, Any]):
        super().__init__(llm, config, "scoring")
        self.prompt_template = config['agent_prompts']['scoring']
    
    def score_topic(
        self,
        topic_config: Dict[str, Any],
        topic_name: str,
        memory: TreeMemory
    ) -> Tuple[int, str, str]:
        """
        Score a topic based on conversation history.
        
        Args:
            topic_config: Configuration for the current topic
            topic_name: Name of the topic
            memory: Tree-structured memory with conversation context
            
        Returns:
            Tuple of (score, summary, basis)
        """
        self.logger.info(f"Scoring topic: {topic_name}")
        
        conversation = memory.get_current_topic_context(topic_name)
        
        prompt = self.prompt_template.format(
            topic=topic_config['name'],
            conversation=conversation,
            rating_criteria=json.dumps(topic_config['rating_criteria'], indent=2)
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        response = self.llm.generate(prompt)
        
        self.logger.debug(f"Scoring response: {response[:300]}...")
        
        score = self._parse_score(response, topic_config)
        summary = self._parse_summary(response)
        basis = self._parse_basis(response)
        
        self.logger.info(f"Topic '{topic_name}' scored: {score}")
        self.logger.debug(f"Summary: {summary[:100]}...")
        
        return score, summary, basis
    
    def _parse_score(self, response: str, topic_config: Dict[str, Any]) -> int:
        """
        Parse score from LLM response.
        
        Args:
            response: LLM response text
            topic_config: Topic configuration with rating criteria
            
        Returns:
            Score (0-3 or as defined in rating criteria)
        """
        if "Score:" in response or "score:" in response:
            for line in response.split('\n'):
                if 'score:' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_text = parts[1].strip()
                        for word in score_text.split():
                            cleaned = word.strip('.:,;!?')
                            if cleaned.isdigit():
                                score = int(cleaned)
                                if score in topic_config['rating_criteria']:
                                    self.logger.debug(f"Parsed score: {score}")
                                    return score
        
        for word in response.split():
            cleaned = word.strip('.:,;!?')
            if cleaned.isdigit():
                score = int(cleaned)
                if score in topic_config['rating_criteria']:
                    self.logger.debug(f"Parsed score (fallback): {score}")
                    return score
        
        self.logger.warning("Failed to parse score, defaulting to 0")
        return 0
    
    def _parse_summary(self, response: str) -> str:
        """
        Parse summary from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Summary string
        """
        if "Summary:" in response or "summary:" in response:
            for i, line in enumerate(response.split('\n')):
                if 'summary:' in line.lower():
                    summary_text = line.split(':', 1)[1].strip()
                    
                    if not summary_text and i + 1 < len(response.split('\n')):
                        summary_text = response.split('\n')[i + 1].strip()
                    
                    self.logger.debug(f"Parsed summary: {summary_text[:100]}...")
                    return summary_text
        
        summary = response[:200].strip()
        self.logger.debug(f"Parsed summary (fallback): {summary[:100]}...")
        return summary
    
    def _parse_basis(self, response: str) -> str:
        """
        Parse basis/explanation from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Basis string
        """
        if "Basis:" in response or "basis:" in response:
            for i, line in enumerate(response.split('\n')):
                if 'basis:' in line.lower():
                    remaining_lines = response.split('\n')[i:]
                    basis_text = '\n'.join(remaining_lines)
                    basis_text = basis_text.split(':', 1)[1].strip()
                    
                    self.logger.debug(f"Parsed basis: {basis_text[:100]}...")
                    return basis_text
        
        self.logger.debug(f"Using entire response as basis")
        return response
    
    def process(self, *args, **kwargs):
        """Process wrapper for compatibility"""
        return self.score_topic(
            kwargs['topic_config'],
            kwargs['topic_name'],
            kwargs['memory']
        )
