"""
Updating Agent

Aggregates all topic scores and generates comprehensive final assessment report.
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


class UpdatingAgent(BaseAgent):
    """
    Updating Agent
    
    Responsibilities:
    - Aggregate all topic scores
    - Generate comprehensive final report
    - Provide recommendations
    """
    
    def __init__(self, llm: LLMOrchestrator, config: Dict[str, Any]):
        super().__init__(llm, config, "updating")
        self.prompt_template = config['agent_prompts']['updating']
    
    def generate_report(
        self,
        memory: TreeMemory,
        total_score: int,
        scale_name: str
    ) -> str:
        """
        Generate comprehensive final assessment report.
        
        Args:
            memory: Tree-structured memory with all conversation data
            total_score: Total assessment score
            scale_name: Name of the assessment scale
            
        Returns:
            Final report string
        """
        self.logger.info(f"Generating final report for user: {memory.user.user_id}")
        
        topics_summary = self._compile_topics_summary(memory)
        
        prompt = self.prompt_template.format(
            scale_name=scale_name,
            topics_summary=topics_summary,
            total_score=total_score,
            user_info=json.dumps(memory.user.to_dict(), indent=2)
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        report = self.llm.generate(prompt, max_tokens=2048)
        
        self.logger.info(f"Final report generated (length: {len(report)} chars)")
        self.logger.debug(f"Report preview: {report[:200]}...")
        
        return report
    
    def _compile_topics_summary(self, memory: TreeMemory) -> str:
        """
        Compile summary of all assessed topics.
        
        Args:
            memory: Tree-structured memory
            
        Returns:
            Formatted topics summary string
        """
        self.logger.debug("Compiling topics summary")
        
        summary = ""
        for topic_name, topic_node in memory.topics.items():
            summary += f"\n{topic_name}:\n"
            summary += f"  Score: {topic_node.score}\n"
            summary += f"  Summary: {topic_node.summary}\n"
            if topic_node.basis:
                summary += f"  Basis: {topic_node.basis[:200]}...\n"
        
        self.logger.debug(f"Topics summary compiled ({len(summary)} chars)")
        
        return summary
    
    def process(self, *args, **kwargs):
        """Process wrapper for compatibility"""
        return self.generate_report(
            kwargs['memory'],
            kwargs['total_score'],
            kwargs['scale_name']
        )
