"""
Question Generator Agent

Generates initial and follow-up questions for mental health assessment topics.
"""

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


class QuestionGeneratorAgent(BaseAgent):
    """
    Question Generator Agent
    
    Responsibilities:
    - Generate initial questions for each topic
    - Generate follow-up questions to gather more information
    """
    
    def __init__(self, llm: LLMOrchestrator, config: Dict[str, Any]):
        super().__init__(llm, config, "question_generator")
        self.prompts = config['agent_prompts']['question_generator']
    
    def generate_initial_question(
        self, 
        topic_config: Dict[str, Any], 
        memory: TreeMemory
    ) -> str:
        """
        Generate initial question for a topic.
        
        Args:
            topic_config: Configuration for the current topic
            memory: Tree-structured memory with conversation context
            
        Returns:
            Generated question string
        """
        self.logger.info(f"Generating initial question for topic: {topic_config['name']}")
        
        context = memory.get_context_summary(topic_config['name'])
        
        prompt = self.prompts['initial'].format(
            topic=topic_config['name'],
            description=topic_config['description'],
            context=context
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        question = self.llm.generate(prompt)
        question = question.strip()
        
        self.logger.info(f"Generated initial question: {question}...")
        
        return question
    
    def generate_followup_question(
        self,
        topic_config: Dict[str, Any],
        topic_name: str,
        memory: TreeMemory
    ) -> str:
        """
        Generate follow-up question to gather more specific information.
        
        Args:
            topic_config: Configuration for the current topic
            topic_name: Name of the topic
            memory: Tree-structured memory with conversation context
            
        Returns:
            Generated follow-up question string
        """
        self.logger.info(f"Generating follow-up question for topic: {topic_name}")
        
        conversation = memory.get_current_topic_context(topic_name)
        context = memory.get_context_summary(topic_name)
        
        prompt = self.prompts['followup'].format(
            topic=topic_config['name'],
            conversation=conversation,
            context=context
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        question = self.llm.generate(prompt)
        question = question.strip()
        
        self.logger.info(f"Generated follow-up question: {question[:100]}...")
        
        return question
    
    def process(self, *args, **kwargs):
        """Process wrapper for compatibility"""
        question_type = kwargs.get('question_type', 'initial')
        
        if question_type == 'initial':
            return self.generate_initial_question(
                kwargs['topic_config'],
                kwargs['memory']
            )
        else:
            return self.generate_followup_question(
                kwargs['topic_config'],
                kwargs['topic_name'],
                kwargs['memory']
            )
