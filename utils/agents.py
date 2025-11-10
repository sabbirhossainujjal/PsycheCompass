"""
Agent Module

Implements the four specialized agents:
- QuestionGeneratorAgent (AGq)
- EvaluationAgent (AGev)
- ScoringAgent (AGs)
- UpdatingAgent (AGu)
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


class QuestionGeneratorAgent(BaseAgent):
    """
    AGq: Question Generator Agent
    
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
        
        # Get context from memory
        context = memory.get_context_summary(topic_config['name'])
        
        # Format prompt
        prompt = self.prompts['initial'].format(
            topic=topic_config['name'],
            description=topic_config['description'],
            context=context
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Generate question
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
        
        # Get current topic conversation
        conversation = memory.get_current_topic_context(topic_name)
        
        # Get broader context
        context = memory.get_context_summary(topic_name)
        
        # Format prompt
        prompt = self.prompts['followup'].format(
            topic=topic_config['name'],
            conversation=conversation,
            context=context
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Generate follow-up question
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


class EvaluationAgent(BaseAgent):
    """
    AGev: Evaluation Agent
    
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
        
        # Get current conversation
        conversation = memory.get_current_topic_context(topic_name)
        
        # Format prompt
        prompt = self.prompt_template.format(
            topic=topic_config['name'],
            conversation=conversation,
            rating_criteria=json.dumps(topic_config['rating_criteria'], indent=2)
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Generate evaluation
        response = self.llm.generate(prompt)
        
        self.logger.debug(f"Evaluation response: {response[:200]}...")
        
        # Parse necessity score
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
        # Try to extract number from first line or first few words
        lines = response.strip().split('\n')
        first_line = lines[0] if lines else response
        
        # Look for score in first line
        for word in first_line.split():
            cleaned = word.strip('.:,;!?')
            if cleaned.replace('.', '').isdigit():
                score = float(cleaned)
                if 0 <= score <= 2:
                    self.logger.debug(f"Parsed necessity score: {score}")
                    return score
        
        # Fallback: look in entire response
        for word in response.split():
            cleaned = word.strip('.:,;!?')
            if cleaned.replace('.', '').isdigit():
                score = float(cleaned)
                if 0 <= score <= 2:
                    self.logger.debug(f"Parsed necessity score (fallback): {score}")
                    return score
        
        # Default to 1.0 if parsing fails
        self.logger.warning("Failed to parse necessity score, defaulting to 1.0")
        return 1.0
    
    def process(self, *args, **kwargs):
        """Process wrapper for compatibility"""
        return self.evaluate_adequacy(
            kwargs['topic_config'],
            kwargs['topic_name'],
            kwargs['memory']
        )


class ScoringAgent(BaseAgent):
    """
    AGs: Scoring Agent
    
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
        
        # Get conversation history
        conversation = memory.get_current_topic_context(topic_name)
        
        # Format prompt
        prompt = self.prompt_template.format(
            topic=topic_config['name'],
            conversation=conversation,
            rating_criteria=json.dumps(topic_config['rating_criteria'], indent=2)
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Generate score and explanation
        response = self.llm.generate(prompt)
        
        self.logger.debug(f"Scoring response: {response[:300]}...")
        
        # Parse response
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
        # Look for "Score: X" pattern
        if "Score:" in response or "score:" in response:
            for line in response.split('\n'):
                if 'score:' in line.lower():
                    # Extract number after "Score:"
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
        
        # Fallback: look for first number in valid range
        for word in response.split():
            cleaned = word.strip('.:,;!?')
            if cleaned.isdigit():
                score = int(cleaned)
                if score in topic_config['rating_criteria']:
                    self.logger.debug(f"Parsed score (fallback): {score}")
                    return score
        
        # Default to 0 if parsing fails
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
        # Look for "Summary:" section
        if "Summary:" in response or "summary:" in response:
            for i, line in enumerate(response.split('\n')):
                if 'summary:' in line.lower():
                    # Get text after "Summary:"
                    summary_text = line.split(':', 1)[1].strip()
                    
                    # If empty, try next line
                    if not summary_text and i + 1 < len(response.split('\n')):
                        summary_text = response.split('\n')[i + 1].strip()
                    
                    self.logger.debug(f"Parsed summary: {summary_text[:100]}...")
                    return summary_text
        
        # Fallback: use first 200 chars
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
        # Look for "Basis:" section
        if "Basis:" in response or "basis:" in response:
            for i, line in enumerate(response.split('\n')):
                if 'basis:' in line.lower():
                    # Get everything after "Basis:"
                    remaining_lines = response.split('\n')[i:]
                    basis_text = '\n'.join(remaining_lines)
                    basis_text = basis_text.split(':', 1)[1].strip()
                    
                    self.logger.debug(f"Parsed basis: {basis_text[:100]}...")
                    return basis_text
        
        # Fallback: use entire response
        self.logger.debug(f"Using entire response as basis")
        return response
    
    def process(self, *args, **kwargs):
        """Process wrapper for compatibility"""
        return self.score_topic(
            kwargs['topic_config'],
            kwargs['topic_name'],
            kwargs['memory']
        )


class UpdatingAgent(BaseAgent):
    """
    AGu: Updating Agent
    
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
        
        # Compile topics summary
        topics_summary = self._compile_topics_summary(memory)
        
        # Format prompt
        prompt = self.prompt_template.format(
            scale_name=scale_name,
            topics_summary=topics_summary,
            total_score=total_score,
            user_info=json.dumps(memory.user.to_dict(), indent=2)
        )
        
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Generate report
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
