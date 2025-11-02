from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

# ============================================================================ 
# Memory Structures (Tree-based)
# ============================================================================ 

memory_logger = get_logger("memory", "logs/memory.log")

@dataclass
class StatementNode:
    """Statement node - stores turn-level information"""
    emotion: str = ""
    frequency: str = ""
    duration: str = ""
    symptom: str = ""
    impact: str = ""
    
    def to_dict(self):
        return {
            "emotion": self.emotion,
            "frequency": self.frequency,
            "duration": self.duration,
            "symptom": self.symptom,
            "impact": self.impact
        }

@dataclass
class TopicNode:
    """Topic node - stores topic-level assessment"""
    topic: str
    score: int = 0
    summary: str = ""
    basis: str = ""
    statements: List[StatementNode] = field(default_factory=list)
    qa_pairs: List[Tuple[str, str]] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "topic": self.topic,
            "score": self.score,
            "summary": self.summary,
            "basis": self.basis,
            "statements": [s.to_dict() for s in self.statements],
            "qa_pairs": self.qa_pairs
        }

@dataclass
class UserNode:
    """Root node - stores user basic information"""
    user_id: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    gender: Optional[str] = None
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "age": self.age,
            "occupation": self.occupation,
            "gender": self.gender
        }

class TreeMemory:
    """Tree-structured memory for conversation context"""
    def __init__(self, user_id: str):
        self.user = UserNode(user_id=user_id)
        self.topics: Dict[str, TopicNode] = {}
        memory_logger.info(f"TreeMemory initialized for user {user_id}")
    
    def add_topic_node(self, topic: str):
        if topic not in self.topics:
            self.topics[topic] = TopicNode(topic=topic)
            memory_logger.info(f"Added topic node: {topic}")
    
    def add_statement(self, topic: str, statement: StatementNode):
        if topic in self.topics:
            self.topics[topic].statements.append(statement)
            memory_logger.info(f"Added statement to topic {topic}: {statement.to_dict()}")
    
    def add_qa_pair(self, topic: str, question: str, answer: str):
        if topic in self.topics:
            self.topics[topic].qa_pairs.append((question, answer))
            memory_logger.info(f"Added Q&A pair to topic {topic}: Q: {question} A: {answer}")
    
    def update_topic_score(self, topic: str, score: int, summary: str, basis: str):
        if topic in self.topics:
            self.topics[topic].score = score
            self.topics[topic].summary = summary
            self.topics[topic].basis = basis
            memory_logger.info(f"Updated topic {topic} with score: {score}")
    
    def get_context_summary(self, current_topic: Optional[str] = None) -> str:
        """Generate context summary for agents"""
        summary = f"User Info: {self.user.to_dict()}\n\n"
        summary += "Previous Topics:\n"
        for topic_name, topic_node in self.topics.items():
            if current_topic and topic_name == current_topic:
                continue
            if topic_node.score > 0 or topic_node.summary:
                summary += f"- {topic_name}: Score {topic_node.score}, {topic_node.summary}\n"
        return summary
    
    def get_current_topic_context(self, topic: str) -> str:
        """Get current topic conversation history"""
        if topic not in self.topics:
            return ""
        
        topic_node = self.topics[topic]
        context = f"Current Topic: {topic}\n"
        context += "Conversation History:\n"
        for i, (q, a) in enumerate(topic_node.qa_pairs, 1):
            context += f"Q{i}: {q}\nA{i}: {a}\n"
        return context
