from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger('memory', 'logs/memory.log')


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
    """
    Tree-structured memory for conversation context management.
    
    Structure:
        UserNode (root)
        └── TopicNodes (children)
            └── StatementNodes (leaves)
    """
    
    def __init__(self, user_id: str):
        logger.info(f"Initializing TreeMemory for user: {user_id}")
        self.user = UserNode(user_id=user_id)
        self.topics: Dict[str, TopicNode] = {}
        logger.debug(f"TreeMemory initialized with user_id: {user_id}")
    
    def add_topic_node(self, topic: str):
        """Add a new topic node to the memory tree"""
        if topic not in self.topics:
            self.topics[topic] = TopicNode(topic=topic)
            logger.info(f"Added topic node: {topic}")
        else:
            logger.debug(f"Topic node already exists: {topic}")
    
    def add_statement(self, topic: str, statement: StatementNode):
        """Add a statement node to a specific topic"""
        if topic in self.topics:
            self.topics[topic].statements.append(statement)
            logger.debug(f"Added statement to topic '{topic}': {statement.to_dict()}")
        else:
            logger.warning(f"Attempted to add statement to non-existent topic: {topic}")
    
    def add_qa_pair(self, topic: str, question: str, answer: str):
        """Add a question-answer pair to a topic"""
        if topic in self.topics:
            self.topics[topic].qa_pairs.append((question, answer))
            logger.debug(f"Added Q&A pair to topic '{topic}' (Q: {question[:50]}...)")
        else:
            logger.warning(f"Attempted to add Q&A to non-existent topic: {topic}")
    
    def update_topic_score(self, topic: str, score: int, summary: str, basis: str):
        """Update the assessment score and summary for a topic"""
        if topic in self.topics:
            self.topics[topic].score = score
            self.topics[topic].summary = summary
            self.topics[topic].basis = basis
            logger.info(f"Updated topic '{topic}' - Score: {score}")
            logger.debug(f"Summary: {summary[:100]}...")
        else:
            logger.warning(f"Attempted to update non-existent topic: {topic}")
    
    def get_context_summary(self, current_topic: Optional[str] = None) -> str:
        """
        Generate context summary for agents.
        Excludes the current topic being assessed.
        """
        logger.debug(f"Generating context summary (excluding: {current_topic})")
        
        summary = f"User Info: {self.user.to_dict()}\n\n"
        summary += "Previous Topics:\n"
        
        included_topics = 0
        for topic_name, topic_node in self.topics.items():
            if current_topic and topic_name == current_topic:
                continue
            if topic_node.score > 0 or topic_node.summary:
                summary += f"- {topic_name}: Score {topic_node.score}, {topic_node.summary}\n"
                included_topics += 1
        
        if included_topics == 0:
            summary += "  (No previous topics completed yet)\n"
        
        logger.debug(f"Context summary generated with {included_topics} previous topics")
        return summary
    
    def get_current_topic_context(self, topic: str) -> str:
        """
        Get current topic conversation history.
        Used by agents to understand the ongoing conversation.
        """
        logger.debug(f"Retrieving conversation history for topic: {topic}")
        
        if topic not in self.topics:
            logger.warning(f"Topic not found in memory: {topic}")
            return ""
        
        topic_node = self.topics[topic]
        context = f"Current Topic: {topic}\n"
        context += "Conversation History:\n"
        
        for i, (q, a) in enumerate(topic_node.qa_pairs, 1):
            context += f"Q{i}: {q}\nA{i}: {a}\n"
        
        logger.debug(f"Retrieved {len(topic_node.qa_pairs)} Q&A pairs for topic '{topic}'")
        return context
    
    def extract_information(self, answer: str) -> StatementNode:
        """
        Extract key information from user answer.
        Identifies: emotion, frequency, duration, symptom, impact
        """
        logger.debug("Extracting information from user answer")
        
        statement = StatementNode()
        answer_lower = answer.lower()
        
        # Emotion keywords
        if any(word in answer_lower for word in ['sad', 'depressed', 'down', 'hopeless', 'miserable']):
            statement.emotion = "negative"
            logger.debug("Detected emotion: negative")
        elif any(word in answer_lower for word in ['anxious', 'worried', 'nervous', 'scared', 'fearful']):
            statement.emotion = "anxious"
            logger.debug("Detected emotion: anxious")
        elif any(word in answer_lower for word in ['angry', 'irritated', 'frustrated']):
            statement.emotion = "anger"
            logger.debug("Detected emotion: anger")
        
        # Frequency keywords
        if any(word in answer_lower for word in ['always', 'constantly', 'every day', 'daily', 'all the time']):
            statement.frequency = "high"
            logger.debug("Detected frequency: high")
        elif any(word in answer_lower for word in ['often', 'frequently', 'regularly', 'most days']):
            statement.frequency = "medium"
            logger.debug("Detected frequency: medium")
        elif any(word in answer_lower for word in ['sometimes', 'occasionally', 'few times', 'now and then']):
            statement.frequency = "low"
            logger.debug("Detected frequency: low")
        elif any(word in answer_lower for word in ['rarely', 'seldom', 'not often']):
            statement.frequency = "very low"
            logger.debug("Detected frequency: very low")
        
        # Duration keywords
        if any(word in answer_lower for word in ['weeks', 'months', 'long time', 'years']):
            statement.duration = "extended"
            logger.debug("Detected duration: extended")
        elif any(word in answer_lower for word in ['days', 'recently', 'past few days']):
            statement.duration = "short"
            logger.debug("Detected duration: short")
        
        # Impact keywords
        if any(word in answer_lower for word in ['difficult', 'hard', 'struggle', 'affect', 'impact', 'interfere']):
            statement.impact = "significant"
            logger.debug("Detected impact: significant")
        elif any(word in answer_lower for word in ['little', 'minor', 'slight', 'barely']):
            statement.impact = "minimal"
            logger.debug("Detected impact: minimal")
        
        # Symptom presence
        if any(word in answer_lower for word in ['yes', 'experiencing', 'having', 'feeling']):
            statement.symptom = "present"
            logger.debug("Detected symptom: present")
        elif any(word in answer_lower for word in ['no', 'not', "don't", "haven't"]):
            statement.symptom = "absent"
            logger.debug("Detected symptom: absent")
        
        logger.info(f"Information extracted: {statement.to_dict()}")
        return statement
    
    def get_topic_summary(self, topic: str) -> Dict:
        """Get summary of a specific topic"""
        if topic in self.topics:
            logger.debug(f"Retrieving summary for topic: {topic}")
            return self.topics[topic].to_dict()
        logger.warning(f"Topic not found for summary: {topic}")
        return {}
    
    def get_all_topics_summary(self) -> Dict[str, Dict]:
        """Get summary of all topics"""
        logger.debug("Retrieving summary of all topics")
        return {name: node.to_dict() for name, node in self.topics.items()}
    
    def clear_topic(self, topic: str):
        """Clear a specific topic from memory"""
        if topic in self.topics:
            del self.topics[topic]
            logger.info(f"Cleared topic from memory: {topic}")
        else:
            logger.warning(f"Attempted to clear non-existent topic: {topic}")
    
    def clear_all(self):
        """Clear all topics from memory (keep user info)"""
        topic_count = len(self.topics)
        self.topics.clear()
        logger.info(f"Cleared all topics from memory ({topic_count} topics removed)")
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the current memory state"""
        stats = {
            'user_id': self.user.user_id,
            'total_topics': len(self.topics),
            'completed_topics': sum(1 for t in self.topics.values() if t.score > 0),
            'total_qa_pairs': sum(len(t.qa_pairs) for t in self.topics.values()),
            'total_statements': sum(len(t.statements) for t in self.topics.values())
        }
        logger.debug(f"Memory stats: {stats}")
        return stats
    
    def __repr__(self):
        return f"TreeMemory(user_id={self.user.user_id}, topics={len(self.topics)})"
