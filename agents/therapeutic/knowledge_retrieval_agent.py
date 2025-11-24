"""
Knowledge Retrieval Agent

Supporting agent that retrieves:
- Crisis hotlines by location
- Local mental health services
- Evidence-based resources
- Self-help materials
"""

from typing import Dict, List, Optional
from utils.logger import setup_logger
from utils.llm import LLMOrchestrator

logger = setup_logger('knowledge_retrieval_agent', 'logs/knowledge_retrieval_agent.log')


class KnowledgeRetrievalAgent:
    """
    Knowledge Retrieval Agent for information lookup
    
    This agent can be enhanced with actual web search,
    but for now provides curated knowledge
    """
    
    def __init__(self, llm: LLMOrchestrator, config: Dict):
        self.llm = llm
        self.config = config
        
        # Load knowledge base
        self._load_knowledge_base()
        
        logger.info("Knowledge Retrieval Agent initialized")
    
    def _load_knowledge_base(self):
        """Load curated knowledge base"""
        # Crisis hotlines by country
        self.crisis_hotlines = {
            'US': {
                'name': '988 Suicide & Crisis Lifeline',
                'number': '988',
                'text': 'Text HOME to 741741',
                'description': '24/7 suicide prevention and crisis support'
            },
            'UK': {
                'name': 'Samaritans',
                'number': '116 123',
                'description': '24/7 emotional support'
            },
            'Canada': {
                'name': 'Canada Suicide Prevention Service',
                'number': '1-833-456-4566',
                'text': 'Text 45645',
                'description': '24/7 support in English and French'
            },
            'Australia': {
                'name': 'Lifeline',
                'number': '13 11 14',
                'description': '24/7 crisis support and suicide prevention'
            },
            'International': {
                'name': 'International Association for Suicide Prevention',
                'website': 'https://www.iasp.info/resources/Crisis_Centres/',
                'description': 'Directory of crisis centers worldwide'
            }
        }
        
        # Self-help resources by symptom
        self.self_help_resources = {
            'Sleep Problems': [
                "Sleep hygiene: Consistent sleep schedule, cool dark room",
                "Avoid screens 1 hour before bed",
                "Progressive muscle relaxation exercises",
                "Limit caffeine after 2 PM"
            ],
            'Loss of Interest': [
                "Behavioral activation: Schedule one pleasant activity daily",
                "Start small: 10-minute activities",
                "Social connection: Reach out to one person per week",
                "Gratitude journaling: Note 3 positive things daily"
            ],
            'Fatigue or Low Energy': [
                "Light physical activity: 10-minute walks",
                "Balanced nutrition: Regular meals with protein",
                "Sunlight exposure: 15 minutes daily",
                "Break tasks into smaller chunks"
            ],
            'Depressed Mood': [
                "Mood tracking: Journal daily mood patterns",
                "Challenge negative thoughts: Question automatic thinking",
                "Connect with support: Friends, family, support groups",
                "Mindfulness: 5-minute daily practice"
            ]
        }
        
        # Evidence-based resources
        self.therapy_resources = {
            'CBT': {
                'name': 'Cognitive Behavioral Therapy',
                'description': 'Identifies and changes negative thought patterns',
                'self_help': 'MoodGYM (online CBT program)',
                'books': 'Feeling Good by David Burns'
            },
            'Behavioral Activation': {
                'name': 'Behavioral Activation',
                'description': 'Increases engagement in positive activities',
                'self_help': 'Activity scheduling apps',
                'technique': 'Schedule pleasant activities daily'
            },
            'Mindfulness': {
                'name': 'Mindfulness-Based Approaches',
                'description': 'Present-moment awareness and acceptance',
                'self_help': 'Headspace, Calm apps',
                'technique': '5-minute daily meditation'
            }
        }
        
        logger.info("Knowledge base loaded")
    
    def get_crisis_hotline(self, user_info: Dict) -> Optional[str]:
        """
        Get appropriate crisis hotline based on user location
        
        Args:
            user_info: User information including location
            
        Returns:
            Crisis hotline information or None
        """
        logger.info("Retrieving crisis hotline information")
        
        # Try to determine location (simplified)
        # In production, use IP geolocation or user-provided info
        country = 'US'  # Default
        
        if 'location' in user_info:
            location = user_info['location'].upper()
            if 'UK' in location or 'UNITED KINGDOM' in location:
                country = 'UK'
            elif 'CANADA' in location:
                country = 'Canada'
            elif 'AUSTRALIA' in location:
                country = 'Australia'
        
        hotline_info = self.crisis_hotlines.get(country, self.crisis_hotlines['US'])
        
        # Format output
        output = f"**{hotline_info['name']}**\n"
        output += f"📞 Call: {hotline_info['number']}\n"
        
        if 'text' in hotline_info:
            output += f"📱 Text: {hotline_info['text']}\n"
        
        if 'website' in hotline_info:
            output += f"🌐 Website: {hotline_info['website']}\n"
        
        output += f"ℹ️ {hotline_info['description']}"
        
        logger.info(f"Retrieved hotline for country: {country}")
        
        return output
    
    def get_self_help_resources(self, symptoms: List[Dict]) -> str:
        """
        Get self-help resources based on symptoms
        
        Args:
            symptoms: List of symptom dictionaries with 'symptom' key
            
        Returns:
            Formatted self-help resources
        """
        logger.info(f"Retrieving self-help resources for {len(symptoms)} symptoms")
        
        if not symptoms:
            return ""
        
        resources_text = ""
        
        for symptom in symptoms[:3]:  # Top 3 symptoms
            symptom_name = symptom.get('symptom', '')
            
            if symptom_name in self.self_help_resources:
                resources_text += f"\n**For {symptom_name}:**\n"
                
                for resource in self.self_help_resources[symptom_name]:
                    resources_text += f"• {resource}\n"
        
        if resources_text:
            logger.info("Self-help resources retrieved successfully")
            return resources_text
        else:
            logger.info("No specific resources found for symptoms")
            return ""
    
    def get_local_services(self, user_info: Dict) -> str:
        """
        Get local mental health services
        
        In production, this would integrate with:
        - SAMHSA Treatment Locator
        - Psychology Today directory
        - Insurance provider directories
        
        Args:
            user_info: User information including location
            
        Returns:
            Local services information
        """
        logger.info("Retrieving local mental health services")
        
        # Simplified - provide general directories
        services = """**Finding Mental Health Providers:**

• **SAMHSA National Helpline:** 1-800-662-4357
  - Free, confidential, 24/7
  - Treatment referral and information service

• **Psychology Today Directory:** psychologytoday.com/us/therapists
  - Search by location and insurance
  - Filter by specialty and approach

• **Your Insurance Provider:**
  - Call the number on your insurance card
  - Ask for in-network mental health providers

• **Community Mental Health Centers:**
  - Often provide services on a sliding scale
  - Search: "[Your City] community mental health"
"""
        
        logger.info("General services directory provided")
        
        return services
    
    def get_therapy_technique_info(self, technique: str) -> str:
        """
        Get information about specific therapy technique
        
        Args:
            technique: Therapy technique name (e.g., 'CBT', 'Behavioral Activation')
            
        Returns:
            Technique information
        """
        logger.info(f"Retrieving information for technique: {technique}")
        
        technique_upper = technique.upper()
        
        # Try exact match first
        if technique_upper in self.therapy_resources:
            info = self.therapy_resources[technique_upper]
        # Try partial match
        else:
            info = None
            for key in self.therapy_resources:
                if technique_upper in key or key in technique_upper:
                    info = self.therapy_resources[key]
                    break
        
        if not info:
            logger.warning(f"No information found for technique: {technique}")
            return ""
        
        output = f"**{info['name']}:**\n"
        output += f"{info['description']}\n\n"
        
        if 'self_help' in info:
            output += f"Self-Help Resource: {info['self_help']}\n"
        
        if 'books' in info:
            output += f"Recommended Reading: {info['books']}\n"
        
        if 'technique' in info:
            output += f"Quick Technique: {info['technique']}\n"
        
        logger.info("Technique information retrieved")
        
        return output
    
    def search_web(self, query: str) -> str:
        """
        Web search functionality (placeholder)
        
        In production, integrate with:
        - Google Custom Search API
        - Bing Search API
        - PubMed for research articles
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        logger.info(f"Web search requested: {query}")
        logger.warning("Web search not implemented - using knowledge base")
        
        # For now, return message that web search is not available
        return "Web search functionality not available in current version. Using curated knowledge base."