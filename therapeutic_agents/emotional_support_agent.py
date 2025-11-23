"""
Emotional Support Agent

Handles low-risk cases (PHQ-8 Score: 0-4)
Provides validation, psychoeducation, and self-care guidance
"""

from typing import Dict, Optional
from utils.logger import setup_logger
from utils.llm import LLMOrchestrator

logger = setup_logger('emotional_support_agent', 'logs/emotional_support_agent.log')


class EmotionalSupportAgent:
    """
    Emotional Support Agent for low-risk individuals
    
    Approach:
    - Validate feelings without judgment
    - Provide psychoeducation
    - Suggest evidence-based self-care strategies
    - Encourage continued self-monitoring
    - Normalize experiences
    """
    
    def __init__(
        self,
        llm: LLMOrchestrator,
        config: Dict,
        knowledge_agent: Optional[object] = None
    ):
        self.llm = llm
        self.config = config
        self.knowledge_agent = knowledge_agent
        
        # Load agent configuration
        agent_config = config.get('therapeutic_support', {}).get(
            'agents', {}
        ).get('emotional_support', {})
        
        self.techniques = agent_config.get('techniques', [
            'reflective_listening', 'validation', 
            'psychoeducation', 'self_care_guidance'
        ])
        self.max_turns = agent_config.get('max_turns', 5)
        
        # Load prompts
        self.system_prompt_template = config.get('therapeutic_prompts', {}).get(
            'emotional_support', {}
        ).get('system', '')
        
        logger.info("Emotional Support Agent initialized")
        logger.info(f"Techniques: {self.techniques}")
    
    def generate_response(self, assessment_results: Dict) -> str:
        """
        Generate emotional support response
        
        Args:
            assessment_results: Prepared therapeutic input
            
        Returns:
            Therapeutic response text
        """
        logger.info("Generating emotional support response")
        logger.info(f"User PHQ-8 Score: {assessment_results['phq8_score']}")
        
        # Format system prompt
        system_prompt = self._format_system_prompt(assessment_results)
        
        # Create user prompt
        user_prompt = self._create_user_prompt(assessment_results)
        
        # Generate response
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.debug(f"Prompt length: {len(full_prompt)} characters")
        
        try:
            response = self.llm.generate(full_prompt)
            logger.info("Response generated successfully")
            logger.debug(f"Response length: {len(response)} characters")
            
            # Add resources if available
            response = self._add_resources(response, assessment_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._fallback_response(assessment_results)
    
    def _format_system_prompt(self, assessment_results: Dict) -> str:
        """Format system prompt with assessment data"""
        return self.system_prompt_template.format(
            assessment_summary=str(assessment_results['assessment_summary']),
            phq8_score=assessment_results['phq8_score'],
            key_symptoms=assessment_results.get('key_symptoms', [])
        )
    
    def _create_user_prompt(self, assessment_results: Dict) -> str:
        """Create user prompt for emotional support"""
        key_symptoms = assessment_results.get('key_symptoms', [])
        
        prompt = "Based on the assessment, please provide emotional support that includes:\n\n"
        prompt += "1. **Acknowledgment & Validation:**\n"
        prompt += "   - Acknowledge their current experience\n"
        prompt += "   - Validate their feelings as understandable\n\n"
        
        prompt += "2. **Psychoeducation:**\n"
        prompt += "   - Brief explanation of what they're experiencing\n"
        prompt += "   - Normalize their feelings\n"
        prompt += "   - Explain this is manageable\n\n"
        
        prompt += "3. **Self-Care Strategies:**\n"
        prompt += "   - Evidence-based self-care recommendations\n"
        prompt += "   - Specific, actionable suggestions\n"
        
        if len(key_symptoms):
            prompt += "\n\n**Focus areas based on symptoms:**\n"
            for symptom in key_symptoms[:3]:  # Top 3 symptoms
                prompt += f"- {symptom['symptom']}: {symptom['severity']}\n"
        
        prompt += "\n\n**Tone:** Warm, empathetic, encouraging, non-clinical"
        prompt += "\n**Length:** 2-3 paragraphs"
        
        return prompt
    
    def _add_resources(self, response: str, assessment_results: Dict) -> str:
        """Add helpful resources to response"""
        if not self.knowledge_agent:
            return response
        
        try:
            # Get self-help resources
            resources = self.knowledge_agent.get_self_help_resources(
                symptoms=assessment_results.get('key_symptoms', [])
            )
            
            if resources:
                response += "\n\n**Helpful Resources:**\n"
                response += resources
                
        except Exception as e:
            logger.warning(f"Could not add resources: {e}")
        
        return response
    
    def continue_conversation(
        self,
        user_message: str,
        conversation_history: list,
        assessment_results: Dict
    ) -> str:
        """
        Continue multi-turn conversation
        
        Args:
            user_message: User's latest message
            conversation_history: Previous conversation turns
            assessment_results: Original assessment data
            
        Returns:
            Follow-up response
        """
        logger.info(f"Continuing conversation (turn {len(conversation_history) + 1})")
        
        # Build context from history
        context = self._build_conversation_context(conversation_history)
        
        # Create follow-up prompt
        prompt = f"""You are continuing an emotional support conversation.
        
**Assessment Context:**
{assessment_results['assessment_summary']}

**Conversation So Far:**
{context}

**User's Latest Message:**
{user_message}

**Your Task:**
Respond empathetically to the user's message. Continue to:
- Validate their feelings
- Provide relevant support
- Answer any questions they have
- Encourage positive steps

Keep your response conversational and warm (2-3 sentences).
"""
        
        try:
            response = self.llm.generate(prompt)
            logger.info("Follow-up response generated")
            return response
            
        except Exception as e:
            logger.error(f"Error in follow-up: {e}")
            return "I'm here to listen. Can you tell me more about how you're feeling?"
    
    def _build_conversation_context(self, conversation_history: list) -> str:
        """Build context string from conversation history"""
        context = ""
        for turn in conversation_history[-4:]:  # Last 4 turns
            speaker = "Therapist" if turn['speaker'] == 'therapist' else "User"
            context += f"{speaker}: {turn['message']}\n\n"
        return context
    
    def _fallback_response(self, assessment_results: Dict) -> str:
        """Fallback response if generation fails"""
        logger.warning("Using fallback response")
        
        return f"""I want to acknowledge what you've shared about your current experience. 
Based on your assessment, it's clear you're going through a challenging time, and your 
feelings are completely valid and understandable.

The symptoms you described - such as {', '.join([s['symptom'] for s in assessment_results.get('key_symptoms', [])][:2])} 
- are experiences many people face, and there are effective ways to address them.

Here are some evidence-based strategies that might help:

• **Sleep Hygiene:** Try to maintain a consistent sleep schedule. Avoid screens 1 hour before bed.
• **Physical Activity:** Even a 10-minute walk can improve mood and energy levels.
• **Social Connection:** Reach out to a friend or family member, even if just for a brief chat.
• **Mindfulness:** Try 5 minutes of deep breathing or meditation daily.

I encourage you to continue monitoring how you're feeling. If symptoms persist or worsen, 
consider reaching out to a mental health professional who can provide personalized support.

Remember, taking small steps toward self-care is a sign of strength, not weakness. 
You're already showing that strength by completing this assessment and seeking information.

Is there anything specific you'd like to explore further or any questions I can help with?
"""