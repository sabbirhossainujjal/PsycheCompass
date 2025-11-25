from typing import Dict, Optional
from utils.logger import setup_logger
from utils.llm import LLMOrchestrator

logger = setup_logger('therapeutic_agent', 'logs/therapeutic_agent.log')


class TherapeuticAgent:
    """
    Therapeutic Agent for moderate-risk individuals
    
    Approach:
    - Evidence-based interventions (CBT, BA, PST, Mindfulness)
    - Structured therapeutic techniques
    - Specific homework assignments
    - STRONG professional referral
    - Safety assessment for suicide ideation
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
        ).get('therapeutic', {})
        
        self.techniques = agent_config.get('techniques', [
            'cbt', 'behavioral_activation', 
            'problem_solving', 'mindfulness'
        ])
        self.max_turns = agent_config.get('max_turns', 10)
        self.follow_up_scheduling = agent_config.get('follow_up_scheduling', True)
        
        # Load prompts
        self.system_prompt_template = config.get('therapeutic_prompts', {}).get(
            'therapeutic', {}
        ).get('system', '')
        
        logger.info("Therapeutic Agent initialized")
        logger.info(f"Techniques: {self.techniques}")
        logger.info(f"Max turns: {self.max_turns}")
    
    def generate_response(self, assessment_results: Dict) -> str:
        """
        Generate therapeutic intervention response
        
        Args:
            assessment_results: Prepared therapeutic input
            
        Returns:
            Therapeutic response text
        """
        logger.info("Generating therapeutic intervention")
        logger.info(f"User PHQ-8 Score: {assessment_results['phq8_score']}")
        
        # Select appropriate technique
        technique = self._select_technique(assessment_results)
        logger.info(f"Selected technique: {technique}")
        
        # Format system prompt
        system_prompt = self._format_system_prompt(assessment_results)
        
        # Create user prompt
        user_prompt = self._create_user_prompt(assessment_results, technique)
        
        # Generate response
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.debug(f"Prompt length: {len(full_prompt)} characters")
        
        try:
            response = self.llm.generate(full_prompt, max_tokens=2048)
            logger.info("Response generated successfully")
            logger.debug(f"Response length: {len(response)} characters")
            
            # Add technique information if available
            response = self._add_technique_info(response, technique, assessment_results)
            
            # Add resources
            response = self._add_resources(response, assessment_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._fallback_response(assessment_results)
    
    def _select_technique(self, assessment_results: Dict) -> str:
        """
        Select most appropriate therapeutic technique
        based on key symptoms
        """
        key_symptoms = assessment_results.get('key_symptoms', [])
        
        if not key_symptoms:
            return 'cbt'  # Default
        
        # Map symptoms to techniques
        symptom_names = [s['symptom'] for s in key_symptoms]
        
        # Behavioral Activation for loss of interest, low energy
        if any(s in symptom_names for s in ['Loss of Interest', 'Fatigue or Low Energy']):
            return 'behavioral_activation'
        
        # CBT for depressed mood, negative thoughts
        if any(s in symptom_names for s in ['Depressed Mood', 'Low Self-Worth']):
            return 'cbt'
        
        # Problem-solving for concentration, multiple issues
        if 'Concentration Difficulties' in symptom_names or len(key_symptoms) >= 4:
            return 'problem_solving'
        
        # Mindfulness for sleep, psychomotor changes
        if any(s in symptom_names for s in ['Sleep Problems', 'Psychomotor Changes']):
            return 'mindfulness'
        
        return 'cbt'  # Default
    
    def _format_system_prompt(self, assessment_results: Dict) -> str:
        """Format system prompt with assessment data"""
        key_symptoms = [s['symptom'] for s in assessment_results.get('key_symptoms', [])]
        
        return self.system_prompt_template.format(
            assessment_summary=assessment_results['assessment_summary'],
            phq8_score=assessment_results['phq8_score'],
            key_symptoms=', '.join(key_symptoms) if key_symptoms else 'Multiple symptoms'
        )
    
    def _create_user_prompt(self, assessment_results: Dict, technique: str) -> str:
        """Create user prompt for therapeutic intervention"""
        key_symptoms = assessment_results.get('key_symptoms', [])
        
        technique_descriptions = {
            'cbt': """
**Cognitive Behavioral Therapy (CBT):**
- Identify negative thought patterns
- Challenge automatic thoughts
- Restructure thinking
- Link thoughts to feelings and behaviors
""",
            'behavioral_activation': """
**Behavioral Activation:**
- Schedule pleasant activities
- Break down overwhelming tasks
- Increase engagement with life
- Build momentum through action
""",
            'problem_solving': """
**Problem-Solving Therapy:**
- Identify specific problems
- Generate potential solutions
- Evaluate options
- Create action plan
""",
            'mindfulness': """
**Mindfulness-Based Approach:**
- Present-moment awareness
- Acceptance without judgment
- Grounding exercises
- Meditation techniques
"""
        }
        
        prompt = f"""Based on the assessment, provide therapeutic intervention using {technique.upper()}.

**Structure your response as follows:**

1. **Validation & Acknowledgment** (2-3 sentences)
   - Acknowledge the severity of their experience
   - Validate that these symptoms are significant and deserve attention
   - Express empathy for their struggle

2. **Psychoeducation** (1 paragraph)
   - Brief explanation of moderate depression
   - Why their symptoms make sense given their situation
   - Hope that evidence-based interventions can help

3. **{technique.upper()} Intervention** (2-3 paragraphs)
{technique_descriptions.get(technique, '')}
   
   - Explain the technique clearly
   - Provide a specific example relevant to their symptoms
   - Give detailed homework/practice assignment
   - Make it actionable and concrete

4. **Professional Consultation** (STRONG recommendation)
   - CLEARLY state they should consult a mental health professional
   - Explain that while self-help is valuable, professional support is important for moderate depression
   - Provide guidance on finding a therapist

5. **Safety Check** (1-2 sentences)
   - Ask directly but sensitively about suicide ideation
   - "Are you having any thoughts of harming yourself or suicide?"
   - Make clear this is important to address immediately

6. **Follow-Up Plan** (brief)
   - Suggest checking in on their progress
   - Timeline for reassessment
   - When to seek additional help
"""
        
        if key_symptoms:
            prompt += "\n\n**Primary symptoms to address:**\n"
            for symptom in key_symptoms[:3]:
                prompt += f"- {symptom['symptom']}: {symptom['severity']}\n"
        
        prompt += "\n\n**Tone:** Professional yet warm, action-oriented, hopeful, evidence-based"
        prompt += "\n**Length:** 4-5 paragraphs total"
        
        return prompt
    
    def _add_technique_info(self, response: str, technique: str, assessment_results: Dict) -> str:
        """Add additional technique information"""
        if not self.knowledge_agent:
            return response
        
        try:
            technique_info = self.knowledge_agent.get_therapy_technique_info(technique)
            if technique_info:
                response += "\n\n---\n\n" + technique_info
        except Exception as e:
            logger.warning(f"Could not add technique info: {e}")
        
        return response
    
    def _add_resources(self, response: str, assessment_results: Dict) -> str:
        """Add helpful resources to response"""
        if not self.knowledge_agent:
            return response
        
        try:
            # Get local services info
            user_info = assessment_results.get('user_info', {})
            services = self.knowledge_agent.get_local_services(user_info)
            
            if services:
                response += "\n\n---\n\n" + services
                
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
        Continue multi-turn therapeutic conversation
        
        Args:
            user_message: User's latest message
            conversation_history: Previous conversation turns
            assessment_results: Original assessment data
            
        Returns:
            Follow-up response
        """
        logger.info(f"Continuing therapeutic conversation (turn {len(conversation_history) + 1})")
        
        # Check for safety concerns
        if self._check_safety_concern(user_message):
            logger.warning("Safety concern detected in user message")
            return self._safety_response(user_message)
        
        # Build context from history
        context = self._build_conversation_context(conversation_history)
        
        # Create follow-up prompt
        prompt = f"""You are continuing a therapeutic conversation for moderate depression.

**Assessment Context:**
{assessment_results['assessment_summary']}

**Conversation So Far:**
{context}

**User's Latest Message:**
{user_message}

**Your Task:**
Respond therapeutically to the user's message. Continue to:
- Validate their feelings and experiences
- Provide relevant therapeutic support
- Answer questions about techniques
- Offer additional strategies if needed
- Check on their safety if concerning statements
- Encourage professional consultation
- Maintain hopeful, supportive tone

Keep your response focused and conversational (2-4 paragraphs).
"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=1024)
            logger.info("Follow-up response generated")
            return response
            
        except Exception as e:
            logger.error(f"Error in follow-up: {e}")
            return "Thank you for sharing that. How have you been finding the techniques we discussed? Is there anything specific you'd like to explore further?"
    
    def _check_safety_concern(self, message: str) -> bool:
        """Check if message contains safety concerns"""
        crisis_keywords = ['suicide', 'kill myself', 'want to die', 'end it all', 'harm myself']
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in crisis_keywords)
    
    def _safety_response(self, message: str) -> str:
        """Generate safety-focused response"""
        logger.warning("Generating safety response due to concern")
        
        return """I'm very concerned about what you've shared. Your safety is the most important thing right now.

**Please reach out for immediate support:**

📞 **988 Suicide & Crisis Lifeline:** Call or text 988
📱 **Crisis Text Line:** Text HOME to 741741
🚑 **Emergency:** Call 911 or go to nearest emergency room

These services are available 24/7, free, and confidential. Trained counselors are ready to help you right now.

If you're in immediate danger, please call 911 or have someone take you to the emergency room.

I want you to know that you don't have to face this alone, and there are people who care and want to help. Will you reach out to one of these resources right now?"""
    
    def _build_conversation_context(self, conversation_history: list) -> str:
        """Build context string from conversation history"""
        context = ""
        for turn in conversation_history[-6:]:  # Last 6 turns
            speaker = "Therapist" if turn['speaker'] == 'therapist' else "User"
            context += f"{speaker}: {turn['message']}\n\n"
        return context
    
    def _fallback_response(self, assessment_results: Dict) -> str:
        """Fallback response if generation fails"""
        logger.warning("Using fallback therapeutic response")
        
        key_symptoms = [s['symptom'] for s in assessment_results.get('key_symptoms', [])][:2]
        
        return f"""I want to acknowledge the challenges you're experiencing with {', '.join(key_symptoms) if key_symptoms else 'depression symptoms'}. Based on your assessment, these symptoms are at a level that warrants professional attention and evidence-based intervention.

**Understanding Your Experience:**
Moderate depression symptoms can significantly impact daily functioning, relationships, and quality of life. What you're experiencing is real, valid, and treatable. Many people with similar symptoms have found relief through structured interventions.

**Evidence-Based Approach - Behavioral Activation:**
One effective technique for depression is Behavioral Activation. The core idea is that when we're depressed, we often withdraw from activities, which actually makes depression worse. By intentionally engaging in activities—even when we don't feel like it—we can start to shift our mood.

**Your Action Plan:**
1. **Schedule one pleasant activity daily** - Start small (15-20 minutes)
2. **Break larger tasks into tiny steps** - Make them manageable
3. **Track your mood before and after activities** - Notice any changes
4. **Focus on "doing" rather than "feeling motivated"** - Action comes first

**Example:** If you used to enjoy reading, schedule 15 minutes of reading tomorrow at a specific time, even if you don't feel like it.

**Professional Support - IMPORTANT:**
Given your symptom level, I strongly recommend consulting with a mental health professional. While self-help strategies can be valuable, moderate depression often benefits from professional guidance, such as therapy with a licensed counselor or therapist. They can provide personalized treatment, including evidence-based therapy like CBT or consideration of medication if appropriate.

To find a therapist:
- Call SAMHSA National Helpline: 1-800-662-4357 (free, confidential)
- Search Psychology Today directory: psychologytoday.com/us/therapists
- Contact your insurance provider for in-network options

**Safety Check:**
I need to ask: Are you having any thoughts of harming yourself or suicide? If so, please reach out to the 988 Suicide & Crisis Lifeline immediately by calling or texting 988. Your safety is paramount.

**Moving Forward:**
Start with the behavioral activation exercise this week. Consider reaching out to a mental health professional within the next few days. You deserve support, and help is available.

Is there any specific aspect of what I've shared that you'd like to explore further?"""