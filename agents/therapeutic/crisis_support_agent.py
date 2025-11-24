"""
Crisis Support Agent

Handles high-risk cases (PHQ-8 Score: 15+) and detected crisis situations
Provides immediate crisis intervention with SAFETY as absolute priority
"""

from typing import Dict, Optional
from utils.logger import setup_logger
from utils.llm import LLMOrchestrator

logger = setup_logger('crisis_support_agent', 'logs/crisis_support_agent.log')


class CrisisSupportAgent:
    """
    Crisis Support Agent for high-risk individuals and crisis situations

    PROTOCOL:
    1. Express immediate concern for safety
    2. Provide PROMINENT crisis hotline (988)
    3. Conduct immediate safety assessment
    4. De-escalation techniques
    5. Safety planning
    6. Urgent professional help guidance
    7. Directive, caring closing

    SAFETY IS ABSOLUTE PRIORITY
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
        ).get('crisis', {})

        self.crisis_keywords = [
            'want to die', 'kill myself', 'end it all', 'no point living',
            'suicide', 'suicidal', 'self harm', 'hurt myself',
            'better off dead', 'end my life', 'kill'
        ]
        self.techniques = agent_config.get('techniques', [
            'safety_assessment', 'de_escalation',
            'crisis_intervention', 'resource_connection'
        ])
        self.immediate_escalation = agent_config.get(
            'immediate_escalation', True)
        self.hotlines = agent_config.get('hotlines', {})

        # Load prompts
        self.system_prompt_template = config.get('therapeutic_prompts', {}).get(
            'crisis', {}
        ).get('system', '')

        logger.critical("Crisis Support Agent initialized")
        logger.critical(f"Immediate escalation: {self.immediate_escalation}")
        logger.critical(f"Hotlines configured: {list(self.hotlines.keys())}")

    def generate_response(self, assessment_results: Dict) -> str:
        """
        Generate crisis intervention response

        Args:
            assessment_results: Prepared therapeutic input with crisis indicators

        Returns:
            Crisis response text with PROMINENT safety resources
        """
        logger.critical("="*70)
        logger.critical("CRISIS RESPONSE ACTIVATED")
        logger.critical(f"PHQ-8 Score: {assessment_results['phq8_score']}")
        logger.critical(
            f"Crisis Indicators: {len(assessment_results.get('crisis_indicators', []))}")
        logger.critical("="*70)

        # Get crisis hotline for user location
        crisis_hotline = self._get_crisis_hotline(
            assessment_results.get('user_info', {}))

        # Format system prompt
        system_prompt = self._format_system_prompt(
            assessment_results, crisis_hotline)

        # Create user prompt
        user_prompt = self._create_crisis_prompt(assessment_results)

        # Generate response
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        logger.debug(f"Crisis prompt length: {len(full_prompt)} characters")

        try:
            response = self.llm.generate(
                full_prompt, max_tokens=2048, temperature=0.0)
            logger.critical("Crisis response generated")

            # ENSURE crisis hotline is in response
            if '988' not in response and 'crisis' not in response.lower():
                logger.warning(
                    "Crisis hotline not detected in response - adding safety net")
                response = self._add_safety_net(response, crisis_hotline)

            logger.critical("Crisis response ready for delivery")
            return response

        except Exception as e:
            logger.critical(
                f"ERROR generating crisis response: {e}", exc_info=True)
            logger.critical("Using emergency fallback response")
            return self._emergency_fallback(crisis_hotline, assessment_results)

    def _format_system_prompt(self, assessment_results: Dict, crisis_hotline: str) -> str:
        """Format system prompt with crisis data"""
        crisis_indicators = assessment_results.get('crisis_indicators', [])
        crisis_keywords = [k for k in self.crisis_keywords if k in crisis_indicators]

        return self.system_prompt_template.format(
            assessment_summary=assessment_results['assessment_summary'],
            phq8_score=assessment_results['phq8_score'],
            crisis_indicators=', '.join(
                crisis_keywords) if crisis_keywords else 'High severity score',
            crisis_hotline=crisis_hotline
        )

    def _create_crisis_prompt(self, assessment_results: Dict) -> str:
        """Create crisis intervention prompt"""
        crisis_indicators = assessment_results.get('crisis_indicators', [])

        prompt = """Generate a CRISIS INTERVENTION response following the EXACT protocol below.

**ABSOLUTE REQUIREMENTS - FOLLOW EXACTLY:**

1. **URGENT OPENING (2-3 sentences):**
   - Express IMMEDIATE concern for their safety
   - Acknowledge severity of their pain/situation
   - Provide hope that help is available RIGHT NOW

2. **CRISIS HOTLINE - MUST BE IN FIRST PARAGRAPH:**
   Display prominently:
   📞 **Call or Text 988** - Suicide & Crisis Lifeline
   📱 **Text HOME to 741741** - Crisis Text Line
   🚑 **Emergency: Call 911** or go to nearest ER
   
   Emphasize: "24/7, free, confidential, trained crisis counselors available NOW"

3. **IMMEDIATE SAFETY ASSESSMENT (Ask directly):**
   - "Are you thinking about harming yourself right now?"
   - "Do you have a plan?"
   - "Do you have access to means?"
   State: "These questions are critical - please be honest"

4. **DE-ESCALATION (Brief, 2-3 sentences):**
   - Empathetically acknowledge their pain is real
   - State feelings are valid but suicide is permanent
   - Remind them they are not alone
   - Express that help IS available immediately

5. **SAFETY PLANNING (If engaged):**
   - Warning signs to watch for
   - Coping strategies for crisis moments
   - People they can call RIGHT NOW
   - Remove access to means (weapons, pills, etc.)

6. **URGENT PROFESSIONAL HELP (Directive):**
   - STRONGLY encourage calling 988 or 911 NOW
   - If in immediate danger → ER visit ESSENTIAL
   - Crisis counselors are specifically trained for this
   - This is EXACTLY what these services are for

7. **CLOSING - DIRECTIVE & CARING:**
   - Reiterate: calling crisis line or ER is ESSENTIAL
   - Remind: they are NOT alone
   - Final URGENT encouragement to reach out NOW
   - Express care and concern

**CRITICAL RULES - NEVER VIOLATE:**
- ❌ NO minimization ("just relax", "think positive")
- ❌ NO complex therapy techniques
- ❌ NO lengthy explanations that delay help
- ✅ CRISIS HOTLINE in first 3-4 sentences
- ✅ DIRECTIVE tone (this is emergency)
- ✅ 100% focus on immediate safety
- ✅ Urgent but caring tone

**Tone:** URGENT, caring, directive, hopeful without minimization
**Length:** 5-6 paragraphs (keep focused on IMMEDIATE safety)
"""

        if crisis_indicators:
            prompt += f"\n\n**Crisis Indicators Detected:**\n"
            for indicator in crisis_indicators[:3]:
                prompt += f"- Keyword: '{indicator}'\n"

        return prompt

    def _get_crisis_hotline(self, user_info: Dict) -> str:
        """Get appropriate crisis hotline"""
        if self.knowledge_agent:
            try:
                return self.knowledge_agent.get_crisis_hotline(user_info)
            except Exception as e:
                logger.error(f"Error getting crisis hotline: {e}")

        # Default US hotline
        return """**988 Suicide & Crisis Lifeline**
📞 Call or Text: 988
📱 Crisis Text Line: Text HOME to 741741
🚑 Emergency: 911
ℹ️ 24/7, free, confidential support"""

    def _add_safety_net(self, response: str, crisis_hotline: str) -> str:
        """Add crisis hotline if not present in response"""
        logger.warning("Adding safety net to response")

        safety_net = f"""

🚨 **IMMEDIATE CRISIS RESOURCES:**

{crisis_hotline}

**If you are in immediate danger, please call 911 or go to your nearest emergency room.**

"""
        # Add at the beginning
        return safety_net + "\n\n" + response

    def handle_immediate_crisis(self, user_input: str) -> str:
        """
        Handle immediate crisis escalation during conversation

        Args:
            user_input: User message containing crisis indicators

        Returns:
            Emergency crisis response
        """
        logger.critical("="*70)
        logger.critical("IMMEDIATE CRISIS DETECTED IN CONVERSATION")
        logger.critical(f"User input: {user_input[:100]}...")
        logger.critical("="*70)

        # Get default crisis hotline
        crisis_hotline = """**988 Suicide & Crisis Lifeline**
📞 Call or Text: 988
📱 Text HOME to 741741
🚑 Emergency: 911"""

        response = f"""🚨 **IMMEDIATE CONCERN FOR YOUR SAFETY**

I'm very concerned about what you've just shared. Your safety is the absolute priority right now.

**PLEASE REACH OUT FOR IMMEDIATE HELP:**

{crisis_hotline}

These services are available RIGHT NOW - 24/7, free, and confidential. Trained crisis counselors are ready to help you this moment.

**If you are in immediate danger:**
- Call 911
- Go to your nearest emergency room
- Call the crisis line (988) immediately

I know you're in pain, and I want you to know that you don't have to face this alone. There are people who care and are trained specifically to help in situations like this. Please reach out to one of these resources right now.

**This is urgent. Please don't wait. Call or text 988 now.**

Will you reach out to one of these crisis resources right now? Your safety matters.
"""

        logger.critical("Emergency crisis response generated")
        return response

    def continue_conversation(
        self,
        user_message: str,
        conversation_history: list,
        assessment_results: Dict
    ) -> str:
        """
        Continue crisis conversation with ongoing safety focus

        For crisis cases, conversation should focus on:
        - Safety planning
        - Resource connection
        - Professional help facilitation
        - NOT therapeutic techniques
        """
        logger.critical(
            f"Continuing crisis conversation (turn {len(conversation_history) + 1})")

        # Check if user is engaging with safety planning
        message_lower = user_message.lower()

        if any(word in message_lower for word in ['called', 'talking to', 'reached out', 'contacted']):
            logger.info("User reports reaching out for help")
            return self._support_help_seeking(user_message)

        if any(word in message_lower for word in ["won't", "can't", "don't want", "not ready"]):
            logger.warning("User resistant to seeking help")
            return self._address_resistance(user_message)

        # Check for escalation
        if any(word in message_lower for word in ['suicide', 'kill myself', 'want to die']):
            logger.critical("Crisis escalation in conversation")
            return self.handle_immediate_crisis(user_message)

        # Default: continue safety focus
        return self._continue_safety_focus(user_message, conversation_history)

    def _support_help_seeking(self, user_message: str) -> str:
        """Support user who is reaching out for help"""
        return """I'm so glad to hear you're reaching out for help. That takes courage, and it's exactly the right step to take right now.

As you talk with the crisis counselor or mental health professional:
- Be honest about how you're feeling
- Don't minimize your struggles  
- Let them know about any thoughts of self-harm
- Ask about immediate support options

You're doing the right thing by seeking help. Please continue to stay connected with professional support, and don't hesitate to reach out again if you need immediate help.

Is there anything you need right now while you're getting connected with support?"""

    def _address_resistance(self, user_message: str) -> str:
        """Address resistance to seeking help"""
        return """I hear that you're hesitant about reaching out. That's understandable - it can feel difficult or scary to ask for help. But I want to emphasize that your safety is what matters most right now.

The crisis counselors at 988 are trained specifically for situations like this. They:
- Will NOT judge you
- Understand exactly what you're going through
- Can help you work through these feelings
- Are available right now, 24/7

You don't have to commit to anything - just talking to someone can help. One conversation could make a difference.

**Please consider:**
- Texting is an option (Text HOME to 741741) if calling feels too hard
- The conversation is completely confidential
- You can hang up at any time
- They're trained to help people who feel exactly like you do right now

Your life has value, and there is help available. Will you consider reaching out, even if it's just sending a text? What's holding you back from making that call or text right now?"""

    def _continue_safety_focus(self, user_message: str, conversation_history: list) -> str:
        """Continue conversation with safety focus"""
        return """Thank you for continuing to talk with me. While I'm here to support you, I want to emphasize again that professional crisis support is essential right now.

**988 Suicide & Crisis Lifeline: Call or text 988**
**Crisis Text Line: Text HOME to 741741**

These services are specifically designed to help in situations like yours, and the counselors are trained to provide the support you need right now.

**Can we work on a quick safety plan together?**

1. **Warning signs:** What thoughts or feelings might indicate you're in crisis?
2. **Coping strategies:** What has helped you get through difficult moments before?
3. **People to contact:** Who can you call if you need immediate support?
4. **Remove means:** Can you remove or limit access to anything you might use to harm yourself?

Most importantly: **Will you commit to reaching out to 988 if you feel you're in immediate danger?**

What would help you feel safer right now?"""

    def _emergency_fallback(self, crisis_hotline: str, assessment_results: Dict) -> str:
        """Emergency fallback response if generation fails"""
        logger.critical("USING EMERGENCY FALLBACK RESPONSE")

        return f"""🚨 **YOUR SAFETY IS MY IMMEDIATE CONCERN**

Based on your assessment, I am very concerned about your safety and well-being right now. You are experiencing severe symptoms that require immediate professional support.

**PLEASE REACH OUT FOR HELP RIGHT NOW:**

{crisis_hotline}

**These resources are available 24/7, FREE, and CONFIDENTIAL.**

**If you are in immediate danger:**
🚑 Call 911
🏥 Go to your nearest emergency room
📞 Call 988 Suicide & Crisis Lifeline

**I need to ask you directly:**
- Are you thinking about harming yourself or suicide right now?
- Do you have a plan?
- Do you have access to means to harm yourself?

**Please know:**
- You are not alone in this
- Your pain is real, but suicide is a permanent solution to temporary problems
- Help IS available right now
- Crisis counselors are specifically trained to help people feeling exactly like you do

**This is urgent. Please call or text 988 right now, or call 911 if you're in immediate danger.**

Your life matters. Please reach out for help immediately.
"""

    def _fallback_response(self, assessment_results: Dict) -> str:
        """Standard fallback (same as emergency fallback for crisis)"""
        return self._emergency_fallback(
            self._get_crisis_hotline(assessment_results.get('user_info', {})),
            assessment_results
        )
