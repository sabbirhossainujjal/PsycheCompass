from typing import Dict
from utils.logger import setup_logger

logger = setup_logger('therapeutic_router', 'logs/therapeutic_router.log')


class TherapeuticRouter:
    """
    Routes assessment results to appropriate therapeutic agent
    
    Routing Logic:
    - Crisis indicators detected → Crisis Support Agent
    - PHQ-8 Score 15+ → Crisis Support Agent
    - PHQ-8 Score 5-14 → Therapeutic Agent
    - PHQ-8 Score 0-4 → Emotional Support Agent
    """
    
    def __init__(self, config: Dict):
        logger.info("Initializing Therapeutic Router")
        
        self.config = config
        
        # Load routing thresholds
        routing_config = config.get('therapeutic_support', {}).get('routing', {})
        
        self.low_risk_max = routing_config.get('low_risk_max', 4)
        self.moderate_risk_min = routing_config.get('moderate_risk_min', 5)
        self.moderate_risk_max = routing_config.get('moderate_risk_max', 14)
        self.high_risk_min = routing_config.get('high_risk_min', 15)
        
        # Crisis keywords
        self.crisis_keywords = routing_config.get('crisis_keywords', [
            'want to die', 'kill myself', 'end it all', 'no point living',
            'suicide', 'suicidal', 'self harm', 'hurt myself', 
            'better off dead', 'end my life', 'kill'
        ])
        
        logger.info(f"Routing thresholds configured:")
        logger.info(f"  Low risk: 0-{self.low_risk_max}")
        logger.info(f"  Moderate risk: {self.moderate_risk_min}-{self.moderate_risk_max}")
        logger.info(f"  High risk: {self.high_risk_min}+")
        logger.info(f"  Crisis keywords: {len(self.crisis_keywords)} loaded")
        
    def route(self, therapeutic_input: Dict) -> str:
        """
        Route to appropriate therapeutic agent
        
        Args:
            therapeutic_input: Prepared assessment data including:
                - phq8_score: int (0-24)
                - crisis_indicators: list
                - risk_level: str
                
        Returns:
            Agent type: 'crisis', 'therapeutic', or 'emotional_support'
        """
        phq8_score = therapeutic_input.get('phq8_score', 0)
        crisis_indicators = therapeutic_input.get('crisis_indicators', [])


        logger.info(f"Routing decision requested - PHQ-8 Score: {phq8_score}")
        
        
        # Priority 1: Crisis indicators detected
        if crisis_indicators:
            logger.warning(f"Crisis indicators detected: {len(crisis_indicators)} indicators")
            
            logger.warning(f"Crisis keywords: {[ k for k in self.crisis_keywords if k in crisis_indicators]}")
            logger.info("→ Routing to: CRISIS SUPPORT AGENT")
            return 'crisis'
        
        # If crisis word found in user-message but score is low
        if therapeutic_input.get('user_message', None):
            if self.detect_crisis_in_text(therapeutic_input.get('user_message')):
                return 'crisis'
        
        
        # Priority 2: High score (severe symptoms)
        if phq8_score >= self.high_risk_min:
            logger.warning(f"High risk score detected: {phq8_score} (>= {self.high_risk_min})")
            logger.info("→ Routing to: CRISIS SUPPORT AGENT")
            return 'crisis'
        
        # Priority 3: Moderate score
        if self.moderate_risk_min <= phq8_score <= self.moderate_risk_max:
            logger.info(f"Moderate risk score: {phq8_score} ({self.moderate_risk_min}-{self.moderate_risk_max})")
            logger.info("→ Routing to: THERAPEUTIC AGENT")
            return 'therapeutic'
        
        # Priority 4: Low score
        logger.info(f"Low risk score: {phq8_score} (<= {self.low_risk_max})")
        logger.info("→ Routing to: EMOTIONAL SUPPORT AGENT")
        return 'emotional_support'
    
    def detect_crisis_in_text(self, text: str) -> bool:
        """
        Quick check for crisis keywords in text
        
        Args:
            text: Text to check
            
        Returns:
            True if crisis keywords detected
        """
        text_lower = text.lower()
        
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                logger.warning(f"Crisis keyword detected in text: '{keyword}'")
                return True
        
        return False
    
    def get_routing_explanation(self, therapeutic_input: Dict) -> str:
        """
        Get human-readable explanation of routing decision
        
        Args:
            therapeutic_input: Assessment input data
            
        Returns:
            Explanation string
        """
        agent_type = self.route(therapeutic_input)
        phq8_score = therapeutic_input.get('phq8_score', 0)
        crisis_indicators = therapeutic_input.get('crisis_indicators', [])
        
        explanations = {
            'crisis': f"""
**Routing Decision: Crisis Support**

Reason: {'Crisis indicators detected' if crisis_indicators else f'High risk score ({phq8_score} >= {self.high_risk_min})'}

Action: Immediate crisis intervention protocol activated
- Safety assessment
- Crisis hotline resources (988)
- Emergency services guidance
- Urgent professional referral
""",
            'therapeutic': f"""
**Routing Decision: Therapeutic Intervention**

Reason: Moderate depression symptoms (Score: {phq8_score}/{self.moderate_risk_min}-{self.moderate_risk_max})

Action: Evidence-based therapeutic support
- CBT/Behavioral Activation techniques
- Problem-solving therapy
- Mindfulness interventions
- Strong professional referral
""",
            'emotional_support': f"""
**Routing Decision: Emotional Support**

Reason: Minimal to mild symptoms (Score: {phq8_score} <= {self.low_risk_max})

Action: Supportive counseling
- Validation and normalization
- Psychoeducation
- Self-care strategies
- Monitoring guidance
"""
        }
        
        return explanations.get(agent_type, "Unknown routing decision")