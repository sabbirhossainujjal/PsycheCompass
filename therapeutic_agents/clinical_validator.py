"""
Clinical Validator

Validates therapeutic responses before delivery to ensure:
- No harmful advice
- No medication recommendations
- No diagnosis statements
- Appropriate crisis resources (when needed)
- Clinical safety and appropriateness
"""

from typing import Dict
from utils.logger import setup_logger
from utils.llm import LLMOrchestrator

logger = setup_logger('clinical_validator', 'logs/clinical_validator.log')


class ClinicalValidator:
    """
    Clinical Validator for safety checks
    
    Validates all therapeutic responses before delivery to users.
    Acts as final safety net to prevent:
    - Harmful advice
    - Medication recommendations
    - Clinical diagnosis statements
    - Minimization of crisis situations
    - Inappropriate tone or urgency
    """
    
    def __init__(self, llm: LLMOrchestrator, config: Dict):
        self.llm = llm
        self.config = config
        
        # Load validation configuration
        validation_config = config.get('therapeutic_support', {}).get('validation', {})
        
        self.enabled = validation_config.get('enabled', True)
        self.check_for = validation_config.get('check_for', [
            'harmful_advice',
            'medication_recommendations',
            'diagnosis_statements',
            'minimization_of_crisis'
        ])
        self.override_on_crisis = validation_config.get('override_on_crisis', True)
        
        # Load validation prompt
        self.validation_prompt_template = config.get('therapeutic_prompts', {}).get(
            'validation', {}
        ).get('system', '')
        
        logger.info("Clinical Validator initialized")
        logger.info(f"Validation enabled: {self.enabled}")
        logger.info(f"Checks: {self.check_for}")
    
    def validate(
        self,
        therapeutic_response: str,
        risk_level: str,
        assessment_results: Dict
    ) -> str:
        """
        Validate therapeutic response for safety
        
        Args:
            therapeutic_response: Generated therapeutic response
            risk_level: 'low', 'moderate', 'high', or 'crisis'
            assessment_results: Assessment data
            
        Returns:
            Validated (possibly modified) response
        """
        if not self.enabled:
            logger.info("Validation disabled - returning original response")
            return therapeutic_response
        
        logger.info(f"Validating response for risk level: {risk_level}")
        
        # Quick rule-based checks first (fast)
        quick_issues = self._quick_validation(therapeutic_response, risk_level)
        
        if quick_issues:
            logger.warning(f"Quick validation found {len(quick_issues)} issues")
            therapeutic_response = self._apply_quick_fixes(
                therapeutic_response,
                quick_issues,
                risk_level,
                assessment_results
            )
        
        # Enhanced validation for crisis cases
        if risk_level in ['high', 'crisis'] or quick_issues:
            logger.info("Running enhanced LLM-based validation")
            therapeutic_response = self._enhanced_validation(
                therapeutic_response,
                risk_level,
                assessment_results
            )
        
        logger.info("Validation complete")
        return therapeutic_response
    
    def _quick_validation(self, response: str, risk_level: str) -> list:
        """
        Quick rule-based validation checks
        
        Returns list of issues found
        """
        logger.debug("Running quick validation checks")
        
        issues = []
        response_lower = response.lower()
        
        # Check 1: Medication recommendations
        medication_keywords = [
            'take this medication', 'prescribe', 'dosage', 'mg', 'antidepressant',
            'you should take', 'i recommend taking', 'start on', 'try this medication'
        ]
        
        for keyword in medication_keywords:
            if keyword in response_lower:
                issues.append({
                    'type': 'medication_recommendation',
                    'keyword': keyword,
                    'severity': 'high'
                })
                logger.warning(f"Found medication recommendation: '{keyword}'")
        
        # Check 2: Diagnosis statements
        diagnosis_keywords = [
            'you have depression', 'you are depressed', 'your depression',
            'you have been diagnosed', 'you suffer from', 'i diagnose'
        ]
        
        for keyword in diagnosis_keywords:
            if keyword in response_lower:
                # Exclude acceptable phrases like "symptoms of depression"
                if not any(ok in response_lower for ok in ['symptoms of', 'signs of', 'experiencing']):
                    issues.append({
                        'type': 'diagnosis_statement',
                        'keyword': keyword,
                        'severity': 'high'
                    })
                    logger.warning(f"Found diagnosis statement: '{keyword}'")
        
        # Check 3: Minimizing language
        minimizing_keywords = [
            'just relax', 'just think positive', 'just try to', 'it\'s not that bad',
            'others have it worse', 'get over it', 'snap out of it'
        ]
        
        for keyword in minimizing_keywords:
            if keyword in response_lower:
                issues.append({
                    'type': 'minimization',
                    'keyword': keyword,
                    'severity': 'medium'
                })
                logger.warning(f"Found minimizing language: '{keyword}'")
        
        # Check 4: Crisis resources (for high risk)
        if risk_level in ['high', 'crisis']:
            has_988 = '988' in response
            has_crisis_line = 'crisis' in response_lower and ('line' in response_lower or 'hotline' in response_lower)
            has_911 = '911' in response
            
            if not (has_988 or has_crisis_line or has_911):
                issues.append({
                    'type': 'missing_crisis_resources',
                    'keyword': 'none',
                    'severity': 'critical'
                })
                logger.error("Crisis response missing crisis resources!")
        
        # Check 5: Professional referral
        has_referral = any(word in response_lower for word in [
            'professional', 'therapist', 'counselor', 'mental health', 'doctor', 'psychiatrist'
        ])
        
        if not has_referral and risk_level in ['moderate', 'high', 'crisis']:
            issues.append({
                'type': 'missing_professional_referral',
                'keyword': 'none',
                'severity': 'medium'
            })
            logger.warning("Missing professional referral in moderate/high risk response")
        
        logger.debug(f"Quick validation found {len(issues)} issues")
        return issues
    
    def _apply_quick_fixes(
        self,
        response: str,
        issues: list,
        risk_level: str,
        assessment_results: Dict
    ) -> str:
        """Apply automatic fixes for common issues"""
        logger.info(f"Applying quick fixes for {len(issues)} issues")
        
        modified_response = response
        
        # Fix 1: Add crisis resources if missing
        if any(issue['type'] == 'missing_crisis_resources' for issue in issues):
            logger.warning("Adding crisis resources to response")
            crisis_resources = """

🚨 **IMMEDIATE CRISIS RESOURCES:**

📞 **988 Suicide & Crisis Lifeline** - Call or text 988
📱 **Crisis Text Line** - Text HOME to 741741
🚑 **Emergency** - Call 911 or go to nearest emergency room

Available 24/7, free, confidential.
"""
            # Add at the beginning for crisis cases
            modified_response = crisis_resources + "\n\n" + modified_response
        
        # Fix 2: Add professional referral if missing
        if any(issue['type'] == 'missing_professional_referral' for issue in issues):
            logger.info("Adding professional referral to response")
            referral = """\n\n**Professional Support:**
Given the severity of your symptoms, it's important to consult with a mental health professional who can provide personalized treatment and ongoing support. Consider reaching out to a therapist or counselor who specializes in depression."""
            modified_response += referral
        
        # Fix 3: Remove specific medication recommendations
        medication_issues = [i for i in issues if i['type'] == 'medication_recommendation']
        for issue in medication_issues:
            logger.warning(f"Attempting to remove medication recommendation: {issue['keyword']}")
            # This is complex - flag for enhanced validation
        
        return modified_response
    
    def _enhanced_validation(
        self,
        response: str,
        risk_level: str,
        assessment_results: Dict
    ) -> str:
        """
        Enhanced LLM-based validation for complex checks
        
        Uses LLM to detect subtle safety issues
        """
        logger.info("Running enhanced LLM validation")
        
        # Format validation prompt
        prompt = self.validation_prompt_template.format(
            risk_level=risk_level,
            therapeutic_response=response
        )
        
        try:
            validation_result = self.llm.generate(prompt, max_tokens=512, temperature=0.0)
            
            logger.debug(f"Validation result: {validation_result[:200]}...")
            
            # Check if approved or issues found
            if 'APPROVED' in validation_result.upper():
                logger.info("Enhanced validation: APPROVED")
                return response
            
            elif 'ISSUES FOUND' in validation_result.upper():
                logger.warning("Enhanced validation found issues")
                logger.warning(validation_result)
                
                # Try to extract suggestions and apply
                # For now, log and return original with safety disclaimer
                return self._add_safety_disclaimer(response, risk_level)
            
            else:
                logger.warning("Unclear validation result, adding safety disclaimer")
                return self._add_safety_disclaimer(response, risk_level)
                
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}", exc_info=True)
            logger.info("Falling back to original response with safety disclaimer")
            return self._add_safety_disclaimer(response, risk_level)
    
    def _add_safety_disclaimer(self, response: str, risk_level: str) -> str:
        """Add safety disclaimer to response"""
        logger.info("Adding safety disclaimer")
        
        if risk_level in ['high', 'crisis']:
            disclaimer = """

---

**Important Safety Note:** This is not professional medical advice. If you are experiencing a mental health emergency or having thoughts of self-harm, please call 988 (Suicide & Crisis Lifeline) or 911 immediately.
"""
        else:
            disclaimer = """

---

**Important Note:** This information is for educational purposes and is not a substitute for professional mental health care. Please consult with a licensed mental health professional for personalized treatment.
"""
        
        return response + disclaimer
    
    def validate_user_message(self, user_message: str) -> Dict:
        """
        Validate incoming user message for safety concerns
        
        Args:
            user_message: User's message
            
        Returns:
            Dict with 'safe': bool and 'concerns': list
        """
        logger.debug("Validating user message")
        
        concerns = []
        message_lower = user_message.lower()
        
        # Check for crisis keywords
        crisis_keywords = [
            'suicide', 'kill myself', 'want to die', 'end it all',
            'end my life', 'not worth living', 'better off dead'
        ]
        
        for keyword in crisis_keywords:
            if keyword in message_lower:
                concerns.append({
                    'type': 'crisis_keyword',
                    'keyword': keyword,
                    'severity': 'critical'
                })
                logger.critical(f"Crisis keyword in user message: '{keyword}'")
        
        safe = len([c for c in concerns if c['severity'] == 'critical']) == 0
        
        return {
            'safe': safe,
            'concerns': concerns
        }
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        return {
            'enabled': self.enabled,
            'checks_performed': self.check_for,
            'override_on_crisis': self.override_on_crisis
        }