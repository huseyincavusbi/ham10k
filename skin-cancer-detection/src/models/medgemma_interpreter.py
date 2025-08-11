import requests
import subprocess
import json
from typing import Dict, Any, Optional
import time
from src.utils.config import settings, CLASS_DESCRIPTIONS, CLASS_URGENCY
from src.utils.logging import get_logger

logger = get_logger(__name__)

class LocalMedicalInterpreter:
    def __init__(self):
        self.lm_studio_endpoint = settings.LM_STUDIO_ENDPOINT
        self.gemini_cli_path = settings.GEMINI_CLI_PATH
        self.class_descriptions = CLASS_DESCRIPTIONS
        self.class_urgency = CLASS_URGENCY
        
    def create_medical_prompt(self, prediction_class: str, confidence: float,
                              image_features: Optional[Dict] = None) -> str:
        """Create structured prompt for MedGemma-4B.

        Returns a rigorously formatted markdown instruction for the language model.
        """
        urgency = self.class_urgency.get(prediction_class, "unknown")
        description = self.class_descriptions.get(prediction_class, "Unknown condition")
        return f"""
You are MedGemma, an evidence‑informed medical AI assistant (dermatology focus).

CONTEXT:
- Predicted Condition: {prediction_class}
- Model Confidence: {confidence:.1%}
- Urgency Level: {urgency}
- Brief Description: {description}
- Image Analysis Context: {image_features or "Basic classification performed"}

TASK: Produce a concise, patient‑friendly advisory (≤400 words) with clearly delimited sections using Markdown. Do NOT overclaim. Avoid diagnostic certainty. Always advise professional evaluation.

STRUCTURE (use these exact section titles):
1. Condition Overview – Plain language explanation (2–4 sentences).
2. What the Confidence Means – Explain what {confidence:.1%} confidence indicates, and that AI probabilities are not guarantees.
3. Key Visual / Clinical Features – Bullet list (3–6 short bullets) of typical characteristics patients might observe.
4. Recommended Next Actions – Tailor to urgency '{urgency}'. Provide timeframe (e.g., same day / 1–2 weeks / routine visit) and immediate protective steps.
5. When to Seek Immediate Care – Bullet list of red flags (bleeding, rapid change, pain, ulceration, ABCDE changes, etc.).
6. Self‑Monitoring Checklist – 4–6 short actionable bullets (photograph, note changes, avoid sun, do not self‑treat aggressively, etc.).
7. Lifestyle & Prevention Tips – 3–5 concise bullets (sun protection, skin exams, risk awareness).
8. Professional Follow‑Up Rationale – Why an in‑person dermatology assessment is important (biopsy possibility, differential diagnosis).
9. Important Disclaimers – Clear statement: AI screening, not a diagnosis; may be wrong; must not delay professional care.

STYLE GUIDELINES:
- 8th–10th grade reading level.
- Empathetic, neutral, non-alarming unless urgency is 'urgent' or 'high'.
- No definitive diagnostic language; use phrases like "suggests", "may be consistent with".
- Avoid repeating the same sentence; be concise.
- No speculative treatment regimens—only general next steps.

OUTPUT REQUIREMENTS:
- Markdown only; no surrounding backticks.
- Use bold for key terms (e.g., **melanoma**, **confidence**, **urgent evaluation**).
- Ensure each section header appears exactly once and in the specified order.
"""
    
    def interpret_via_lm_studio(self, prompt: str) -> str:
        """Query MedGemma-4B via LM Studio API"""
        payload = {
            "model": settings.LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            logger.info("Requesting interpretation from LM Studio...")
            response = requests.post(
                self.lm_studio_endpoint, 
                json=payload, 
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            interpretation = result["choices"][0]["message"]["content"]
            logger.info("Successfully received interpretation from LM Studio")
            return interpretation
            
        except requests.exceptions.ConnectionError:
            logger.warning("LM Studio not available, using fallback interpretation")
            return self.fallback_interpretation()
        except requests.exceptions.Timeout:
            logger.warning("LM Studio request timed out, using fallback interpretation")
            return self.fallback_interpretation()
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
            return self.fallback_interpretation()
    
    def enhance_with_gemini_cli(self, interpretation: str, 
                               additional_context: Dict) -> str:
        """Use Gemini CLI for additional medical context and validation"""
        gemini_prompt = f"""
        Review this medical AI interpretation for accuracy and completeness:
        
        {interpretation}
        
        Additional context: {json.dumps(additional_context, indent=2)}
        
        Please enhance this interpretation with:
        - Medical accuracy verification
        - Additional patient education points
        - Clearer next-step recommendations based on urgency level
        - Any important warnings or precautions
        
        Maintain the patient-friendly tone and keep under 400 words.
        """
        
        try:
            logger.info("Enhancing interpretation with Gemini CLI...")
            result = subprocess.run([
                self.gemini_cli_path, 
                "--prompt", gemini_prompt,
                "--model", settings.GEMINI_MODEL,
                "--temperature", "0.1"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                logger.info("Successfully enhanced interpretation with Gemini CLI")
                return result.stdout.strip()
            else:
                logger.warning(f"Gemini CLI returned error: {result.stderr}")
                return interpretation
                
        except subprocess.TimeoutExpired:
            logger.warning("Gemini CLI request timed out")
            return interpretation
        except FileNotFoundError:
            logger.warning("Gemini CLI not found, returning original interpretation")
            return interpretation
        except Exception as e:
            logger.error(f"Gemini CLI enhancement failed: {e}")
            return interpretation
    
    def fallback_interpretation(self, prediction_class: Optional[str] = None, 
                               confidence: Optional[float] = None) -> str:
        """Provide a fallback interpretation when AI services are unavailable"""
        if prediction_class and confidence:
            urgency = self.class_urgency.get(prediction_class, "unknown")
            description = self.class_descriptions.get(prediction_class, "Unknown condition")
            
            return f"""
**AI Analysis Results:**

**Condition Identified:** {prediction_class}
**Confidence Level:** {confidence:.1%}

**What this means:**
{description}

**Recommended Action (Urgency: {urgency}):**
{self._get_urgency_recommendation(urgency)}

**Important Disclaimer:**
This analysis is provided by an AI system for educational purposes only. It is not a medical diagnosis and should not replace professional medical advice. Always consult with a qualified healthcare provider for proper evaluation and treatment of any skin condition.

**Next Steps:**
1. Save or take note of this analysis
2. Schedule an appointment with a dermatologist or your primary care physician
3. Mention this AI screening result during your consultation
4. Follow your healthcare provider's recommendations

Remember: Early detection and professional medical evaluation are key to proper skin health management.
"""
        else:
            return """
**AI Service Temporarily Unavailable**

Our advanced medical interpretation service is currently unavailable. However, we recommend:

1. **Seek Professional Evaluation:** Consult with a dermatologist or healthcare provider for any concerning skin lesions
2. **Document the Area:** Take clear photos and note any changes in size, color, or texture
3. **Monitor Changes:** Keep track of any evolution in the lesion over time
4. **Don't Delay:** If you notice rapid changes or the lesion looks unusual, seek medical attention promptly

**When to Seek Immediate Care:**
- Rapid growth or changes in appearance
- Bleeding or ulceration
- Irregular borders or multiple colors
- Any concerns about the lesion

This AI screening tool is for educational purposes only and cannot replace professional medical diagnosis.
"""
    
    def _get_urgency_recommendation(self, urgency: str) -> str:
        """Get appropriate recommendation based on urgency level"""
        recommendations = {
            "urgent": "Seek immediate medical attention. Contact a dermatologist or visit an urgent care facility as soon as possible.",
            "high": "Schedule an appointment with a dermatologist within 1-2 weeks. Do not delay seeking professional evaluation.",
            "moderate": "Schedule an appointment with a dermatologist within 1-2 months. Monitor for any changes in the meantime.",
            "low": "Consider scheduling a routine dermatology check-up within 3-6 months or during your next regular skin screening."
        }
        return recommendations.get(urgency, "Consult with a healthcare professional for proper evaluation.")
    
    def generate_final_report(self, prediction_result: Dict) -> Dict:
        """
        Generate a complete medical interpretation report
        
        Args:
            prediction_result: Dictionary containing prediction results from the vision model
            
        Returns:
            Dictionary containing the complete medical report
        """
        start_time = time.time()
        
        try:
            # Extract prediction information
            ensemble_pred = prediction_result.get("ensemble_prediction", prediction_result.get("final_prediction", {}))
            predicted_class = ensemble_pred.get("class", "unknown")
            confidence = ensemble_pred.get("confidence", 0.0)
            urgency = ensemble_pred.get("urgency", "unknown")
            
            # Prepare additional context
            additional_context = {
                "confidence_level": confidence,
                "urgency": urgency,
                "model_info": {
                    "ensemble_used": "ensemble_prediction" in prediction_result,
                    "tta_enabled": prediction_result.get("tta_enabled", False),
                    "inference_time": prediction_result.get("inference_time", 0)
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "all_probabilities": prediction_result.get("all_class_probabilities", {})
            }
            
            # Step 1: Create medical prompt
            base_prompt = self.create_medical_prompt(
                predicted_class, 
                confidence,
                additional_context.get("model_info", {})
            )
            
            # Step 2: Get MedGemma interpretation via LM Studio
            medgemma_response = self.interpret_via_lm_studio(base_prompt)
            
            # Step 3: Enhance with Gemini CLI (if available)
            final_interpretation = self.enhance_with_gemini_cli(
                medgemma_response,
                additional_context
            )
            
            processing_time = time.time() - start_time
            
            report = {
                "medical_interpretation": final_interpretation,
                "prediction_summary": {
                    "predicted_condition": predicted_class,
                    "confidence_score": confidence,
                    "urgency_level": urgency,
                    "description": self.class_descriptions.get(predicted_class, "Unknown condition")
                },
                "technical_details": {
                    "model_ensemble": "ensemble_prediction" in prediction_result,
                    "tta_enabled": prediction_result.get("tta_enabled", False),
                    "inference_time": prediction_result.get("inference_time", 0),
                    "interpretation_time": processing_time,
                    "all_class_probabilities": prediction_result.get("all_class_probabilities", {})
                },
                "disclaimers": {
                    "primary": "This AI analysis is for educational and screening purposes only.",
                    "secondary": "Always consult qualified healthcare professionals for medical decisions.",
                    "limitation": "AI systems can make errors and should not be relied upon for final diagnosis."
                },
                "next_steps": self._get_urgency_recommendation(urgency),
                "metadata": {
                    "report_generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_version": "HAM10000-ensemble-v1.0",
                    "language_model": settings.LM_STUDIO_MODEL if self._check_lm_studio_availability() else "fallback",
                    "enhancement": "gemini-cli" if self._check_gemini_cli_availability() else "none"
                }
            }
            
            logger.info(f"Medical report generated successfully in {processing_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate medical report: {e}")
            # Return a safe fallback report
            return {
                "medical_interpretation": self.fallback_interpretation(
                    prediction_result.get("ensemble_prediction", {}).get("class"),
                    prediction_result.get("ensemble_prediction", {}).get("confidence")
                ),
                "error": "Report generation encountered an error",
                "metadata": {
                    "report_generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "fallback_mode"
                }
            }
    
    def _check_lm_studio_availability(self) -> bool:
        """Check if LM Studio is available"""
        try:
            response = requests.get(self.lm_studio_endpoint.replace("/v1/chat/completions", "/v1/models"), timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_gemini_cli_availability(self) -> bool:
        """Check if Gemini CLI is available"""
        try:
            result = subprocess.run([self.gemini_cli_path, "--version"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def health_check(self) -> Dict:
        """Perform health check on all interpretation services"""
        return {
            "lm_studio_available": self._check_lm_studio_availability(),
            "gemini_cli_available": self._check_gemini_cli_availability(),
            "fallback_ready": True,
            "endpoint": self.lm_studio_endpoint,
            "gemini_path": self.gemini_cli_path
        }
