import re
from typing import Dict, Any, List

class StreamIntentController:
    """
    5-Gate Deterministic Routing Pipeline (Section 1 of System Architecture Report)
    - G1: Wait Gate (Low turn entropy / incomplete sentence)
    - G2: Provisional Retrieve Gate (Speculative probe determination)
    - G3: Commit Gate (Multi-intent decomposition & execution)
    - G4: Suppress Gate (Formatting/context update suppression)
    - G5: Fallback Gate (Low confidence / zero match handling)
    """
    def __init__(self):
        # Trigger words indicating formatting, summarizing, or rephrasing (G4 Suppression)
        self.suppress_patterns = [
            r"\bsummarize\b", r"\bbullet points\b", r"\bformat as\b", r"\brephrase\b", 
            r"\bmake it shorter\b", r"\btranslate\b", r"\bsimplify\b", r"\bthanks\b",
            r"\bthank you\b", r"\bbye\b", r"\bhello\b", r"\bhi\b", r"\bhey\b",
            r"\bformat\b", r"\bsummary\b", r"\bshorten\b", r"\bexplain again\b"
        ]
        # Incomplete sentence endings or single trailing prepositions/connectives (G1 Wait Gate)
        self.incomplete_endings = {
            "a", "an", "the", "in", "at", "to", "for", "with", "and", "or",
            "is", "are", "was", "were", "of", "about", "that", "this", "my",
            "your", "their", "our", "what", "where", "who", "how", "if", "when"
        }

    def analyze_stream(self, current_transcript: str) -> Dict[str, Any]:
        text = current_transcript.strip()
        text_lower = text.lower()
        words = text_lower.split()

        # Gate G1: Empty or minimal input check
        if not words:
            return {
                "action": "WAIT",
                "gate": "G1_WAIT",
                "reason": "Empty input payload."
            }

        # Gate G4: Query Suppression Check (Formatting, filler, conversational commands)
        for pattern in self.suppress_patterns:
            if re.search(pattern, text_lower):
                return {
                    "action": "SUPPRESS",
                    "gate": "G4_SUPPRESS",
                    "reason": "Formatting, conversational filler, or non-retrieval command detected. Bypassing vector search."
                }

        # Gate G1: Mid-sentence incomplete input check
        if len(words) < 3 or words[-1] in self.incomplete_endings:
            return {
                "action": "WAIT",
                "gate": "G1_WAIT",
                "reason": "Sentence incomplete or context expanding. Holding for turn stability."
            }

        # Gate G3 & G2: Multi-Intent Decomposition & Provisional Commit
        intents = self._split_intents(text)
        
        return {
            "action": "RETRIEVE",
            "gate": "G3_COMMIT",
            "intents": intents,
            "reason": f"Stable turn state confirmed (Gate G3). Executing hybrid vector retrieval across {len(intents)} intent(s)."
        }

    def evaluate_fallback(self, search_results: List[Dict[str, Any]], is_cached_turn: bool = False) -> Dict[str, Any]:
        """
        Gate G5: Fallback Gate
        Evaluates whether search results pass threshold or require fallback strategy.
        """
        if is_cached_turn:
            return {
                "action": "COMMIT",
                "gate": "G3_COMMIT",
                "reason": "State-diffing preserved active context state without redundant retrieval."
            }

        if not search_results:
            return {
                "action": "FALLBACK",
                "gate": "G5_FALLBACK",
                "reason": "Zero high-confidence vector matches found. Triggering parametric synthesis / query broadening."
            }
        
        max_score = max(doc.get("score", 0.0) for doc in search_results)
        if max_score < 0.01:
            return {
                "action": "FALLBACK",
                "gate": "G5_FALLBACK",
                "reason": f"Top match score ({max_score:.4f}) below precision threshold tau_commit. Triggering G5 fallback."
            }
            
        return {
            "action": "COMMIT",
            "gate": "G3_COMMIT",
            "reason": f"Top match score ({max_score:.4f}) validated against tau_commit."
        }

    def _split_intents(self, transcript: str) -> List[str]:
        """
        Splits multi-intent compound inputs into distinct atomic sub-queries (Section 2).
        """
        # Split on conjunctions, multi-question markers, or clause delimiters
        parts = re.split(r'\band\b|\balso\b|\bwhat about\b|\bhow about\b|\bas well as\b|\?', transcript, flags=re.IGNORECASE)
        cleaned = [p.strip() for p in parts if len(p.strip()) > 3]
        return cleaned if cleaned else [transcript]

if __name__ == "__main__":
    controller = StreamIntentController()
    
    print("Test G1 Wait:", controller.analyze_stream("I want to know about the"))
    print("Test G4 Suppress:", controller.analyze_stream("Please summarize in bullet points"))
    print("Test G3 Commit:", controller.analyze_stream("What is the venue capacity and what is the cancellation fee?"))