import re

class StreamIntentController:
    def __init__(self):
        # Trigger words indicating a reformat request (Gate G4 Suppression)
        self.suppress_patterns = [
            r"summarize", r"bullet points", r"format as", r"rephrase", 
            r"make it shorter", r"translate", r"simplify"
        ]
        # Common stop words to check chunk completeness
        self.incomplete_endings = ["a", "an", "the", "in", "at", "to", "for", "with", "and", "or"]

    def analyze_stream(self, current_transcript: str) -> dict:
        text = current_transcript.strip().lower()
        words = text.split()

        if not words:
            return {"action": "WAIT", "reason": "Empty input"}

        # 1. Gate G4: Query Suppression Check
        for pattern in self.suppress_patterns:
            if re.search(pattern, text):
                return {
                    "action": "SUPPRESS", 
                    "reason": "Formatting/Contextual update request detected. Bypassing vector search."
                }

        # 2. Gate G2: Speculative Retrieval Check (Mid-sentence stability)
        if len(words) < 3 or words[-1] in self.incomplete_endings:
            return {"action": "WAIT", "reason": "Sentence incomplete or context expanding."}

        # 3. Gate G3: Multi-Intent Detection
        intents = self._split_intents(current_transcript)
        
        return {
            "action": "RETRIEVE",
            "intents": intents,
            "reason": f"Stable entity detected. Executing {len(intents)} search(es)."
        }

    def _split_intents(self, transcript: str) -> list[str]:
        # Splits multi-intent queries on 'and', 'also', 'what about'
        parts = re.split(r'\band\b|\balso\b|\bwhat about\b|\?', transcript, flags=re.IGNORECASE)
        cleaned = [p.strip() for p in parts if len(p.strip()) > 3]
        return cleaned if cleaned else [transcript]

if __name__ == "__main__":
    controller = StreamIntentController()
    
    # Test Cases
    print(controller.analyze_stream("I want to know about the"))  # Should WAIT
    print(controller.analyze_stream("Format that as 2 bullet points"))  # Should SUPPRESS
    print(controller.analyze_stream("What is the venue capacity and what is the cancellation fee?"))  # Should RETRIEVE 2 intents