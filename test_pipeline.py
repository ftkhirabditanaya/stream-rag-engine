import unittest
import asyncio
from intent_controller import StreamIntentController
from state_differ import SessionStateDiffer
from search_engine import deduplicate_results

class TestStreamingRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        self.controller = StreamIntentController()
        self.differ = SessionStateDiffer()

    def test_g1_wait_gate(self):
        """Test Gate G1: Incomplete sentence holds turn state."""
        res = self.controller.analyze_stream("I want to know about the")
        self.assertEqual(res["action"], "WAIT")
        self.assertEqual(res["gate"], "G1_WAIT")

    def test_g4_suppress_gate(self):
        """Test Gate G4: Summarize command bypasses vector search."""
        res = self.controller.analyze_stream("Please summarize in bullet points")
        self.assertEqual(res["action"], "SUPPRESS")
        self.assertEqual(res["gate"], "G4_SUPPRESS")

    def test_g3_multi_intent_decomposition(self):
        """Test Gate G3: Compound queries split into sub-queries."""
        res = self.controller.analyze_stream("What is the venue capacity and what is the cancellation fee?")
        self.assertEqual(res["action"], "RETRIEVE")
        self.assertEqual(res["gate"], "G3_COMMIT")
        self.assertEqual(len(res["intents"]), 2)
        self.assertIn("What is the venue capacity", res["intents"])
        self.assertIn("what is the cancellation fee", res["intents"])

    def test_state_differ_versioning(self):
        """Test State Differ versioning v1 -> v2 and delta extraction."""
        # Turn 1
        res1 = self.differ.process_intents(["What is the venue capacity"])
        self.assertEqual(res1["version"], "v1")
        self.assertEqual(res1["next_version"], "v2")
        self.assertTrue(res1["is_updated"])
        self.assertEqual(res1["delta_intents"], ["What is the venue capacity"])

        # Turn 2: Duplicate query (should yield empty delta)
        res2 = self.differ.process_intents(["What is the venue capacity"])
        self.assertEqual(res2["version"], "v1")
        self.assertFalse(res2["is_updated"])

        # Turn 3: New Intent (version bump to v2)
        res3 = self.differ.process_intents(["What is the cancellation fee?"])
        self.assertEqual(res3["version"], "v2")
        self.assertEqual(res3["next_version"], "v3")
        self.assertTrue(res3["is_updated"])
        self.assertEqual(res3["delta_intents"], ["What is the cancellation fee?"])

    def test_state_differ_reset(self):
        """Test resetting session state differ back to v1."""
        self.differ.process_intents(["Query 1"])
        self.differ.process_intents(["Query 2"])
        self.assertEqual(self.differ.version, 3)
        self.differ.reset()
        self.assertEqual(self.differ.version, 1)
        self.assertEqual(len(self.differ.active_entities), 0)

    def test_deduplicate_results(self):
        """Test search result deduplication by citation and text."""
        raw_docs = [
            {"citation": "[Doc_01 §1.2]", "text": "Capacity 150", "score": 0.85},
            {"citation": "[Doc_01 §1.2]", "text": "Capacity 150", "score": 0.92},
            {"citation": "[Doc_02 §3.1]", "text": "Cancellation 50%", "score": 0.78}
        ]
        deduped = deduplicate_results(raw_docs)
        self.assertEqual(len(deduped), 2)
        # Should retain the highest score (0.92) for Doc_01
        doc1 = next(d for d in deduped if d["citation"] == "[Doc_01 §1.2]")
        self.assertEqual(doc1["score"], 0.92)

if __name__ == "__main__":
    unittest.main()
