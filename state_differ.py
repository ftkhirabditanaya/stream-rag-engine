import hashlib
import re
from typing import List, Dict, Any, Set, Tuple

class SessionStateDiffer:
    """
    Dynamic State-Diffing Mechanics (Section 3 of System Architecture)
    Tracks active entities, graph hashes, and semantic query hashes across session turns.
    Prevents redundant vector similarity retrievals by extracting delta updates (Δ)
    and managing version bumps (v1 -> v2 -> v3...).
    """
    def __init__(self):
        self.version: int = 1
        self.active_hashes: Set[str] = set()
        self.active_entities: Set[str] = set()
        self.history: List[Dict[str, Any]] = []
        self.doc_cache: Dict[str, List[Dict[str, Any]]] = {}  # query_hash -> docs

    def _compute_hash(self, text: str) -> str:
        """Computes lightweight MD5 checksum for normalized query string."""
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:8]

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extracts key domain entities, terms, numbers, or capitalized nouns.
        """
        tokens = re.findall(r'\b[A-Za-z0-9_§#-]+\b', text)
        stop_words = {
            "what", "is", "the", "and", "or", "in", "on", "at", "to", "for",
            "with", "a", "an", "of", "how", "does", "that", "relate", "can",
            "please", "tell", "me", "about", "are", "there", "any", "which"
        }
        entities = [t.lower() for t in tokens if t.lower() not in stop_words and len(t) > 2]
        return list(dict.fromkeys(entities))  # Deduplicated preserving order

    def compute_graph_checksum(self) -> str:
        """Computes lightweight checksum of active session entities and parameters."""
        sorted_entities = sorted(list(self.active_entities))
        combined = "|".join(sorted_entities) + f"|v{self.version}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:8]

    def cache_docs_for_intent(self, intent: str, docs: List[Dict[str, Any]]):
        """Caches retrieved docs for a specific intent hash."""
        query_hash = self._compute_hash(intent)
        self.doc_cache[query_hash] = docs

    def get_cached_docs_for_intents(self, intents: List[str]) -> List[Dict[str, Any]]:
        """Retrieves cached docs for a list of intents."""
        cached_docs = []
        for intent in intents:
            query_hash = self._compute_hash(intent)
            if query_hash in self.doc_cache:
                cached_docs.extend(self.doc_cache[query_hash])
        return cached_docs

    def process_intents(self, intents: List[str]) -> Dict[str, Any]:
        """
        Calculates delta updates (Δ = {Added: [...], Invalidated: []})
        and returns delta sub-queries for selective vector fetch.
        """
        delta_intents: List[str] = []
        cached_intents: List[str] = []
        added_entities: Set[str] = set()
        new_hashes: Set[str] = set()

        for intent in intents:
            query_hash = self._compute_hash(intent)
            entities = self._extract_entities(intent)

            if query_hash not in self.active_hashes:
                delta_intents.append(intent)
                new_hashes.add(query_hash)
                added_entities.update(entities)
            else:
                cached_intents.append(intent)

        if delta_intents:
            # Commit new context nodes and bump state version
            self.active_hashes.update(new_hashes)
            self.active_entities.update(added_entities)
            
            # Store turn state
            current_version_str = f"v{self.version}"
            self.history.append({
                "version": current_version_str,
                "delta_intents": delta_intents,
                "added_entities": list(added_entities),
                "graph_checksum": self.compute_graph_checksum()
            })
            
            self.version += 1
            is_updated = True
        else:
            is_updated = False
            current_version_str = f"v{max(1, self.version - 1)}"

        return {
            "version": current_version_str,
            "next_version": f"v{self.version}",
            "delta_intents": delta_intents,
            "cached_intents": cached_intents,
            "added_entities": list(added_entities),
            "active_entities": sorted(list(self.active_entities)),
            "graph_checksum": self.compute_graph_checksum(),
            "is_updated": is_updated
        }

    def reset(self):
        """Resets the state differ back to initial state (v1)."""
        self.version = 1
        self.active_hashes.clear()
        self.active_entities.clear()
        self.history.clear()
        self.doc_cache.clear()

if __name__ == "__main__":
    differ = SessionStateDiffer()
    t1 = differ.process_intents(["What is the venue capacity"])
    print("Turn 1 State:", t1)
    t2 = differ.process_intents(["What is the venue capacity"])
    print("Turn 2 (Duplicate):", t2)
    t3 = differ.process_intents(["What is the cancellation fee?"])
    print("Turn 3 (Delta):", t3)

