import spacy
import logging
from typing import Tuple


class QueryComplexityClassifier:
    """
    4D Query Complexity Classifier for QALF.
    Classifies queries along 4 dimensions: Linguistic, Semantic, Modality, Contextual.
    Each dimension returns: "Low", "Medium", or "High"
    """

    def __init__(self, nlp_model="en_core_web_sm"):
        self._logger = logging.getLogger(self.__class__.__name__)
        try:
            self._logger.debug(f"Loading spaCy model: {nlp_model}")
            self.nlp = spacy.load(nlp_model)
            self._logger.debug(f"spaCy model loaded successfully")
        except OSError:
            raise ValueError(
                f"spaCy model '{nlp_model}' not found. "
                f"Install it with: python -m spacy download {nlp_model}"
            )

    def classify_linguistic(self, query: str) -> str:
        """
        Linguistic Complexity: Parse tree depth, sentence length.
        Returns: "Low", "Medium", or "High"
        """
        doc = self.nlp(query)
        
        # Parse tree depth (max depth of dependency tree)
        # Use a safer approach: find root and compute depth from root
        max_depth = 0
        for token in doc:
            depth = 0
            current = token
            visited = set()  # Safety: prevent infinite loops on malformed parses
            max_iterations = 100

            # Walk up to the root, which spaCy marks via head == self.
            while current.head != current and depth < max_iterations:
                if id(current) in visited:
                    self._logger.warning(f"Cycle detected in dependency tree for token: {token.text}")
                    break
                visited.add(id(current))
                depth += 1
                current = current.head

            if depth >= max_iterations:
                self._logger.warning(f"Max iterations reached for token: {token.text}, using depth={max_iterations}")

            max_depth = max(max_depth, depth)
        
        # Sentence length
        sentence_length = len(doc)
        
        # Classification thresholds
        if sentence_length <= 10 and max_depth <= 3:
            return "Low"
        elif sentence_length <= 20 and max_depth <= 5:
            return "Medium"
        else:
            return "High"

    def classify_semantic(self, query: str) -> str:
        """
        Semantic Complexity: Entity count, relationship depth, reasoning steps.
        Returns: "Low", "Medium", or "High"
        """
        doc = self.nlp(query)
        
        # Entity count
        entity_count = len(doc.ents)
        
        # Relationship indicators (comparative, causal, temporal)
        relationship_keywords = {
            "compare", "versus", "vs", "difference", "between",
            "why", "how does", "because", "caused", "due to",
            "before", "after", "when", "since", "trend", "over time",
            "and", "then", "after that", "sequence", "process"
        }
        relationship_count = sum(
            1 for token in doc 
            if token.lemma_.lower() in relationship_keywords
        )
        
        # Multi-hop indicators (multiple questions, conjunctions)
        question_count = query.count("?")
        conjunction_count = sum(1 for token in doc if token.dep_ == "cc")
        
        # Classification
        complexity_score = entity_count + relationship_count + question_count + conjunction_count
        
        if complexity_score <= 2:
            return "Low"
        elif complexity_score <= 5:
            return "Medium"
        else:
            return "High"

    def classify_modality(self, query: str) -> str:
        """
        Modality Complexity: Visual/tabular keywords.
        Returns: "Low", "Medium", or "High"
        """
        query_lower = query.lower()
        
        # Visual/tabular keywords
        modality_keywords = {
            "show", "chart", "table", "graph", "visualize",
            "diagram", "figure", "image", "map", "list",
            "spreadsheet", "plot", "visual"
        }
        
        keyword_count = sum(1 for keyword in modality_keywords if keyword in query_lower)
        
        if keyword_count == 0:
            return "Low"
        elif keyword_count <= 2:
            return "Medium"
        else:
            return "High"

    def classify_contextual(self, query: str) -> str:
        """
        Contextual Complexity: Domain-specificity (general vs expert-level).
        Returns: "Low", "Medium", or "High"
        """
        query_lower = query.lower()
        
        # Contextual references (pronouns, references to prior context)
        context_phrases = {
            "as above", "previous", "that", "those", "it", 
            "refer", "continue", "mentioned", "discussed"
        }
        
        # Domain-specific indicators (expert-level terms)
        # This can be customized based on your domain (cybersecurity in this case)
        domain_keywords = {
            "vulnerability", "exploit", "mitigation", "attack vector",
            "threat model", "adversarial", "poisoning", "injection",
            "cve", "cwe", "mitre", "tactics", "techniques"
        }
        
        context_count = sum(1 for phrase in context_phrases if phrase in query_lower)
        domain_count = sum(1 for keyword in domain_keywords if keyword in query_lower)
        
        complexity_score = context_count + (domain_count * 2)  # Domain terms weighted higher
        
        if complexity_score == 0:
            return "Low"
        elif complexity_score <= 2:
            return "Medium"
        else:
            return "High"

    def classify_complexity_4d(self, query: str) -> Tuple[str, str, str, str]:
        """
        Main classification method returning 4D complexity tuple.
        
        Returns:
            Tuple of (linguistic, semantic, modality, contextual)
            Each value is "Low", "Medium", or "High"
        """
        import time
        start_time = time.time()
        self._logger.debug(f"Starting 4D complexity classification for query: '{query[:50]}...'")
        
        try:
            self._logger.debug("  → Classifying linguistic complexity...")
            linguistic = self.classify_linguistic(query)
            self._logger.debug(f"    Linguistic: {linguistic}")
            
            self._logger.debug("  → Classifying semantic complexity...")
            semantic = self.classify_semantic(query)
            self._logger.debug(f"    Semantic: {semantic}")
            
            self._logger.debug("  → Classifying modality complexity...")
            modality = self.classify_modality(query)
            self._logger.debug(f"    Modality: {modality}")
            
            self._logger.debug("  → Classifying contextual complexity...")
            contextual = self.classify_contextual(query)
            self._logger.debug(f"    Contextual: {contextual}")
            
            elapsed = time.time() - start_time
            self._logger.debug(f"4D complexity classification completed in {elapsed:.3f}s")
            
            return (linguistic, semantic, modality, contextual)
        except Exception as e:
            elapsed = time.time() - start_time
            self._logger.error(f"Error in complexity classification after {elapsed:.3f}s: {e}")
            raise

    def classify(self, query: str) -> dict:
        """
        Legacy method for backward compatibility.
        Returns dict with complexity scores.
        """
        linguistic, semantic, modality, contextual = self.classify_complexity_4d(query)
        
        # Convert to numeric scores for backward compatibility
        level_to_score = {"Low": 1, "Medium": 2, "High": 3}
        
        return {
            "linguistic": level_to_score[linguistic],
            "semantic": level_to_score[semantic],
            "modality": level_to_score[modality],
            "contextual": level_to_score[contextual]
        }

    def total_score(self, query: str) -> int:
        """Legacy method: returns total numeric score."""
        scores = self.classify(query)
        return sum(scores.values())
