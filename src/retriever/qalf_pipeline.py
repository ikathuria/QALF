"""
End-to-End QALF Pipeline.
Complete Query-Adaptive Learned Fusion system using Neo4j.
"""

from typing import List, Dict, Any, Tuple
import logging
import time
import yaml

from src.qalf.query_complexity import QueryComplexityClassifier
from src.qalf.query_intent import QueryIntentClassifier
from src.retriever.neo4j_retriever import Neo4jMultiModalRetriever
from src.retriever.qalf_fusion import QALFFusion
from src.generator.rag_generator import RAGGenerator
from configs.routing_table import route_to_modalities
from configs.alpha_weights import get_alpha_weights


class QALFPipeline:
    """
    Complete QALF pipeline implementing:
    1. Query complexity classification (4D)
    2. Intent classification
    3. Adaptive routing
    4. Multi-modal retrieval (Neo4j)
    5. Consensus-based fusion
    6. Adaptive weight computation
    """

    def __init__(
        self,
        neo4j_manager,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        config_path: str = "configs/hyperparameters.yaml",
        enable_generator: bool = False,
        generator_temperature: float = 0.3,
        learned_alpha=None,
        use_learned_routing: bool = False,
        use_learned_weighting: bool = True,
    ):
        """
        Initialize QALF pipeline.
        
        Args:
            neo4j_manager: Neo4jManager instance
            embedding_model: Sentence transformer model name
            embedding_dim: Embedding dimension
            config_path: Path to hyperparameters YAML
            enable_generator: Whether to enable RAG generation
            generator_temperature: Temperature for generator LLM
        """
        self.neo4j_manager = neo4j_manager
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        # Optional src.qalf.learned_alpha.LearnedAlphaWeights instance. When
        # provided, its predictions can replace either or both of: fusion
        # weighting (configs.alpha_weights.get_alpha_weights, "qalf_learned"
        # in SystemRegistry) and modality routing (configs.routing_table.
        # route_to_modalities, "qalf_learned_routing" in SystemRegistry) --
        # controlled independently so evaluating one doesn't change the other.
        self.learned_alpha = learned_alpha
        self.use_learned_routing = use_learned_routing
        self.use_learned_weighting = use_learned_weighting

        self._logger = self._setup_logging()
        
        # Load hyperparameters
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.complexity_classifier = QueryComplexityClassifier()
        self.intent_classifier = QueryIntentClassifier()
        self.retriever = Neo4jMultiModalRetriever(
            neo4j_manager,
            embedding_model,
            embedding_dim
        )
        self.fusion = QALFFusion(
            k=self.config.get("rrf", {}).get("k", 60),
            beta=self.config.get("consensus", {}).get("beta", 0.5)
        )
        
        # Initialize generator if enabled
        self.generator = None
        if enable_generator:
            try:
                self.generator = RAGGenerator(
                    neo4j_manager=neo4j_manager,
                    llm_temperature=generator_temperature
                )
                self._logger.info("RAG Generator enabled")
            except Exception as e:
                self._logger.warning(f"Failed to initialize generator: {e}")
                self._logger.warning("Continuing without generator (retrieval only)")
        
        self._logger.info("QALF Pipeline initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_config(self, config_path: str) -> dict:
        """Load hyperparameters from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                "rrf": {"k": 60},
                "consensus": {"beta": 0.5},
                "retrieval": {"top_k": 10, "top_k_final": 10}
            }

    def setup_indexes(self):
        """Setup Neo4j indexes (one-time operation)"""
        self.neo4j_manager.setup_indexes(self.embedding_dim)
        self._logger.info("✅ Neo4j indexes setup complete")

    def qalf_retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Complete QALF retrieval pipeline.
        
        Args:
            query: User query string
            top_k: Number of final results (defaults to config)
        
        Returns:
            List of ranked documents with doc_id, title, score, and metadata
        """
        pipeline_start_time = time.time()
        self._logger.info("=" * 80)
        self._logger.info(f"🚀 QALF PIPELINE START - Query: '{query}'")
        self._logger.info("=" * 80)
        
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k_final", 10)
        
        retrieval_top_k = self.config.get("retrieval", {}).get("top_k", 10)
        self._logger.info(f"📋 Configuration: top_k={top_k}, retrieval_top_k={retrieval_top_k}")
        
        # Step 1: Analyze query
        step_start = time.time()
        self._logger.info("\n📊 STEP 1: Query Analysis")
        self._logger.info("-" * 80)
        complexity = self.complexity_classifier.classify_complexity_4d(query)
        intent = self.intent_classifier.classify(query)
        step_time = time.time() - step_start
        self._logger.info(f"✅ Complexity: {complexity} | Intent: {intent} (took {step_time:.3f}s)")
        
        # Step 2: Route to modalities
        step_start = time.time()
        self._logger.info("\n🔄 STEP 2: Adaptive Routing")
        self._logger.info("-" * 80)
        if self.use_learned_routing and self.learned_alpha is not None:
            active_modalities = self.learned_alpha.route_to_modalities(complexity, intent)
            self._logger.info(f"   Routing source: learned")
        else:
            active_modalities = route_to_modalities(complexity, intent)
            self._logger.info(f"   Routing source: rule-based")
        step_time = time.time() - step_start
        self._logger.info(f"✅ Active modalities: {active_modalities} (took {step_time:.3f}s)")
        
        # Step 3: Generate query representations
        step_start = time.time()
        self._logger.info("\n🔧 STEP 3: Query Representation Generation")
        self._logger.info("-" * 80)
        query_embedding = self.retriever.get_embedding(query)
        query_entities = self.retriever.extract_entities(query)
        query_string = query  # For full-text search
        step_time = time.time() - step_start
        self._logger.info(f"✅ Generated embedding (dim={len(query_embedding)}) and extracted {len(query_entities)} entities (took {step_time:.3f}s)")
        if query_entities:
            self._logger.info(f"   Entities: {query_entities}")
        
        # Step 4: Parallel retrieval from Neo4j
        step_start = time.time()
        self._logger.info("\n🔍 STEP 4: Multi-Modal Retrieval")
        self._logger.info("-" * 80)
        retrieval_results = {}
        
        if "vector" in active_modalities:
            retrieval_results["vector"] = self.retriever.retrieve_vector(
                query_embedding,
                retrieval_top_k
            )
            self._logger.info(f"   Vector: {len(retrieval_results['vector'])} results")
        
        if "keyword" in active_modalities:
            retrieval_results["keyword"] = self.retriever.retrieve_keyword(
                query_string,
                retrieval_top_k
            )
            self._logger.info(f"   Keyword: {len(retrieval_results['keyword'])} results")
        
        if "graph" in active_modalities:
            retrieval_results["graph"] = self.retriever.retrieve_graph(
                query_entities,
                retrieval_top_k,
                query_text=query  # Pass query text for relationship detection
            )
            self._logger.info(f"   Graph: {len(retrieval_results['graph'])} results")
        
        step_time = time.time() - step_start
        total_retrieved = sum(len(results) for results in retrieval_results.values())
        self._logger.info(f"✅ Retrieval complete: {total_retrieved} total documents across {len(retrieval_results)} modalities (took {step_time:.3f}s)")
        
        # Step 5: Compute consensus scores
        step_start = time.time()
        self._logger.info("\n🤝 STEP 5: Consensus Score Computation")
        self._logger.info("-" * 80)
        consensus_scores = self.fusion.compute_consensus_scores(retrieval_results)
        step_time = time.time() - step_start
        self._logger.info(f"✅ Computed consensus scores for {len(consensus_scores)} documents (took {step_time:.3f}s)")
        if consensus_scores:
            max_consensus = max(consensus_scores.values())
            avg_consensus = sum(consensus_scores.values()) / len(consensus_scores)
            self._logger.debug(f"   Max consensus: {max_consensus:.3f}, Avg consensus: {avg_consensus:.3f}")
        
        # Step 6: Adaptive weight computation
        step_start = time.time()
        self._logger.info("\n⚖️  STEP 6: Adaptive Weight Computation")
        self._logger.info("-" * 80)
        if self.use_learned_weighting and self.learned_alpha is not None:
            alpha_intent = self.learned_alpha.predict_alpha(complexity, intent)
            self._logger.info(f"   Base weights (α_intent, learned): {alpha_intent}")
        else:
            alpha_intent = get_alpha_weights(intent)
            self._logger.info(f"   Base weights (α_intent): {alpha_intent}")
        weights = self.fusion.compute_adaptive_weights(
            retrieval_results,
            consensus_scores,
            alpha_intent
        )
        step_time = time.time() - step_start
        self._logger.info(f"✅ Adaptive weights computed (took {step_time:.3f}s)")
        for modality, weight in weights.items():
            self._logger.info(f"   {modality}: {weight:.4f}")
        
        # Step 7: Fuse with consensus
        step_start = time.time()
        self._logger.info("\n🔀 STEP 7: Consensus-Based Fusion")
        self._logger.info("-" * 80)
        ranked_results = self.fusion.fuse_with_consensus(
            retrieval_results,
            weights
        )
        step_time = time.time() - step_start
        self._logger.info(f"✅ Fusion complete: {len(ranked_results)} documents ranked (took {step_time:.3f}s)")
        
        # Step 8: Format and return top-K
        step_start = time.time()
        self._logger.info("\n📝 STEP 8: Result Formatting")
        self._logger.info("-" * 80)
        formatted_results = []
        for rank, (doc_id, score) in enumerate(ranked_results[:top_k], 1):
            # Find document metadata from retrieval results
            doc_metadata = self._find_doc_metadata(doc_id, retrieval_results)
            
            formatted_results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": doc_metadata.get("title", doc_id),
                "score": score,
                "consensus": consensus_scores.get(doc_id, 0.0),
                "complexity": complexity,
                "intent": intent,
                "modalities": active_modalities
            })
        step_time = time.time() - step_start
        
        total_time = time.time() - pipeline_start_time
        self._logger.info(f"✅ Formatted {len(formatted_results)} results (took {step_time:.3f}s)")
        self._logger.info("=" * 80)
        self._logger.info(f"🎉 QALF PIPELINE COMPLETE - Total time: {total_time:.3f}s")
        self._logger.info(f"   Returned {len(formatted_results)} documents")
        self._logger.info("=" * 80)
        
        return formatted_results

    def _find_doc_metadata(
        self,
        doc_id: str,
        retrieval_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Find document metadata from retrieval results"""
        for modality, docs in retrieval_results.items():
            for doc in docs:
                if doc.get("doc_id") == doc_id:
                    return doc
        return {}

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Main retrieval method (alias for qalf_retrieve with dict output).
        
        Args:
            query: User query string
            top_k: Number of final results
        
        Returns:
            Dictionary with results and metadata
        """
        results = self.qalf_retrieve(query, top_k)
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "success": len(results) > 0
        }

    def qalf_retrieve_and_generate(
        self,
        query: str,
        top_k: int = None,
        generate: bool = True
    ) -> Dict[str, Any]:
        """
        Complete QALF pipeline with optional generation.
        
        Args:
            query: User query string
            top_k: Number of final results
            generate: Whether to generate response (requires generator enabled)
        
        Returns:
            Dictionary with retrieval results and generated response
        """
        # Step 1: Retrieve documents
        retrieved_docs = self.qalf_retrieve(query, top_k)
        
        result = {
            "query": query,
            "retrieval": {
                "results": retrieved_docs,
                "count": len(retrieved_docs),
                "success": len(retrieved_docs) > 0
            },
            "generation": None
        }
        
        # Step 2: Generate response if enabled and requested
        if generate and self.generator and retrieved_docs:
            try:
                generation_result = self.generator.generate(
                    query=query,
                    retrieved_docs=retrieved_docs,
                    include_sources=True
                )
                result["generation"] = generation_result
            except Exception as e:
                self._logger.error(f"Generation failed: {e}")
                result["generation"] = {
                    "response": f"Error during generation: {str(e)}",
                    "success": False
                }
        elif generate and not self.generator:
            self._logger.warning("Generation requested but generator not enabled")
            result["generation"] = {
                "response": "Generator not enabled. Enable it during pipeline initialization.",
                "success": False
            }
        
        return result
