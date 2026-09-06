"""
RAG Generator Module for QALF Pipeline.
Generates responses using retrieved documents and LLM.
"""

from typing import List, Dict, Any, Optional
import logging
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

from src.neo4j.neo4j_manager import Neo4jManager
import src.utils.constants as C


class RAGGenerator:
    """
    RAG Generator that takes retrieved documents and generates responses.
    Fetches chunk content from Neo4j and uses LLM to generate answers.
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        llm_model: str = None,
        llm_temperature: float = 0.3,
        max_context_chunks: int = 5
    ):
        """
        Initialize RAG Generator.

        Args:
            neo4j_manager: Neo4jManager instance for fetching chunk content
            llm_model: LLM model name (defaults to C.GENERATOR_MODEL)
            llm_temperature: Temperature for LLM generation
            max_context_chunks: Maximum number of chunks to include in context
        """
        self.neo4j_manager = neo4j_manager
        self.llm_model = llm_model or C.GENERATOR_MODEL
        self.llm_temperature = llm_temperature
        self.max_context_chunks = max_context_chunks

        self._logger = self._setup_logging()

        # Same model used to embed Chunk/Table nodes at ingestion time (see
        # src/neo4j/vector_ingestion.py) -- needed to rank each document's own
        # chunks by query relevance instead of by position (see
        # fetch_chunk_content / _fetch_chunks_by_position).
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize LLM
        try:
            self.llm = OllamaLLM(
                model=self.llm_model,
                temperature=self.llm_temperature,
                base_url=C.OLLAMA_URI
            )
            self._logger.info(f"Initialized RAG Generator with LLM: {self.llm_model}")
        except Exception as e:
            self._logger.error(f"Failed to initialize LLM: {e}")
            raise

        # Setup prompt template with enhanced instructions for tables and numerical data
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context from retrieved documents.

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite sources using [Source: document_name, Chunk/Table X, Page Y] format
- Be concise but comprehensive
- If multiple sources mention the same information, you can combine them
- Maintain accuracy and avoid hallucination

SPECIAL INSTRUCTIONS FOR TABLES:
- When context includes tables, carefully read the table structure (rows, columns, headers)
- For questions about specific tables (e.g., "Table 3"), look for tables with matching identifiers
- Extract exact numerical values from tables (e.g., "65%" not "about 65%")
- Pay attention to column headers and row labels when answering questions
- If a table shows multiple models/metrics, identify which one matches the question

SPECIAL INSTRUCTIONS FOR COUNTING:
- For "how many times" questions, count ALL occurrences in the provided context
- If the context is limited, mention that the count is based on the provided context only
- Be precise with numbers"""),
            ("human", """Context from retrieved documents:

{context}

Question: {query}

Please provide a detailed answer based on the context above. Include source citations where relevant. For table questions, extract exact values. For counting questions, count all occurrences in the context.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

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

    def _is_counting_query(self, query: str) -> bool:
        """Check if query is asking for a count (e.g., 'how many times')"""
        if not query:
            return False
        counting_patterns = [
            r'how many times',
            r'how many mentions',
            r'how many occurrences',
            r'count.*times',
            r'number of times'
        ]
        import re
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in counting_patterns)

    def _fetch_chunks_by_position(
        self, doc_ids: List[str], max_chunks_per_doc: int
    ) -> List[Dict[str, Any]]:
        """First max_chunks_per_doc text chunks per document, by chunk_index."""
        cypher_query = """
        MATCH (c:Chunk)-[:IN_DOCUMENT]->(doc:Document)
        WHERE doc.id IN $doc_ids AND c.modality = 'text'
        WITH doc, c
        ORDER BY doc.id, c.chunk_index ASC
        WITH doc, collect(c)[..$max_chunks] AS chunks
        UNWIND chunks AS chunk
        RETURN doc.id AS doc_id,
               doc.source_file AS doc_title,
               chunk.id AS chunk_id,
               chunk.content AS content,
               chunk.modality AS modality,
               chunk.chunk_index AS chunk_index,
               COALESCE(chunk.page, 1) AS page
        """
        return self.neo4j_manager.query_graph(
            cypher_query, {"doc_ids": doc_ids, "max_chunks": max_chunks_per_doc}
        )

    def _fetch_chunks_by_relevance(
        self, doc_ids: List[str], max_chunks_per_doc: int, query_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Top max_chunks_per_doc text chunks per document, by cosine similarity
        to query_embedding -- exact (not approximate-index) similarity, since
        doc_ids is already a small, QALF-selected set of target documents."""
        cypher_query = """
        MATCH (c:Chunk)-[:IN_DOCUMENT]->(doc:Document)
        WHERE doc.id IN $doc_ids AND c.modality = 'text' AND c.embedding IS NOT NULL
              AND size(c.embedding) = size($query_embedding)
        WITH doc, c, vector.similarity.cosine(c.embedding, $query_embedding) AS score
        ORDER BY doc.id, score DESC
        WITH doc, collect({chunk: c, score: score})[..$max_chunks] AS top
        UNWIND top AS tc
        RETURN doc.id AS doc_id,
               doc.source_file AS doc_title,
               tc.chunk.id AS chunk_id,
               tc.chunk.content AS content,
               tc.chunk.modality AS modality,
               tc.chunk.chunk_index AS chunk_index,
               COALESCE(tc.chunk.page, 1) AS page,
               tc.score AS relevance_score
        """
        return self.neo4j_manager.query_graph(
            cypher_query,
            {"doc_ids": doc_ids, "max_chunks": max_chunks_per_doc, "query_embedding": query_embedding},
        )

    def fetch_chunk_content(
        self,
        doc_ids: List[str],
        max_chunks_per_doc: int = 3,
        query: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch chunk content for retrieved documents.
        
        Args:
            doc_ids: List of document IDs
            max_chunks_per_doc: Maximum chunks to fetch per document
            query: Optional query string for table-specific retrieval
        
        Returns:
            List of chunks with content, doc_id, and metadata
        """
        if not doc_ids:
            return []

        start_time = time.time()
        self._logger.debug(f"Fetching chunk content for {len(doc_ids)} documents")

        # For counting queries, fetch more chunks for accurate counting -- and
        # keep positional order, since counting needs broad coverage of the
        # document rather than the few chunks most similar to the query.
        is_counting = bool(query and self._is_counting_query(query))
        if is_counting:
            max_chunks_per_doc = 50
            self._logger.debug(f"Detected counting query - fetching {max_chunks_per_doc} chunks per doc")

        # Check if query mentions a specific table (e.g., "Table 3", "Table 5")
        table_identifier = None
        if query:
            import re
            table_match = re.search(r'Table\s+(\d+)', query, re.IGNORECASE)
            if table_match:
                table_identifier = f"Table {table_match.group(1)}"
                self._logger.debug(f"Detected table-specific query: {table_identifier}")

        # Rank each document's own chunks by cosine similarity to the query
        # instead of taking the first max_chunks_per_doc by position -- a
        # document's most relevant passage is rarely at the very start, so
        # positional selection was diluting context with irrelevant
        # front-matter regardless of what QALF's fusion actually retrieved.
        query_embedding = None
        if query and not is_counting:
            try:
                query_embedding = self.embedder.encode(query).tolist()
            except Exception as e:
                self._logger.warning(
                    f"Failed to embed query for relevance ranking, falling back to positional order: {e}"
                )

        try:
            if query_embedding is not None:
                chunk_results = self._fetch_chunks_by_relevance(doc_ids, max_chunks_per_doc, query_embedding)
                covered = {r.get("doc_id") for r in chunk_results}
                missing = [d for d in doc_ids if d not in covered]
                if missing:
                    # Documents with no embedded chunks (or an embedding-dim
                    # mismatch) fall back to positional order rather than
                    # being silently dropped from the context.
                    chunk_results = list(chunk_results) + list(
                        self._fetch_chunks_by_position(missing, max_chunks_per_doc)
                    )
            else:
                chunk_results = self._fetch_chunks_by_position(doc_ids, max_chunks_per_doc)

            # Then, fetch tables - prioritize specific table if mentioned in query
            if table_identifier:
                # Fetch specific table first
                table_query = """
                MATCH (t:Table)-[:IN_DOCUMENT]->(doc:Document)
                WHERE doc.id IN $doc_ids AND t.table_identifier = $table_identifier
                RETURN doc.id AS doc_id,
                       doc.source_file AS doc_title,
                       t.id AS chunk_id,
                       t.content AS content,
                       'table' AS modality,
                       COALESCE(t.page, 1) AS chunk_index,
                       COALESCE(t.page, 1) AS page,
                       t.table_identifier AS table_identifier
                ORDER BY doc.id, t.page ASC
                LIMIT 10
                """
                table_results = self.neo4j_manager.query_graph(
                    table_query,
                    {
                        "doc_ids": doc_ids,
                        "table_identifier": table_identifier
                    }
                )
                # Also fetch other tables if needed
                if not table_results:
                    table_query = """
                    MATCH (t:Table)-[:IN_DOCUMENT]->(doc:Document)
                    WHERE doc.id IN $doc_ids
                    RETURN doc.id AS doc_id,
                           doc.source_file AS doc_title,
                           t.id AS chunk_id,
                           t.content AS content,
                           'table' AS modality,
                           COALESCE(t.page, 1) AS chunk_index,
                           COALESCE(t.page, 1) AS page,
                           t.table_identifier AS table_identifier
                    ORDER BY doc.id, t.page ASC
                    LIMIT $max_chunks
                    """
                    table_results = self.neo4j_manager.query_graph(
                        table_query,
                        {
                            "doc_ids": doc_ids,
                            "max_chunks": max_chunks_per_doc
                        }
                    )
            else:
                # Fetch all tables
                table_query = """
                MATCH (t:Table)-[:IN_DOCUMENT]->(doc:Document)
                WHERE doc.id IN $doc_ids
                RETURN doc.id AS doc_id,
                       doc.source_file AS doc_title,
                       t.id AS chunk_id,
                       t.content AS content,
                       'table' AS modality,
                       COALESCE(t.page, 1) AS chunk_index,
                       COALESCE(t.page, 1) AS page,
                       t.table_identifier AS table_identifier
                ORDER BY doc.id, t.page ASC
                LIMIT $max_chunks
                """
                table_results = self.neo4j_manager.query_graph(
                    table_query,
                    {
                        "doc_ids": doc_ids,
                        "max_chunks": max_chunks_per_doc
                    }
                )

            # Combine results
            results = list(chunk_results) + list(table_results)

            chunks = []
            for record in results:
                chunks.append({
                    "doc_id": record.get("doc_id", ""),
                    "doc_title": record.get("doc_title", ""),
                    "chunk_id": record.get("chunk_id", ""),
                    "content": record.get("content", ""),
                    "modality": record.get("modality", "text"),
                    "chunk_index": record.get("chunk_index", 0),
                    "page": record.get("page", None)
                })

            elapsed_time = time.time() - start_time
            self._logger.debug(f"Fetched {len(chunks)} chunks in {elapsed_time:.3f}s")

            return chunks

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._logger.error(f"Failed to fetch chunk content after {elapsed_time:.3f}s: {e}")
            return []

    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        max_length: int = 4000
    ) -> str:
        """
        Build context string from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            max_length: Maximum context length in characters
        
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for chunk in chunks[:self.max_context_chunks]:
            content = chunk.get("content", "")
            doc_title = chunk.get("doc_title", chunk.get("doc_id", "Unknown"))
            chunk_idx = chunk.get("chunk_index", 0)
            page = chunk.get("page")
            modality = chunk.get("modality", "text")

            # Format with page number and modality information
            if modality == "table":
                page_info = f", Page {page}" if page else ""
                chunk_text = f"[Source: {doc_title}, Table {chunk_idx}{page_info}]\n{content}\n\n"
            else:
                page_info = f", Page {page}" if page else ""
                chunk_text = f"[Source: {doc_title}, Chunk {chunk_idx}{page_info}]\n{content}\n\n"

            if current_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "".join(context_parts)
        self._logger.debug(f"Built context with {len(context_parts)} chunks ({len(context)} chars)")

        return context

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response from retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries from QALF
            include_sources: Whether to include source citations
        
        Returns:
            Dictionary with generated response, sources, and metadata
        """
        generation_start = time.time()
        self._logger.info("=" * 80)
        self._logger.info(f"🤖 RAG GENERATION START - Query: '{query}'")
        self._logger.info("=" * 80)

        if not retrieved_docs:
            self._logger.warning("No documents retrieved, cannot generate response")
            return {
                "response": "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or check if documents have been ingested.",
                "sources": [],
                "chunks_used": 0,
                "generation_time": 0.0,
                "success": False
            }

        # Step 1: Extract document IDs
        step_start = time.time()
        self._logger.info("\n📚 STEP 1: Fetching Document Content")
        self._logger.info("-" * 80)
        doc_ids = [doc.get("doc_id", "") for doc in retrieved_docs if doc.get("doc_id")]
        self._logger.info(f"   Retrieved {len(doc_ids)} document IDs")

        # Step 2: Fetch chunk content (pass query for table-specific retrieval)
        chunks = self.fetch_chunk_content(doc_ids, max_chunks_per_doc=3, query=query)
        step_time = time.time() - step_start
        self._logger.info(f"✅ Fetched {len(chunks)} chunks from {len(doc_ids)} documents (took {step_time:.3f}s)")

        if not chunks:
            self._logger.warning("No chunk content found for retrieved documents")
            return {
                "response": "I found relevant documents but couldn't retrieve their content. This may indicate an issue with the database.",
                "sources": [doc.get("title", doc.get("doc_id", "")) for doc in retrieved_docs],
                "chunks_used": 0,
                "generation_time": time.time() - generation_start,
                "success": False
            }

        # Step 3: Build context
        step_start = time.time()
        self._logger.info("\n📝 STEP 2: Building Context")
        self._logger.info("-" * 80)
        context = self.build_context(chunks)
        step_time = time.time() - step_start
        self._logger.info(f"✅ Context built: {len(context)} characters from {len(chunks)} chunks (took {step_time:.3f}s)")

        # Step 4: Generate response
        step_start = time.time()
        self._logger.info("\n💬 STEP 3: Generating Response")
        self._logger.info("-" * 80)
        try:
            response = self.chain.invoke({
                "query": query,
                "context": context
            })
            step_time = time.time() - step_start
            self._logger.info(f"✅ Response generated (took {step_time:.3f}s)")
        except Exception as e:
            step_time = time.time() - step_start
            self._logger.error(f"❌ Generation failed after {step_time:.3f}s: {e}")
            response = f"I encountered an error while generating a response: {str(e)}"

        # Step 5: Extract sources
        sources = []
        if include_sources:
            seen_docs = set()
            for chunk in chunks:
                doc_title = chunk.get("doc_title", chunk.get("doc_id", "Unknown"))
                if doc_title not in seen_docs:
                    sources.append({
                        "title": doc_title,
                        "doc_id": chunk.get("doc_id", ""),
                        "chunks_used": sum(1 for c in chunks if c.get("doc_id") == chunk.get("doc_id"))
                    })
                    seen_docs.add(doc_title)

        total_time = time.time() - generation_start
        self._logger.info("=" * 80)
        self._logger.info(f"🎉 RAG GENERATION COMPLETE - Total time: {total_time:.3f}s")
        self._logger.info(f"   Response length: {len(response)} characters")
        self._logger.info(f"   Sources: {len(sources)} documents")
        self._logger.info("=" * 80)

        return {
            "response": response,
            "sources": sources,
            "chunks_used": len(chunks),
            "generation_time": total_time,
            "success": True,
            "query": query,
            "retrieved_docs_count": len(retrieved_docs)
        }
