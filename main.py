"""
Complete Adaptive RAG Pipeline with QALF integration.

This script orchestrates:
1. Document parsing (text, images, tables)
2. Graph ingestion with LLM-based entity/relationship extraction
3. Vector embedding generation and storage
4. Adaptive query-based retrieval with QALF pipeline
5. Interactive query interface

Usage:
    python main.py [--mode {ingest|query|both|evaluate}]
"""

from glob import glob
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
import argparse

from src.preprocessing.document_parser import DocumentParser
from src.neo4j.graph_ingestion import GraphDBManager
from src.neo4j.vector_ingestion import VectorDBManager
from src.neo4j.neo4j_manager import Neo4jManager
from src.retriever.qalf_pipeline import QALFPipeline
from src.utils.base_parser import ParsedContent
import src.utils.constants as C


def load_parsed_contents(parsed_pkl_paths: List[str]) -> List[ParsedContent]:
    """Load previously parsed content from pickle files"""
    parsed_contents = []
    for pkl_path in parsed_pkl_paths:
        try:
            parsed_content = ParsedContent.from_pkl(pkl_path)
            parsed_contents.append(parsed_content)
            print(f"✓ Loaded: {pkl_path}")
        except Exception as e:
            print(f"✗ Failed to load {pkl_path}: {str(e)}")
    return parsed_contents


def ingest_pipeline(file_paths: List[str]) -> Dict[str, Any]:
    """
    Run the ingestion pipeline: parse → graph → embeddings

    Args:
        file_paths: List of document paths to ingest

    Returns:
        Dict with ingestion statistics
    """
    stats = {"parsed": 0, "graph": {}, "embeddings": {}}

    # Step 0: Filter redundant files
    print("\n" + "=" * 80)
    print("STEP 0: Checking for Redundant Files")
    print("=" * 80)

    # We need a temporary Neo4jManager to check existing files
    temp_neo4j = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )
    existing_docs = temp_neo4j.get_existing_document_ids()
    temp_neo4j.close()

    files_to_process = []
    skipped_files = []

    for f in file_paths:
        # Normalize path separators for consistent comparison
        normalized_path = f.replace("\\", "/")
        # Check if the file path (or maybe just basename?) matches.
        # The storage stores the path provided during ingestion.
        # Ideally we should store absolute paths or hashes, but based on current code it stores the path string.
        # We'll check if the path is in the existing set.

        # NOTE: existing_docs contains what was stored in d.source_file or d.id
        # Let's assume d.id was set to file path or source_file.
        # In graph_ingestion.py: doc_id = parsed_content.source_file.

        if f in existing_docs or normalized_path in existing_docs:
            skipped_files.append(f)
        else:
            files_to_process.append(f)

    if skipped_files:
        print(f"✓ Skipping {len(skipped_files)} already ingested files:")
        for f in skipped_files:
            print(f"  - {f}")

    if not files_to_process:
        print("\n✓ All files already ingested. Nothing to do.")
        return stats

    # Step 1: Parse documents
    print("\n" + "=" * 80)
    print(f"STEP 1: Parsing {len(files_to_process)} Documents")
    print("=" * 80)

    parser_config = {"dtype": "auto", "device_map": "auto"}
    parser = DocumentParser(config=parser_config)
    parsed_contents = parser.parse_batch(files_to_process)
    stats["parsed"] = len(parsed_contents)

    print(f"\n✓ Parsed {len(parsed_contents)} documents successfully")
    for pc in parsed_contents:
        print(
            f"  - {pc.source_file}: "
            f"{len(pc.text.get('content', ''))} chars text, "
            f"{len(pc.images)} images, "
            f"{len(pc.tables)} tables"
        )

    # Step 2: Initialize LLM
    print("\n" + "=" * 80)
    print("STEP 2: Initializing LLM")
    print("=" * 80)

    # Two guards against a single graph-extraction call stalling for hours
    # (observed twice: one call ran ~17.5 hours, another was killed after
    # 25+ min with zero output). httpx read-timeout alone does NOT catch
    # this -- Ollama streams tokens, and the read-timeout resets on every
    # chunk received, so a model stuck slowly drip-feeding tokens in a
    # non-terminating loop never trips it. num_predict caps the model's
    # own generation length server-side, which bounds worst-case runaway
    # output regardless of streaming behavior; client_kwargs timeout still
    # catches a genuinely dead connection.
    llm = OllamaLLM(
        model=C.OLLAMA_MODEL,
        temperature=0.0,
        base_url=C.OLLAMA_URI,
        num_predict=4096,
        client_kwargs={"timeout": 180.0},
    )
    print(f"✓ LLM initialized: Ollama ({C.OLLAMA_MODEL})")

    # Step 3: Graph ingestion
    print("\n" + "=" * 80)
    print("STEP 3: Graph Ingestion (LLM-based)")
    print("=" * 80)

    graph_manager = GraphDBManager(
        llm=llm,
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    graph_stats = graph_manager.ingest_batch_parsed_content(parsed_contents)
    stats["graph"] = graph_stats

    print(f"\n✓ Graph Ingestion Complete:")
    print(f"  - Documents: {graph_stats['documents']}")
    print(f"  - Text Chunks: {graph_stats['text_chunks']}")
    print(f"  - Images: {graph_stats['images']}")
    print(f"  - Tables: {graph_stats['tables']}")
    print(f"  - Entities: {graph_stats['entities']}")
    print(f"  - Relationships: {graph_stats['relationships']}")

    # Step 4: Vector embeddings
    print("\n" + "=" * 80)
    print("STEP 4: Vector Embedding Generation")
    print("=" * 80)

    vector_manager = VectorDBManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
        text_embedding_model=C.TRANSFORMER_EMBEDDING_MODEL,
    )

    embedding_stats = vector_manager.batch_embed_parsed_contents(parsed_contents)
    stats["embeddings"] = embedding_stats

    print(f"\n✓ Embedding Complete:")
    print(f"  - Text Embeddings: {embedding_stats['text_embeddings']}")
    print(f"  - Image Embeddings: {embedding_stats['image_embeddings']}")
    print(f"  - Table Embeddings: {embedding_stats['table_embeddings']}")
    print(f"  - Entity Embeddings: {embedding_stats['entity_embeddings']}")

    # Cleanup
    graph_manager.close()
    vector_manager.close()

    return stats


def query_pipeline():
    """
    Run interactive query pipeline with QALF adaptive retrieval
    """
    print("\n" + "=" * 80)
    print("INITIALIZING QALF QUERY PIPELINE")
    print("=" * 80)

    # Initialize Neo4j connection
    print("\n1. Connecting to Neo4j...")
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )
    print("✓ Neo4j connected")

    # Initialize QALF pipeline
    print("\n2. Initializing QALF Pipeline...")
    # Note: all-MiniLM-L6-v2 produces 384-dimensional embeddings
    # Enable generator for RAG responses
    enable_generator = True  # Set to False to disable generation

    # Model configuration - can be changed here or via environment variables
    import os

    llm_model = os.getenv("OLLAMA_MODEL", C.GENERATOR_MODEL)  # Default: llama3.1:8b
    llm_temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

    print(f"   Using LLM: {llm_model} (temperature: {llm_temperature})")
    print(f"   To change model, set GENERATOR_MODEL environment variable")
    print(f"   Example: export OLLAMA_MODEL=llama3.2:3b")

    qalf_pipeline = QALFPipeline(
        neo4j_manager=neo4j_manager,
        embedding_model=C.TRANSFORMER_EMBEDDING_MODEL,
        embedding_dim=384,  # Dimension for all-MiniLM-L6-v2
        enable_generator=enable_generator,
        generator_temperature=llm_temperature,
    )

    # Override generator LLM model if different from default
    if enable_generator and qalf_pipeline.generator and llm_model != C.GENERATOR_MODEL:
        from langchain_ollama import OllamaLLM
        from langchain_core.output_parsers import StrOutputParser

        qalf_pipeline.generator.llm = OllamaLLM(
            model=llm_model, temperature=llm_temperature, base_url=C.OLLAMA_URI
        )
        qalf_pipeline.generator.chain = (
            qalf_pipeline.generator.prompt_template
            | qalf_pipeline.generator.llm
            | StrOutputParser()
        )
        print(f"   ✓ Generator LLM updated to: {llm_model}")
    print("✓ QALF pipeline initialized")
    if enable_generator and qalf_pipeline.generator:
        print("✓ RAG Generator enabled")

    # Setup indexes (one-time operation, safe to call multiple times)
    print("\n3. Setting up QALF indexes...")
    try:
        qalf_pipeline.setup_indexes()
        print("✓ QALF indexes ready")
    except Exception as e:
        print(f"⚠️  Index setup note: {e}")
        print("Continuing (indexes may already exist)...")

    print("\n" + "=" * 80)
    print("ADAPTIVE QUERY INTERFACE (QALF)")
    print("=" * 80)
    print("Commands:")
    print("  - Enter a query to search")
    print("  - 'stats' - Show database statistics")
    print("  - 'help' - Show this help message")
    print("  - 'quit' or 'exit' - Exit the interface")
    print("=" * 80)

    # Interactive loop
    while True:
        print("\n" + "-" * 80)
        query = input("Query> ").strip()

        if not query:
            continue

        if query.lower() in ["quit", "exit"]:
            print("\n✓ Exiting query interface...")
            break

        if query.lower() == "help":
            print("\nAvailable commands:")
            print("  - Enter any natural language query")
            print("  - 'stats' - Show database statistics")
            print("  - 'quit' or 'exit' - Exit")
            continue

        if query.lower() == "stats":
            print("\nFetching database statistics...")
            stats_query = """
            MATCH (d:Document) WITH count(d) as docs
            MATCH (c:Chunk) WITH docs, count(c) as chunks
            MATCH (e:Entity) WITH docs, chunks, count(e) as entities
            MATCH ()-[r]->() WITH docs, chunks, entities, count(r) as rels
            RETURN docs, chunks, entities, rels
            """
            result = neo4j_manager.query_graph(stats_query)
            if result:
                r = result[0]
                print(f"\n📊 Database Statistics:")
                print(f"  - Documents: {r.get('docs', 0)}")
                print(f"  - Chunks: {r.get('chunks', 0)}")
                print(f"  - Entities: {r.get('entities', 0)}")
                print(f"  - Relationships: {r.get('rels', 0)}")
            continue

        # Process query with QALF
        print(f"\n🔍 Processing: '{query}'")

        try:
            # Use QALF pipeline with generation if enabled
            if qalf_pipeline.generator:
                full_result = qalf_pipeline.qalf_retrieve_and_generate(
                    query, top_k=7, generate=True
                )
                results = full_result.get("retrieval", {}).get("results", [])
                generation = full_result.get("generation")
            else:
                results = qalf_pipeline.qalf_retrieve(query, top_k=7)
                generation = None

            print("\n" + "=" * 80)
            print("=== QALF Adaptive Output ===")
            print("=" * 80)

            if results:
                # Display complexity and intent from first result (all results have same metadata)
                first_result = results[0]
                complexity = first_result.get("complexity", "N/A")
                intent = first_result.get("intent", "N/A")
                modalities = first_result.get("modalities", [])

                print(f"\n📊 Query Analysis:")
                print(f"  Complexity (4D): {complexity}")
                print(f"  Intent: {intent}")
                print(f"  Active Modalities: {', '.join(modalities)}")

                # Display generated response if available
                if generation and generation.get("success"):
                    print(f"\n💬 Generated Response:")
                    print("-" * 80)
                    print(generation.get("response", ""))
                    print("-" * 80)

                    # Display sources
                    sources = generation.get("sources", [])
                    if sources:
                        print(f"\n📚 Sources ({len(sources)} documents):")
                        for i, source in enumerate(sources, 1):
                            print(
                                f"  {i}. {source.get('title', 'Unknown')} "
                                f"({source.get('chunks_used', 0)} chunks)"
                            )

                    print(
                        f"\n⏱️  Generation time: {generation.get('generation_time', 0):.2f}s"
                    )
                    print("-" * 80)

                print(f"\n📄 Top Results [{len(results)}]:")
                print("-" * 80)

                for result in results:
                    print(f"\n[{result['rank']}] {result['title']}")
                    print(
                        f"    Score: {result['score']:.4f} | "
                        f"Consensus: {result['consensus']:.2f}"
                    )

                    # Try to get additional metadata if available
                    doc_id = result.get("doc_id", "")
                    if doc_id:
                        print(f"    Doc ID: {doc_id}")

                print("\n" + "-" * 80)
            else:
                print("\n⚠️  No results found.")
                print("This could mean:")
                print("  - No documents match the query")
                print("  - Indexes may need to be created")
                print("  - Documents may need to be ingested first")

            print("=" * 80)

        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback

            traceback.print_exc()
            print()

    # Cleanup
    neo4j_manager.close()
    print("\n✓ QALF pipeline closed")


def evaluate_pipeline(config, dataset, systems, output_dir):
    """Legacy evaluation entrypoint; superseded by evaluate.py's function-based evaluators."""
    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE")
    print("=" * 80)
    print("This mode is superseded by evaluate.py (see `python evaluate.py --mode all`).")


def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(description="Adaptive RAG Pipeline with QALF")
    parser.add_argument(
        "--mode",
        choices=["ingest", "query", "both", "evaluate"],
        default="both",
        help="Pipeline mode: ingest documents, query interface, both, or evaluate pipeline",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[C.TEST_PDF],
        help="Document files to ingest (for ingest/both modes)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="clean",
        help="Dataset key in config (e.g., 'clean')",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["all"],
        help="Systems to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ADAPTIVE MULTIMODAL RAG PIPELINE WITH QALF")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")

    try:
        if args.mode in ["ingest", "both"]:
            expanded_files = []
            for pattern in args.files:
                if "*" in pattern:
                    expanded_files.extend(glob(pattern))
                else:
                    expanded_files.append(pattern)
            args.files = expanded_files
            print(f"Files to ingest: {args.files}")
            stats = ingest_pipeline(args.files)

            print("\n" + "=" * 80)
            print("INGESTION SUMMARY")
            print("=" * 80)
            print(f"✓ Parsed: {stats['parsed']} documents")
            print(f"✓ Graph Nodes: {stats['graph'].get('entities', 0)} entities")
            print(
                f"✓ Graph Relations: {stats['graph'].get('relationships', 0)} relationships"
            )
            print(f"✓ Vector Embeddings: {sum(stats['embeddings'].values())} total")

        if args.mode in ["query", "both"]:
            if args.mode == "both":
                input("\nPress Enter to start query interface...")
            query_pipeline()

        if args.mode == "evaluate":
            if args.dataset:
                evaluate_pipeline(args.config, args.dataset, args.systems, args.output_dir)
            else:
                print("\nNo dataset specified for evaluation")

        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Pipeline error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
