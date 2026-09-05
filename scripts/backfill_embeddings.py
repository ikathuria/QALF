"""
Backfills missing embeddings for Chunk/Table/Image nodes.

Root cause: main.py's ingest_pipeline runs graph-ingestion (STEP 3) for a
whole batch of documents, then vector-embedding (STEP 4) for the same
batch -- but STEP 3 commits each document's graph data to Neo4j as it
completes, while STEP 4 only runs after the *entire* batch's graph step
returns without exception. Several ingestion runs this session crashed or
were killed partway through a multi-document batch (see git history /
session notes), leaving documents with complete graph data but a
never-run STEP 4 -- and the redundant-file check only looks for the
Document node, so they're silently skipped on retry rather than repaired.

This script finds every Chunk/Table/Image node with a NULL embedding and
backfills it directly, matching src/neo4j/vector_ingestion.py's exact
per-type text source: Chunk.content, Table.content (truncated to 500
chars, same as vector_ingestion.py), Image.caption.

Usage:
    python scripts/backfill_embeddings.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

from src.neo4j.neo4j_manager import Neo4jManager
import src.utils.constants as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def backfill_label(session, model, label: str, text_property: str, truncate: int = None):
    query = f"""
        MATCH (n:{label})
        WHERE n.embedding IS NULL AND n.{text_property} IS NOT NULL
        RETURN n.id as id, n.{text_property} as text
        LIMIT {BATCH_SIZE}
    """
    total = 0
    while True:
        records = list(session.run(query))
        if not records:
            break
        ids = [r["id"] for r in records]
        texts = [
            (r["text"][:truncate] if truncate else r["text"]) or f"{label} (no content)"
            for r in records
        ]
        embeddings = model.encode(texts).tolist()

        session.run(
            f"""
            UNWIND $rows as row
            MATCH (n:{label} {{id: row.id}})
            SET n.embedding = row.embedding
            """,
            {"rows": [{"id": i, "embedding": e} for i, e in zip(ids, embeddings)]},
        )
        total += len(ids)
        logger.info(f"  Backfilled {total} {label} nodes so far...")
    return total


def main():
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI, username=C.NEO4J_USERNAME, password=C.NEO4J_PASSWORD, database=C.NEO4J_DB
    )
    model = SentenceTransformer(C.TRANSFORMER_EMBEDDING_MODEL)

    with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
        logger.info("Backfilling Chunk embeddings...")
        n_chunks = backfill_label(session, model, "Chunk", "content")
        logger.info("Backfilling Table embeddings...")
        n_tables = backfill_label(session, model, "Table", "content", truncate=500)
        logger.info("Backfilling Image embeddings...")
        n_images = backfill_label(session, model, "Image", "caption")

    neo4j_manager.close()
    logger.info(
        f"Done. Backfilled {n_chunks} Chunk, {n_tables} Table, {n_images} Image embeddings."
    )


if __name__ == "__main__":
    main()
