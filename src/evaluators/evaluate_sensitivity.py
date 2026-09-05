import os
import json
import glob
import pandas as pd
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from src.neo4j.neo4j_manager import Neo4jManager
from src.utils.systems import SystemRegistry
import src.utils.constants as C
from src.utils.metrics import ndcg_at_k, recall_at_k

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sensitivity_analysis(
    target_dir: str,
    betas: List[float],
    output_dir: str = "data/results",
    top_k: int = 10,
):
    logger.info(f"Initializing sensitivity analysis with top_k={top_k}...")

    # Connect to Neo4j
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    # Initialize Systems
    registry = SystemRegistry(neo4j_manager)

    # Sweep beta across ALL queries in ALL document subdirectories under
    # target_dir's parent corpus (not just the first query of the first
    # document) -- a single-query sweep can't show a meaningful
    # utility-robustness curve, since one easy query may score identically
    # across every beta regardless of whether consensus weighting is doing
    # anything on harder queries.
    corpus_dir = os.path.dirname(os.path.normpath(target_dir))
    subdirs = [
        d for d in glob.glob(os.path.join(corpus_dir, "*")) if os.path.isdir(d)
    ]

    query_gold_pairs = []  # (query, relevant_basename)
    for subdir in subdirs:
        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))
        pdf_files = glob.glob(os.path.join(subdir, "*.pdf"))
        if not qa_files or not pdf_files:
            continue
        # Match by basename, not full path: the document parser caches parsed
        # content by filename, so a document ingested earlier under a different
        # path prefix (e.g. re-ingested from a reorganized data/raw directory)
        # keeps its original stored doc_id -- an exact full-path match would
        # spuriously fail even though it's the same document.
        relevant_basename = os.path.basename(os.path.normpath(pdf_files[0]))
        with open(qa_files[0], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    query_gold_pairs.append((item["question"], relevant_basename))
                except Exception:
                    continue

    if not query_gold_pairs:
        logger.error(f"No (query, gold) pairs found under {corpus_dir}")
        neo4j_manager.close()
        return
    logger.info(f"Sweeping beta across {len(query_gold_pairs)} queries from {len(subdirs)} documents.")

    sensitivity_results = []

    for beta in tqdm(betas, desc="Varying Beta"):
        # Update registry's QALF beta manually
        registry.qalf.fusion.beta = beta

        ndcg_scores = []
        for query, relevant_basename in query_gold_pairs:
            # Call retrieval directly (not run_qalf/qalf_retrieve_and_generate)
            # to skip an unnecessary LLM generation call per query -- this
            # metric only needs the ranked retrieval results.
            results = registry.qalf.qalf_retrieve(query, top_k=top_k)
            retrieved_ids = [
                os.path.basename(os.path.normpath(
                    res.get("id") or res.get("doc_id") or res.get("source") or ""
                ))
                for res in results
            ]
            ndcg_scores.append(ndcg_at_k(retrieved_ids, {relevant_basename}, k=top_k))

        mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
        sensitivity_results.append(
            {
                "Beta": beta,
                f"Mean_NDCG@{top_k}": mean_ndcg,
                "N_Queries": len(ndcg_scores),
            }
        )

    # Save results
    df = pd.DataFrame(sensitivity_results)
    print("\n=== Sensitivity Analysis Summary ===")
    print(df)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "sensitivity_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Sensitivity results saved to {output_path}")

    neo4j_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw/DocBench/P19-1598")
    args = parser.parse_args()

    betas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    run_sensitivity_analysis(args.dir, betas)
