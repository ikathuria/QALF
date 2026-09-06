import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Any
from tqdm import tqdm
import logging
import argparse

from src.neo4j.neo4j_manager import Neo4jManager
from src.utils.systems import SystemRegistry
import src.utils.constants as C
from src.utils.metrics import ndcg_at_k, recall_at_k
from src.qalf.query_complexity import QueryComplexityClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_path(path: str) -> str:
    return os.path.normpath(path)


def get_ingested_documents(neo4j_manager: Neo4jManager) -> Set[str]:
    query = "MATCH (d:Document) RETURN d.id as id"
    with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
        result = session.run(query)
        documents = {normalize_path(record["id"]) for record in result}
    return documents


def get_retrieved_doc_ids(results: List[Dict[str, Any]]) -> List[str]:
    doc_ids = []
    for res in results:
        if "doc_id" in res:
            doc_ids.append(normalize_path(res["doc_id"]))
        elif "source" in res:
            doc_ids.append(normalize_path(res["source"]))
        elif "metadata" in res and "source" in res["metadata"]:
            doc_ids.append(normalize_path(res["metadata"]["source"]))
    return doc_ids


def run_ablation_study(
    doc_bench_dir: str,
    limit: int = None,
    output_dir: str = "data/results",
    top_k: int = 10,
):
    logger.info(f"Starting Ablation Study with top_k={top_k}...")
    output_file = os.path.join(output_dir, "ablation_results.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )

    registry = SystemRegistry(neo4j_manager)
    complexity_classifier = QueryComplexityClassifier()
    ingested_docs = get_ingested_documents(neo4j_manager)

    if not ingested_docs:
        logger.error("No documents found in Neo4j.")
        return

    subdirs = glob.glob(os.path.join(doc_bench_dir, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    if limit:
        subdirs = subdirs[:limit]

    all_results = []

    systems = ["vector_only", "fixed_rrf", "adaptive_fixed", "qalf", "qalf_learned_routing"]

    for subdir in tqdm(subdirs, desc="Ablation Evaluation"):
        if not os.path.isdir(subdir):
            continue

        pdf_files = glob.glob(os.path.join(subdir, "*.pdf"))
        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))

        if not pdf_files or not qa_files:
            continue

        pdf_path = normalize_path(pdf_files[0])
        qa_path = qa_files[0]

        target_doc_id = None
        if pdf_path in ingested_docs:
            target_doc_id = pdf_path
        else:
            pdf_basename = os.path.basename(pdf_path)
            for doc in ingested_docs:
                if doc.endswith(pdf_basename):
                    target_doc_id = doc
                    break

        if not target_doc_id:
            continue

        with open(qa_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    query = item["question"]
                    relevant_ids = {target_doc_id}

                    # Classify complexity
                    ling, sem, mod, ctx = complexity_classifier.classify_complexity_4d(
                        query
                    )
                    is_complex = (
                        "High" in [ling, sem, mod, ctx]
                        or sum(1 for x in [ling, sem, mod, ctx] if x == "Medium") >= 2
                    )
                    complexity_label = "Complex" if is_complex else "Simple"

                    for sys_name in systems:
                        try:
                            sys_func = registry.get_system(sys_name)
                            system_result = sys_func(query, top_k=top_k)
                            results = system_result["results"]
                            answer = system_result["answer"]
                            retrieved_ids = get_retrieved_doc_ids(results)

                            ndcg_k = ndcg_at_k(retrieved_ids, relevant_ids, k=top_k)
                            recall_k = recall_at_k(retrieved_ids, relevant_ids, k=top_k)

                            all_results.append(
                                {
                                    "System": sys_name,
                                    "Complexity": complexity_label,
                                    f"NDCG@{top_k}": ndcg_k,
                                    f"Recall@{top_k}": recall_k,
                                    "Query": query[:50] + "...",
                                    "Answer": answer,
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error running {sys_name}: {e}")
                except Exception:
                    continue

    if not all_results:
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    # Summary
    print("\n=== Ablation Study Summary ===")
    summary = df.groupby(["System", "Complexity"])[["NDCG@10", "Recall@10"]].mean()
    print(summary)

    overall = df.groupby("System")[["NDCG@10", "Recall@10"]].mean()
    print("\n=== Overall Comparison ===")
    print(overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_bench_dir", default="data/raw/DocBench")
    args = parser.parse_args()
    run_ablation_study(args.doc_bench_dir)
