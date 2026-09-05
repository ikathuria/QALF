"""
Trains src.qalf.learned_alpha.LearnedAlphaWeights from a DocBench-style
corpus (a directory of subdirs, each with one PDF and one *_qa.jsonl).

For each query, runs vector_only / keyword_only / graph_only retrieval and
labels each modality 1 if the gold document appears in that modality's
top-k results, else 0. Complexity (4D) + intent become the feature vector
(see src/qalf/learned_alpha.py:build_feature_vector). Fits one logistic
regression per modality on these (features, label) pairs and saves the
result to data/results/learned_alpha_weights.joblib, where SystemRegistry
picks it up automatically for the "qalf_learned" system.

Usage:
    python scripts/train_learned_alpha.py --doc_bench_dir data/raw/DocBench_download/data
    python scripts/train_learned_alpha.py --doc_bench_dir data/raw/RobotManuals --append
"""

import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.neo4j.neo4j_manager import Neo4jManager
from src.utils.systems import SystemRegistry, RetrievalSystem
import src.utils.constants as C
from src.qalf.query_complexity import QueryComplexityClassifier
from src.qalf.query_intent import QueryIntentClassifier
from src.qalf.learned_alpha import LearnedAlphaWeights, MODALITIES, build_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_path(path: str) -> str:
    return os.path.normpath(path)


def get_ingested_documents(neo4j_manager: Neo4jManager):
    query = "MATCH (d:Document) RETURN d.id as id"
    with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
        result = session.run(query)
        return {normalize_path(record["id"]) for record in result}


def get_retrieved_doc_ids(results):
    doc_ids = []
    for res in results:
        if "doc_id" in res and res["doc_id"]:
            doc_ids.append(normalize_path(res["doc_id"]))
        elif "source" in res:
            doc_ids.append(normalize_path(res["source"]))
        elif "metadata" in res and "source" in res["metadata"]:
            doc_ids.append(normalize_path(res["metadata"]["source"]))
    return doc_ids


def build_training_data(doc_bench_dir: str, top_k: int, limit: int = None):
    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI,
        username=C.NEO4J_USERNAME,
        password=C.NEO4J_PASSWORD,
        database=C.NEO4J_DB,
    )
    registry = SystemRegistry(neo4j_manager)
    complexity_classifier = QueryComplexityClassifier()
    intent_classifier = QueryIntentClassifier()
    ingested_docs = get_ingested_documents(neo4j_manager)

    if not ingested_docs:
        logger.error("No documents found in Neo4j -- ingest the corpus first.")
        neo4j_manager.close()
        return None, None

    subdirs = [d for d in glob.glob(os.path.join(doc_bench_dir, "*")) if os.path.isdir(d)]
    if limit:
        subdirs = subdirs[:limit]

    features_list = []
    labels_list = {m: [] for m in MODALITIES}
    modality_systems = {
        "vector": "vector_only",
        "keyword": "keyword_only",
        "graph": "graph_only",
    }

    for subdir in subdirs:
        pdf_files = glob.glob(os.path.join(subdir, "*.pdf"))
        qa_files = glob.glob(os.path.join(subdir, "*_qa.jsonl"))
        if not pdf_files or not qa_files:
            continue

        pdf_path = normalize_path(pdf_files[0])
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
            logger.warning(f"Skipping {subdir}: PDF not found in ingested documents.")
            continue

        with open(qa_files[0], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    query = item["question"]
                except Exception:
                    continue

                complexity_4d = complexity_classifier.classify_complexity_4d(query)
                intent = intent_classifier.classify(query)
                x = build_feature_vector(complexity_4d, intent)
                features_list.append(x)

                for modality, sys_name in modality_systems.items():
                    try:
                        sys_func: RetrievalSystem = registry.get_system(sys_name)
                        result = sys_func(query, top_k=top_k)
                        retrieved_ids = get_retrieved_doc_ids(result["results"])
                        hit = 1 if target_doc_id in retrieved_ids else 0
                    except Exception as e:
                        logger.error(f"Error running {sys_name} for query in {subdir}: {e}")
                        hit = 0
                    labels_list[modality].append(hit)

    neo4j_manager.close()

    if not features_list:
        return None, None

    X = np.vstack(features_list)
    y = {m: np.array(labels_list[m]) for m in MODALITIES}
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_bench_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output_path", default="data/results/learned_alpha_weights.joblib"
    )
    args = parser.parse_args()

    X, y = build_training_data(args.doc_bench_dir, args.top_k, args.limit)
    if X is None:
        logger.error("No training data collected; aborting.")
        return

    logger.info(f"Collected {X.shape[0]} training examples.")
    for m in MODALITIES:
        logger.info(f"  {m}: {y[m].sum()} positive / {len(y[m])} total")

    model = LearnedAlphaWeights()
    model.fit(X, y)
    model.save(args.output_path)
    logger.info(f"Saved trained learned-alpha model to {args.output_path}")


if __name__ == "__main__":
    main()
