"""
Learned replacement for the hand-tuned alpha_intent lookup table in
configs/alpha_weights.py.

Trains one binary logistic regression per modality (vector/keyword/graph)
predicting P(this modality's top-k results contain the gold document | query
features), where features are the 4D complexity levels and the intent
one-hot encoding already computed by QueryComplexityClassifier and
QueryIntentClassifier. The predicted probabilities are used directly as
alpha_intent-style base weights in QALFFusion's adaptive weight formula,
replacing the fixed per-intent constants with a query-adaptive, data-driven
estimate -- this is the "learned" component the ICTAI reviewers pointed out
was missing despite the name "Query-Adaptive Learned Fusion".
"""

from typing import Dict, List, Tuple
import json
import logging
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

MODALITIES = ["vector", "keyword", "graph"]
INTENTS = [
    "factual_lookup",
    "relationship",
    "comparative",
    "temporal",
    "causal",
    "definitional",
    "visual_tabular",
    "multi_hop",
]
LEVEL_TO_NUM = {"Low": 1, "Medium": 2, "High": 3}


def build_feature_vector(
    complexity_4d: Tuple[str, str, str, str], intent: str
) -> np.ndarray:
    """
    Feature vector: [ling, sem, mod, ctx] (each 1-3) + one-hot intent (8-dim) = 12 features.
    """
    ling, sem, mod, ctx = complexity_4d
    numeric = [
        LEVEL_TO_NUM.get(ling, 2),
        LEVEL_TO_NUM.get(sem, 2),
        LEVEL_TO_NUM.get(mod, 2),
        LEVEL_TO_NUM.get(ctx, 2),
    ]
    intent_onehot = [1.0 if intent == i else 0.0 for i in INTENTS]
    return np.array(numeric + intent_onehot, dtype=float)


class LearnedAlphaWeights:
    """
    Drop-in replacement for configs.alpha_weights.get_alpha_weights(intent),
    but conditioned on the full (complexity, intent) pair and fit from data
    rather than hand-tuned.
    """

    def __init__(self):
        self.models: Dict[str, LogisticRegression] = {}
        self.fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
    ) -> None:
        """
        Args:
            features: (n_samples, n_features) array from build_feature_vector.
            labels: dict mapping modality -> binary (n_samples,) array,
                1 if that modality's top-k retrieval contained the gold
                document for that query, else 0.
        """
        for modality in MODALITIES:
            y = labels[modality]
            if len(np.unique(y)) < 2:
                # Degenerate case (e.g., a modality always/never succeeds in
                # the training sample): fall back to a constant-probability
                # model rather than letting sklearn error out.
                logger.warning(
                    f"Modality '{modality}' has a single class in training "
                    f"labels (all {y[0] if len(y) else 'N/A'}); using constant weight."
                )
                model = _ConstantModel(float(y.mean()) if len(y) else 0.4)
            else:
                model = LogisticRegression(max_iter=1000, class_weight="balanced")
                model.fit(features, y)
            self.models[modality] = model
        self.fitted = True

    def predict_alpha(
        self, complexity_4d: Tuple[str, str, str, str], intent: str
    ) -> Dict[str, float]:
        """Returns predicted alpha weight per modality, same shape as get_alpha_weights()."""
        if not self.fitted:
            raise RuntimeError("LearnedAlphaWeights.fit() must be called before predict_alpha().")
        x = build_feature_vector(complexity_4d, intent).reshape(1, -1)
        weights = {}
        for modality in MODALITIES:
            model = self.models[modality]
            prob = model.predict_proba(x)[0, 1] if hasattr(model, "predict_proba") else model.predict(x)
            weights[modality] = float(prob)
        return weights

    def save(self, path: str) -> None:
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"models": self.models, "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: str) -> "LearnedAlphaWeights":
        import joblib

        state = joblib.load(path)
        obj = cls()
        obj.models = state["models"]
        obj.fitted = state["fitted"]
        return obj


class _ConstantModel:
    """Fallback for a modality whose training labels are all one class."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile([1 - self.prob, self.prob], (n, 1))
