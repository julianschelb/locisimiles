# evaluator.py
from __future__ import annotations

"""
IntertextEvaluator
==================
Evaluate intertextual-link predictions produced by
ClassificationPipelineWithCandidategeneration.

Public API
----------
evaluate_single_query(query_id)          → dict of metrics for that sentence
evaluate_all_queries()                   → DataFrame (row per query sentence)
evaluate(average="macro"|"micro")        → document-level metrics
confusion_matrix(query_id)               → 2×2 numpy array [[TP,FP],[FN,TN]]
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from document import Document
from pipeline import (
    ClassificationPipelineWithCandidategeneration,
    FullDict,  # alias exported by pipeline.py
)

# ────────────────────────────────
# Metric helpers (scalar, no deps)
# ────────────────────────────────


def _precision(tp: int, fp: int) -> float: return tp / \
    (tp + fp) if tp + fp else 0.0


def _recall(tp: int, fn: int) -> float: return tp / \
    (tp + fn) if tp + fn else 0.0


def _f1(p: float, r: float) -> float: return 2 * \
    p * r / (p + r) if p + r else 0.0


class IntertextEvaluator:
    """Compute sentence- and document-level scores for intertextual link prediction."""

    # ─────────── CONSTRUCTOR ───────────
    def __init__(
        self,
        *,
        query_doc: Document,
        source_doc: Document,
        ground_truth_csv: str | pd.DataFrame,
        pipeline: ClassificationPipelineWithCandidategeneration,
        top_k: int = 5,
        threshold: float = 0.5,
    ):
        # Persist inputs
        self.query_doc = query_doc
        self.source_doc = source_doc
        self.pipeline = pipeline
        self.top_k = top_k
        self.threshold = threshold

        # 1) LOAD GOLD LABELS ────────────────────────────────────────────
        gold_df = ground_truth_csv if isinstance(ground_truth_csv, pd.DataFrame) \
            else pd.read_csv(ground_truth_csv)
        req_cols = {"query_id", "source_id", "label"}
        if req_cols - set(gold_df.columns):
            raise ValueError(
                f"ground-truth file must contain columns {req_cols}")
        self.gold_labels: Dict[Tuple[str, str], int] = {
            # type: ignore[attr-defined]
            (row.query_id, row.source_id): int(row.label)
            for row in gold_df.itertuples(index=False)
        }

        # 2) RUN PIPELINE ONCE ──────────────────────────────────────────
        self.predictions: FullDict = pipeline.run(
            query=query_doc,
            source=source_doc,
            top_k=top_k,
        )

        # Inform user if top_k < |D_s|
        self.num_source_sentences = len(self.source_doc.ids())
        if top_k < self.num_source_sentences:
            print(
                f"[IntertextEvaluator] top_k={top_k} < {self.num_source_sentences} "
                "source sentences → pairs not returned by the pipeline "
                "will be treated as negatives."
            )

        # Internal caches (populated lazily)
        self._per_sentence_df: pd.DataFrame | None = None
        self._conf_matrix_cache: Dict[str, Tuple[int, int, int, int]] = {}

    # ─────────── PUBLIC LEVEL A: SINGLE SENTENCE ───────────
    def evaluate_single_query(self, query_id: str) -> Dict[str, float]:
        """Return metrics for one query sentence."""
        df = self.evaluate_all_queries()  # ensures dataframe is built
        row = df.set_index("query_id").loc[query_id]
        return row.to_dict()  # type: ignore[return-value]

    # ─────────── PUBLIC LEVEL B: ALL SENTENCES ────────────
    def evaluate_all_queries(self) -> pd.DataFrame:
        """Compute metrics for every query sentence (cached)."""
        if self._per_sentence_df is not None:
            return self._per_sentence_df.copy()

        source_ids = self.source_doc.ids()
        predicted_links = self._predicted_link_set()

        records: List[Dict[str, float | int | str]] = []
        for q_id in self.query_doc.ids():
            # Gold / predicted vectors for this query sentence
            gold_vec = np.array(
                [self.gold_labels.get((q_id, s_id), 0) for s_id in source_ids],
                dtype=int,
            )
            pred_vec = np.array(
                [1 if (q_id, s_id) in predicted_links else 0 for s_id in source_ids],
                dtype=int,
            )

            tp = int(((gold_vec == 1) & (pred_vec == 1)).sum())
            fp = int(((gold_vec == 0) & (pred_vec == 1)).sum())
            fn = int(((gold_vec == 1) & (pred_vec == 0)).sum())
            tn = int(((gold_vec == 0) & (pred_vec == 0)).sum())

            precision = _precision(tp, fp)
            recall = _recall(tp, fn)
            f1 = _f1(precision, recall)
            accuracy = (tp + tn) / len(source_ids) if source_ids else 0.0

            records.append({
                "query_id": q_id,
                "precision": precision,
                "recall":    recall,
                "f1":        f1,
                "accuracy":  accuracy,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
            self._conf_matrix_cache[q_id] = (tp, fp, fn, tn)

        self._per_sentence_df = pd.DataFrame(records)
        return self._per_sentence_df.copy()

    # ─────────── PUBLIC LEVEL C: DOCUMENT ────────────
    def evaluate(self, *, average: str = "macro") -> Dict[str, float]:
        """
        Document-level metrics.
        average="macro" → mean over sentence scores  
        average="micro" → aggregate TP/FP/FN/TN then derive metrics
        """
        df = self.evaluate_all_queries()

        if average == "macro":
            return {m: float(df[m].mean()) for m in ["precision", "recall", "f1", "accuracy"]}

        if average == "micro":
            tp, fp = int(df["tp"].sum()), int(df["fp"].sum())
            fn, tn = int(df["fn"].sum()), int(df["tn"].sum())
            precision = _precision(tp, fp)
            recall = _recall(tp, fn)
            f1 = _f1(precision, recall)
            accuracy = (tp + tn) / (tp + fp + fn +
                                    tn) if (tp + fp + fn + tn) else 0.0
            return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

        raise ValueError("average must be 'macro' or 'micro'")

    def confusion_matrix(self, query_id: str) -> np.ndarray:
        """Return 2×2 confusion matrix [[TP,FP],[FN,TN]] for one query sentence."""
        if query_id not in self._conf_matrix_cache:
            self.evaluate_single_query(query_id)  # populate if missing
        tp, fp, fn, tn = self._conf_matrix_cache[query_id]
        return np.array([[tp, fp], [fn, tn]], dtype=int)

    # ─────────── INTERNAL HELPERS ───────────
    def _predicted_link_set(self) -> set[Tuple[str, str]]:
        """
        All (query_id, source_id) pairs predicted *positive*.

        - Only pairs returned by the pipeline are considered;  
          every other pair (i.e. omitted due to `top_k`) is implicitly negative.  
        - Within returned pairs, we keep those with P(pos) ≥ threshold.
        """
        link_set: set[Tuple[str, str]] = set()
        for q_id, result_list in self.predictions.items():
            link_set.update(
                {(q_id, seg.id)
                 for seg, _sim, prob in result_list if prob >= self.threshold}
            )
        return link_set


# ─────────── QUICK DEMO ───────────
if __name__ == "__main__":
    qdoc = Document("../data/vergil_samples.csv")
    sdoc = Document("../data/hieronymus_samples.csv")
    pipe = ClassificationPipelineWithCandidategeneration(device="cpu")

    evaluator = IntertextEvaluator(
        query_doc=qdoc,
        source_doc=sdoc,
        ground_truth_csv="../data/ground_truth_links.csv",
        pipeline=pipe,
        top_k=5,
        threshold=0.5,
    )

    print("Single sentence:\n", evaluator.evaluate_single_query("verg. ecl. 4.60"))
    print("\nPer-sentence head:\n", evaluator.evaluate_all_queries().head())
    print("\nMacro scores:", evaluator.evaluate(average="macro"))
    print("Micro scores:", evaluator.evaluate(average="micro"))
