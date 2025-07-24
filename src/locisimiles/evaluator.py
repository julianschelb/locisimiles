# evaluator.py
from __future__ import annotations
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from locisimiles.document import Document
from locisimiles.pipeline import (
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
    
def _smr(tp: int, fp: int, fn: int, tn: int) -> float:
    """SMR (Source Match Rate) — proportion of all pairs that are true positives."""
    total = tp + fp + fn + tn
    return (fp + fn) / total

def _fp_rate(tp: int, fp: int, fn: int, tn: int) -> float:
    """FP / total — proportion of all pairs that are false positives."""
    total = tp + fp + fn + tn
    return fp / total

def _fn_rate(tp: int, fp: int, fn: int, tn: int) -> float:
    """FN / total — proportion of all pairs that are false negatives."""
    total = tp + fp + fn + tn
    return fn / total


# ────────────────────────────────
# Evaluation Pipeline
# ────────────────────────────────

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
        self.gold_labels = self._load_gold_labels(ground_truth_csv)

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

    # ─────────── PUBLIC: EVALUATION ───────────
    
    def evaluate_single_query(self, query_id: str) -> Dict[str, float]:
        """Compute metrics for one query sentence."""
        source_ids       = self.source_doc.ids()
        predicted_links  = self._predicted_link_set()

        gold_vec = np.array(
            [self.gold_labels.get((query_id, s_id), 0) for s_id in source_ids],
            dtype=int,
        )
        pred_vec = np.array(
            [1 if (query_id, s_id) in predicted_links else 0 for s_id in source_ids],
            dtype=int,
        )

        # Calculate confusion matrix components
        tp = int(((gold_vec == 1) & (pred_vec == 1)).sum())
        fp = int(((gold_vec == 0) & (pred_vec == 1)).sum())
        fn = int(((gold_vec == 1) & (pred_vec == 0)).sum())
        tn = int(((gold_vec == 0) & (pred_vec == 0)).sum())

        # Calculate metrics
        precision  = _precision(tp, fp)
        recall     = _recall(tp, fn)
        f1         = _f1(precision, recall)
        accuracy   = (tp + tn) / len(source_ids) if source_ids else 0.0
        total_errs = fp + fn
        smr        = _smr(tp, fp, fn, tn)
        fp_rate    = _fp_rate(tp, fp, fn, tn)
        fn_rate    = _fn_rate(tp, fp, fn, tn)

        # cache confusion matrix for this query
        self._conf_matrix_cache[query_id] = (tp, fp, fn, tn)

        return {
            "query_id":  query_id,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "accuracy":  accuracy,
            "errors":    total_errs,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fpr": fp_rate,
            "fnr": fn_rate,
            "smr": smr,
        }

    def query_ids_with_match(self) -> List[str]:
        """Return query IDs that have ground truth labels."""
        return list({q_id for q_id, _ in self.gold_labels.keys()})

    # ALL QUERIES EVALUATION
    def evaluate_all_queries(self, with_match_only: bool = False) -> pd.DataFrame:
        """Compute metrics for every query sentence (cached)."""
        # if self._per_sentence_df is not None:
        #     return self._per_sentence_df.copy()
        
        # Ignore queries without ground truth labels if requested
        query_ids = self.query_ids_with_match() if with_match_only else self.query_doc.ids()

        # Evaluate each query sentence
        records = [self.evaluate_single_query(q_id) for q_id in query_ids]
        self._per_sentence_df = pd.DataFrame(records)
        return self._per_sentence_df.copy()

    # EVALUATE AND REPORT METRICS
    def evaluate(self, *, average: str = "macro", with_match_only: bool = False) -> Dict[str, float]:
        """ Compute aggregated metrics across all queries. """
        df = self.evaluate_all_queries(with_match_only)

        # global sums (needed for both branches)
        tp_sum = int(df["tp"].sum())
        fp_sum = int(df["fp"].sum())
        fn_sum = int(df["fn"].sum())
        tn_sum = int(df["tn"].sum())
        total  = tp_sum + fp_sum + fn_sum + tn_sum

        base_cols = ["precision", "recall", "f1", "accuracy",
                    "fpr", "fnr", "smr"]

        # ────────── MACRO (uniform) ──────────
        if average == "macro":
            out = {m: float(df[m].mean()) for m in base_cols}
            out.update({"tp": tp_sum, "fp": fp_sum, "fn": fn_sum, "tn": tn_sum})
            aggregated_metrics = out

        # ────────── MICRO (pooled) ───────────
        if average == "micro":
            precision = _precision(tp_sum, fp_sum)
            recall    = _recall(tp_sum, fn_sum)
            f1        = _f1(precision, recall)
            accuracy  = (tp_sum + tn_sum) / total if total else 0.0

            fpr = fp_sum / total if total else 0.0
            fnr = fn_sum / total if total else 0.0
            smr = (fp_sum + fn_sum) / total if total else 0.0

            aggregated_metrics = {
                "precision": precision, "recall": recall, "f1": f1,
                "accuracy": accuracy,   "fpr": fpr,       "fnr": fnr,
                "smr": smr,             "tp": tp_sum,     "fp": fp_sum,
                "fn": fn_sum,           "tn": tn_sum,
            }
        
        if average not in ["macro", "micro"]:
            raise ValueError("average must be 'macro' or 'micro'")
        
        return pd.DataFrame([aggregated_metrics]).reset_index(drop=True)

    def confusion_matrix(self, query_id: str) -> np.ndarray:
        """Return 2x2 confusion matrix [[TP,FP],[FN,TN]] for one query sentence."""
        if query_id not in self._conf_matrix_cache:
            self.evaluate_single_query(query_id)  # populate if missing
        tp, fp, fn, tn = self._conf_matrix_cache[query_id]
        return np.array([[tp, fp], [fn, tn]], dtype=int)

    # ─────────── INTERNAL HELPERS ───────────
    
    def _load_gold_labels(self, ground_truth_csv: Union[str, pd.DataFrame]) -> Dict[Tuple[str, str], int]:
        """ Load ground truth labels from a CSV file or DataFrame. """

        # If a DataFrame is provided, use it directly; otherwise, read from CSV
        if isinstance(ground_truth_csv, pd.DataFrame):
            gold_df = ground_truth_csv
        else:
            gold_df = pd.read_csv(ground_truth_csv)

        # Ensure required columns are present
        req_cols = {"query_id", "source_id", "label"}
        if req_cols - set(gold_df.columns):
            raise ValueError(f"ground-truth file must contain columns {req_cols}")
        
        # Create a dictionary mapping (query_id, source_id) to label
        return {
            (row.query_id, row.source_id): int(row.label)
            for row in gold_df.itertuples(index=False)
        }
    
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
 
    print("\nMacro scores\n\n")
    macro_df = evaluator.evaluate(average="macro")
    print("\n", macro_df.to_string(index=False))

    print("\nMicro scores\n\n")
    micro_df = evaluator.evaluate(average="micro")
    print(micro_df.to_string(index=False))