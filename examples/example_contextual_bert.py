from pathlib import Path

from locisimiles.document import Document
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.pipeline import LatinBertRetrievalPipeline, pretty_print

base = Path(__file__).resolve().parent
query_csv = base / "hieronymus_samples.csv"
source_csv = base / "vergil_samples.csv"
ground_truth_csv = base / "ground_truth.csv"
local_latinbert_path = base / "models" / "latinbert"
latinbert_model_name = "ashleygong03/bamman-burns-latin-bert"

query_doc = Document(query_csv, author="Hieronymus")
source_doc = Document(source_csv, author="Vergil")

print("Loaded query and source documents:")
print(f"Query segments: {len(query_doc)}")
print(f"Source segments: {len(source_doc)}")
print("=" * 70)

# Prefer a local LatinBERT directory when available, otherwise download from HF.
pipeline_kwargs = {
    "top_k": 10,
    "similarity_threshold": 0.85,
    "max_length": 256,
    "min_token_length": 2,
    "use_stopword_filter": True,
}
if local_latinbert_path.exists():
    pipeline_kwargs["model_path"] = local_latinbert_path
    print(f"Using local LatinBERT model path: {local_latinbert_path}")
else:
    pipeline_kwargs["model_name"] = latinbert_model_name
    print(f"Using HuggingFace LatinBERT model: {latinbert_model_name}")

pipeline = LatinBertRetrievalPipeline(**pipeline_kwargs)

results = pipeline.run(query=query_doc, source=source_doc, top_k=10)

print("\nResults of the contextual retrieval pipeline run:")
pretty_print(results)

evaluator = IntertextEvaluator(
    query_doc=query_doc,
    source_doc=source_doc,
    ground_truth_csv=ground_truth_csv,
    pipeline=pipeline,
    top_k=10,
    threshold=0.85,
)

print("\nMacro scores:\n", evaluator.evaluate(average="macro", with_match_only=True))
print("\nMicro scores:\n", evaluator.evaluate(average="micro", with_match_only=True))

out_csv = base / "results_contextual_bert.csv"
out_json = base / "results_contextual_bert.json"
pipeline.to_csv(out_csv, results=results)
pipeline.to_json(out_json, results=results)

print(f"\nSaved CSV to: {out_csv}")
print(f"Saved JSON to: {out_json}")
