from pathlib import Path

from locisimiles.document import Document
from locisimiles.pipeline import LatinBertRetrievalPipeline


def main() -> int:
    base = Path(__file__).resolve().parent
    q_csv = base / "hieronymus_samples.csv"
    s_csv = base / "vergil_samples.csv"
    local_latinbert_path = base / "models" / "latinbert"
    latinbert_model_name = "ashleygong03/bamman-burns-latin-bert"

    query_doc = Document(q_csv, author="Hieronymus")
    source_doc = Document(s_csv, author="Vergil")

    pipeline_kwargs = {
        "top_k": 5,
        "similarity_threshold": 0.85,
        "max_length": 128,
        "min_token_length": 2,
        "use_stopword_filter": True,
    }
    if local_latinbert_path.exists():
        pipeline_kwargs["model_path"] = local_latinbert_path
    else:
        pipeline_kwargs["model_name"] = latinbert_model_name

    pipeline = LatinBertRetrievalPipeline(**pipeline_kwargs)

    results = pipeline.run(query=query_doc, source=source_doc, top_k=5)

    num_queries = len(results)
    total_candidates = sum(len(items) for items in results.values())
    positive = sum(
        sum(1 for item in items if item.judgment_score >= 0.85) for items in results.values()
    )

    out_csv = base / "results_contextual_bert_smoketest.csv"
    out_json = base / "results_contextual_bert_smoketest.json"
    pipeline.to_csv(out_csv, results=results)
    pipeline.to_json(out_json, results=results)

    print(f"QUERIES={num_queries}")
    print(f"CANDIDATES={total_candidates}")
    print(f"ABOVE_THRESHOLD={positive}")
    print(f"CSV={out_csv}")
    print(f"JSON={out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
