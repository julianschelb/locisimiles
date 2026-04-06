import csv
import re
from pathlib import Path

from gensim.models import Word2Vec

from locisimiles.document import Document
from locisimiles.pipeline import Word2VecRetrievalPipeline


def normalize_tokens(text: str) -> list[str]:
    text = text.lower().replace("j", "i").replace("v", "u")
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split(" ") if token]


base = Path("examples")
q_csv = base / "hieronymus_samples.csv"
s_csv = base / "vergil_samples.csv"

sentences: list[list[str]] = []
for path in (q_csv, s_csv):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tokens = normalize_tokens(row.get("text", ""))
            if tokens:
                sentences.append(tokens)

models_dir = base / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "generic_latin_word2vec.model"

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    sg=1,
    epochs=20,
    seed=42,
)
model.save(str(model_path))

query_doc = Document(q_csv, author="Hieronymus")
source_doc = Document(s_csv, author="Vergil")

pipeline = Word2VecRetrievalPipeline(
    model_path=model_path,
    top_k=10,
    similarity_threshold=0.85,
    interval=2,
    order_free=True,
)

results = pipeline.run(query=query_doc, source=source_doc, top_k=10)
num_queries = len(results)
total_candidates = sum(len(items) for items in results.values())
num_positive = sum(
    sum(1 for judgment in items if judgment.judgment_score >= 0.85)
    for items in results.values()
)

out_csv = base / "results_word2vec_generic.csv"
out_json = base / "results_word2vec_generic.json"
pipeline.to_csv(out_csv, results=results)
pipeline.to_json(out_json, results=results)

print(f"MODEL={model_path}")
print(f"QUERIES={num_queries}")
print(f"CANDIDATES={total_candidates}")
print(f"ABOVE_THRESHOLD={num_positive}")
print(f"CSV={out_csv}")
print(f"JSON={out_json}")
