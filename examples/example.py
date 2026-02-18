from locisimiles.document import Document
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.pipeline import (
    ClassificationPipelineWithCandidategeneration,
    pretty_print,
)

# Load example query and source documents
query_doc = Document("./hieronymus_samples.csv", author="Hieronymus")
source_doc = Document("./vergil_samples.csv", author="Vergil")

print("Loaded query and source documents:")
print(f"Query Document: {query_doc}")
print(f"Source Document: {source_doc}")
print("=" * 70)


# Load the pipeline with pre-trained models
pipeline_two_stage = ClassificationPipelineWithCandidategeneration(
    classification_name="julian-schelb/PhilBerta-class-latin-intertext-v1",
    embedding_model_name="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
    device="mps",
)

# Run the pipeline with the query and source documents
results_two_stage = pipeline_two_stage.run(
    query=query_doc,  # Query document
    source=source_doc,  # Source document
    top_k=10,  # Number of top similar candidates to classify
)
print("\nResults of the two-stage pipeline run:")
pretty_print(results_two_stage)

evaluator = IntertextEvaluator(
    query_doc=query_doc,
    source_doc=source_doc,
    ground_truth_csv="./ground_truth.csv",
    pipeline=pipeline_two_stage,
    top_k=10,
    threshold=0.5,
)

print("\nSingle sentence:\n", evaluator.evaluate_single_query("hier. adv. iovin. 1.41"))
print("\nPer-sentence head:\n", evaluator.evaluate_all_queries().head(20))
print("\nMacro scores:\n", evaluator.evaluate(average="macro", with_match_only=True))
print("\nMicro scores:\n", evaluator.evaluate(average="micro", with_match_only=True))
