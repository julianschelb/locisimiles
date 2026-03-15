"""Command-line interface for Loci Similes pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from locisimiles.document import Document
from locisimiles.pipeline import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    DEFAULT_WORD2VEC_MODEL_PATH,
    LatinBertRetrievalPipeline,
    LatinBertTwoStagePipeline,
    TwoStagePipeline,
    Word2VecRetrievalPipeline,
)


def main() -> int:
    """Main entry point for the locisimiles CLI."""
    parser = argparse.ArgumentParser(
        prog="locisimiles",
        description="Find intertextual references in Latin documents using pre-trained language models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default models
  locisimiles query.csv source.csv -o results.csv

  # Use custom models and parameters
  locisimiles query.csv source.csv -o results.csv \\
    --classification-model julian-schelb/xlm-roberta-large-class-lat-intertext-v1 \\
    --embedding-model julian-schelb/multilingual-e5-large-emb-lat-intertext-v1 \\
    --top-k 20 --threshold 0.7

  # Use GPU if available
  locisimiles query.csv source.csv -o results.csv --device cuda

CSV Format:
  Input files must have two columns: 'seg_id' and 'text'
  Output file contains: query_id, query_text, source_id, source_text,
                        similarity, probability, above_threshold
        """,
    )

    # Required arguments
    parser.add_argument(
        "query",
        type=Path,
        help="Path to query document CSV file (columns: seg_id, text)",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to source document CSV file (columns: seg_id, text)",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file for results",
    )

    # Model selection
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=[
            "two-stage",
            "word2vec-retrieval",
            "latin-bert-retrieval",
            "latin-bert-two-stage",
        ],
        default="two-stage",
        help="Pipeline type to run (default: %(default)s)",
    )
    parser.add_argument(
        "--classification-model",
        type=str,
        default="julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
        help="HuggingFace model name for classification (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="julian-schelb/multilingual-e5-large-emb-lat-intertext-v1",
        help="HuggingFace model name for embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--latin-bert-model",
        type=str,
        default=DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
        help="HuggingFace model name for contextual Latin BERT retrieval (default: %(default)s)",
    )
    parser.add_argument(
        "--latin-bert-model-path",
        type=Path,
        default=None,
        help=(
            "Optional local model path for contextual Latin BERT retrieval "
            "(used with --pipeline latin-bert-retrieval or latin-bert-two-stage)"
        ),
    )
    parser.add_argument(
        "--latin-bert-max-length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length for contextual retrieval (default: %(default)s)",
    )
    parser.add_argument(
        "--latin-bert-min-token-length",
        type=int,
        default=2,
        help="Minimum token length used for contextual scoring (default: %(default)s)",
    )
    parser.add_argument(
        "--latin-bert-disable-stopword-filter",
        action="store_true",
        help="Disable built-in Latin stopword filtering for contextual retrieval",
    )
    parser.add_argument(
        "--word2vec-model-path",
        type=Path,
        default=None,
        help=(
            "Path to local gensim Word2Vec .model file "
            "(used only with --pipeline word2vec-retrieval)"
        ),
    )
    parser.add_argument(
        "--word2vec-interval",
        type=int,
        default=0,
        help="Max token gap for Word2Vec bigrams (default: %(default)s)",
    )
    parser.add_argument(
        "--word2vec-order-free",
        action="store_true",
        help="Treat Word2Vec bigrams as order-insensitive",
    )

    # Pipeline parameters
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to retrieve per query segment (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.85,
        help="Classification probability threshold for filtering results (default: %(default)s)",
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for computation (default: auto-detect)",
    )

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.query.exists():
        print(f"Error: Query file not found: {args.query}", file=sys.stderr)
        return 1
    if not args.source.exists():
        print(f"Error: Source file not found: {args.source}", file=sys.stderr)
        return 1

    # Determine device
    if args.device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if args.verbose:
            print(f"Auto-detected device: {device}")
    else:
        device = args.device
        if args.verbose:
            print(f"Using device: {device}")

    try:
        # Load documents
        if args.verbose:
            print(f"Loading query document from {args.query}...")
        query_doc = Document(str(args.query))

        if args.verbose:
            print(f"Loading source document from {args.source}...")
        source_doc = Document(str(args.source))

        if args.verbose:
            print(f"Query segments: {len(query_doc)}")
            print(f"Source segments: {len(source_doc)}")

        # Initialize pipeline
        if args.verbose:
            print("Initializing pipeline...")
            print(f"  Pipeline type: {args.pipeline}")

        if args.pipeline in {"latin-bert-retrieval", "latin-bert-two-stage"}:
            if args.latin_bert_model_path is not None and (
                args.latin_bert_model != DEFAULT_CONTEXTUAL_BERT_MODEL_NAME
            ):
                print(
                    "Error: provide only one contextual model source: "
                    "--latin-bert-model-path or a custom --latin-bert-model.",
                    file=sys.stderr,
                )
                return 1

        if args.pipeline == "two-stage":
            if args.verbose:
                print(f"  Classification model: {args.classification_model}")
                print(f"  Embedding model: {args.embedding_model}")

            pipeline = TwoStagePipeline(
                classification_name=args.classification_model,
                embedding_model_name=args.embedding_model,
                device=device,
            )
        elif args.pipeline == "word2vec-retrieval":
            if args.verbose:
                if args.word2vec_model_path is None:
                    print("  Word2Vec model path: default package path")
                else:
                    print(f"  Word2Vec model path: {args.word2vec_model_path}")

            pipeline = Word2VecRetrievalPipeline(
                model_path=args.word2vec_model_path
                if args.word2vec_model_path is not None
                else DEFAULT_WORD2VEC_MODEL_PATH,
                top_k=args.top_k,
                similarity_threshold=args.threshold,
                interval=args.word2vec_interval,
                order_free=args.word2vec_order_free,
            )
        elif args.pipeline == "latin-bert-retrieval":
            if args.verbose:
                if args.latin_bert_model_path is None:
                    print(f"  Latin BERT model (HF): {args.latin_bert_model}")
                else:
                    print(f"  Latin BERT model (local): {args.latin_bert_model_path}")

            pipeline = LatinBertRetrievalPipeline(
                model_name=args.latin_bert_model,
                model_path=args.latin_bert_model_path,
                device=device,
                top_k=args.top_k,
                similarity_threshold=args.threshold,
                max_length=args.latin_bert_max_length,
                min_token_length=args.latin_bert_min_token_length,
                use_stopword_filter=not args.latin_bert_disable_stopword_filter,
            )
        else:
            if args.verbose:
                if args.latin_bert_model_path is None:
                    print(f"  Latin BERT model (HF): {args.latin_bert_model}")
                else:
                    print(f"  Latin BERT model (local): {args.latin_bert_model_path}")

            pipeline = LatinBertTwoStagePipeline(
                classification_name=args.classification_model,
                model_name=args.latin_bert_model,
                model_path=args.latin_bert_model_path,
                device=device,
                max_length=args.latin_bert_max_length,
                min_token_length=args.latin_bert_min_token_length,
                use_stopword_filter=not args.latin_bert_disable_stopword_filter,
            )

        # Run pipeline
        if args.verbose:
            print(f"Running pipeline (top-k={args.top_k})...")

        results = pipeline.run(
            query=query_doc,
            source=source_doc,
            top_k=args.top_k,
        )

        # Count results
        num_queries = len(results)
        total_matches = sum(len(matches) for matches in results.values())
        above_threshold = sum(
            sum(1 for j in matches if j.judgment_score >= args.threshold)
            for matches in results.values()
        )

        if args.verbose:
            print("Processing complete!")
            print(f"  Query segments with matches: {num_queries}")
            print(f"  Total candidate matches: {total_matches}")
            print(f"  Matches above threshold ({args.threshold}): {above_threshold}")

        # Write results to CSV
        if args.verbose:
            print(f"Writing results to {args.output}...")

        with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "query_id",
                    "query_text",
                    "source_id",
                    "source_text",
                    "similarity",
                    "probability",
                    "above_threshold",
                ]
            )

            for query_segment in query_doc:
                query_id = query_segment.id
                query_text = query_segment.text
                matches = results.get(query_id, [])

                if matches:
                    for judgment in matches:
                        source_id = judgment.segment.id
                        source_text = judgment.segment.text
                        similarity = judgment.candidate_score
                        probability = judgment.judgment_score
                        above_threshold_flag = "Yes" if probability >= args.threshold else "No"

                        sim_str = f"{similarity:.6f}" if similarity is not None else ""
                        writer.writerow(
                            [
                                query_id,
                                query_text,
                                source_id,
                                source_text,
                                sim_str,
                                f"{probability:.6f}",
                                above_threshold_flag,
                            ]
                        )
                else:
                    # Write row even if no matches
                    writer.writerow([query_id, query_text, "", "", "", "", ""])

        print(f"✅ Results saved to {args.output}")
        print(f"   Found {above_threshold} matches above threshold {args.threshold}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
