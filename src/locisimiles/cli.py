"""Command-line interface for Loci Similes pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from locisimiles.document import Document
from locisimiles.pipeline import ClassificationPipelineWithCandidategeneration


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
    --classification-model julian-schelb/PhilBerta-class-latin-intertext-v1 \\
    --embedding-model julian-schelb/SPhilBerta-emb-lat-intertext-v1 \\
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
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to output CSV file for results",
    )
    
    # Model selection
    parser.add_argument(
        "--classification-model",
        type=str,
        default="julian-schelb/PhilBerta-class-latin-intertext-v1",
        help="HuggingFace model name for classification (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
        help="HuggingFace model name for embeddings (default: %(default)s)",
    )
    
    # Pipeline parameters
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to retrieve per query segment (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
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
        "-v", "--verbose",
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
            print(f"  Classification model: {args.classification_model}")
            print(f"  Embedding model: {args.embedding_model}")
        
        pipeline = ClassificationPipelineWithCandidategeneration(
            classification_name=args.classification_model,
            embedding_model_name=args.embedding_model,
            device=device,
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
            sum(1 for _, _, prob in matches if prob >= args.threshold)
            for matches in results.values()
        )
        
        if args.verbose:
            print(f"Processing complete!")
            print(f"  Query segments with matches: {num_queries}")
            print(f"  Total candidate matches: {total_matches}")
            print(f"  Matches above threshold ({args.threshold}): {above_threshold}")
        
        # Write results to CSV
        if args.verbose:
            print(f"Writing results to {args.output}...")
        
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'query_id',
                'query_text',
                'source_id',
                'source_text',
                'similarity',
                'probability',
                'above_threshold'
            ])
            
            for query_segment in query_doc:
                query_id = query_segment.id
                query_text = query_segment.text
                matches = results.get(query_id, [])
                
                if matches:
                    for source_segment, similarity, probability in matches:
                        source_id = source_segment.id
                        source_text = source_segment.text
                        above_threshold_flag = "Yes" if probability >= args.threshold else "No"
                        
                        writer.writerow([
                            query_id,
                            query_text,
                            source_id,
                            source_text,
                            f"{similarity:.6f}",
                            f"{probability:.6f}",
                            above_threshold_flag
                        ])
                else:
                    # Write row even if no matches
                    writer.writerow([
                        query_id,
                        query_text,
                        "",
                        "",
                        "",
                        "",
                        ""
                    ])
        
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
