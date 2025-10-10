"""Main Gradio application for Loci Similes Demo."""

from __future__ import annotations

import sys
from typing import Any

try:  # gradio is an optional dependency
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard
    missing = getattr(exc, "name", None)
    base_msg = (
        "Optional GUI dependencies are missing. Install them via "
        "'pip install locisimiles[gui]' (Python 3.13+ also requires the "
        "audioop-lts backport) to use the Gradio interface."
    )
    if missing and missing != "gradio":
        raise ImportError(f"{base_msg} (missing package: {missing})") from exc
    raise ImportError(base_msg) from exc

from .upload_page import build_upload_page
from .results_page import build_results_page, update_results_display
from .utils import validate_csv

# Import Loci Similes components
from locisimiles.pipeline import ClassificationPipelineWithCandidategeneration
from locisimiles.document import Document


def _show_processing_status() -> dict:
    """Show the processing spinner."""
    spinner_html = """
    <div style="display: flex; align-items: center; justify-content: center; padding: 20px; background-color: #e3f2fd; border-radius: 8px; margin: 20px 0;">
        <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #2196F3; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
            <div style="font-size: 16px; color: #1976D2; font-weight: 500;">Processing documents... This may take several minutes on first run.</div>
            <div style="font-size: 13px; color: #666;">Downloading models, generating embeddings, and classifying candidates...</div>
        </div>
    </div>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """
    return gr.update(value=spinner_html, visible=True)


def _process_documents(
    query_file: str | None,
    source_file: str | None,
    classification_model: str,
    embedding_model: str,
    top_k: int,
    threshold: float,
) -> tuple:
    """Process the documents using the Loci Similes pipeline and navigate to results page.
    
    Args:
        query_file: Path to query CSV file
        source_file: Path to source CSV file
        classification_model: Name of the classification model
        embedding_model: Name of the embedding model
        top_k: Number of top candidates to retrieve
        threshold: Similarity threshold (not used in pipeline, for future filtering)
    
    Returns:
        Tuple of (processing_status_update, upload_page_visibility, results_page_visibility)
    """
    if not query_file or not source_file:
        gr.Warning("Both query and source documents must be uploaded before processing.")
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    # Validate both files
    query_valid, query_msg = validate_csv(query_file)
    source_valid, source_msg = validate_csv(source_file)
    
    if not query_valid or not source_valid:
        gr.Warning("Please ensure both documents are valid before processing.")
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    try:
        # Detect device (prefer GPU if available)
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        # Initialize pipeline
        # Note: First run will download models (~500MB each), subsequent runs use cached models
        pipeline = ClassificationPipelineWithCandidategeneration(
            classification_name=classification_model,
            embedding_model_name=embedding_model,
            device=device,
        )
        
        # Load documents
        query_doc = Document(query_file)
        source_doc = Document(source_file)
        
        # Run pipeline
        results = pipeline.run(
            query=query_doc,
            source=source_doc,
            top_k=top_k,
        )
        
        # Store results
        num_queries = len(results)
        total_matches = sum(len(matches) for matches in results.values())
        
        print(f"Processing complete! Found matches for {num_queries} query segments ({total_matches} total matches).")
        
        # Return results and navigate to results page
        return (
            gr.update(visible=False),  # Hide processing status
            gr.update(visible=False),  # Hide upload page
            gr.update(visible=True),   # Show results page
            results,                   # Store results in state
            query_doc,                 # Store query doc in state
        )
        
    except Exception as e:
        print(f"Processing error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return (
            gr.update(visible=False),  # Hide processing status
            gr.update(visible=True),   # Show upload page
            gr.update(visible=False),  # Hide results page
            None,                      # No results
            None,                      # No query doc
        )


def build_interface() -> gr.Blocks:
    """Create the main Gradio Blocks interface."""
    with gr.Blocks(title="Loci Similes Demo") as demo:
        # State to store pipeline results
        results_state = gr.State(value=None)
        query_doc_state = gr.State(value=None)
        
        # Build both pages
        upload_page, upload_components = build_upload_page()
        results_page, results_components = build_results_page()

        # Process button navigation with all configuration parameters
        # First show the spinner
        upload_components["process_btn"].click(
            fn=_show_processing_status,
            inputs=None,
            outputs=upload_components["processing_status"],
        ).then(
            # Then process the documents
            fn=_process_documents,
            inputs=[
                upload_components["query_upload"],
                upload_components["source_upload"],
                upload_components["classification_model"],
                upload_components["embedding_model"],
                upload_components["top_k"],
                upload_components["threshold"],
            ],
            outputs=[
                upload_components["processing_status"],
                upload_page,
                results_page,
                results_state,
                query_doc_state,
            ],
        ).then(
            # Update results display with actual data
            fn=update_results_display,
            inputs=[results_state, query_doc_state],
            outputs=[
                results_components["query_segments"],
                results_components["query_segments_state"],
                results_components["matches_dict_state"],
            ],
        )

        # Back button navigation
        results_components["back_btn"].click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=None,
            outputs=[upload_page, results_page],
        )
    return demo


def launch(**kwargs: Any) -> None:
    """Launch the Gradio app."""
    demo = build_interface()
    kwargs.setdefault("show_api", False)
    kwargs.setdefault("inbrowser", False)
    kwargs.setdefault("quiet", True)
    try:
        demo.launch(share=False, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "shareable link must be created" in msg:
            print(
                "⚠️  Unable to open the Gradio UI because localhost is blocked "
                "in this environment. Exiting without starting the server.",
                file=sys.stderr,
            )
            return
        raise


def main() -> None:
    """Entry point for the ``locisimiles-gui`` console script."""
    launch()


if __name__ == "__main__":  # pragma: no cover
    main()
