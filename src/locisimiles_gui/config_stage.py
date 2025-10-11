"""Configuration stage for the Loci Similes GUI."""

from __future__ import annotations

import sys

try:
    import gradio as gr
except ImportError as exc:
    missing = getattr(exc, "name", None)
    base_msg = (
        "Optional GUI dependencies are missing. Install them via "
        "'pip install locisimiles[gui]' (Python 3.13+ also requires the "
        "audioop-lts backport) to use the Gradio interface."
    )
    if missing and missing != "gradio":
        raise ImportError(f"{base_msg} (missing package: {missing})") from exc
    raise ImportError(base_msg) from exc

from .utils import validate_csv
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
    """Process the documents using the Loci Similes pipeline and navigate to results step.
    
    Args:
        query_file: Path to query CSV file
        source_file: Path to source CSV file
        classification_model: Name of the classification model
        embedding_model: Name of the embedding model
        top_k: Number of top candidates to retrieve
        threshold: Similarity threshold (not used in pipeline, for future filtering)
    
    Returns:
        Tuple of (processing_status_update, walkthrough_update, results_state, query_doc_state)
    """
    if not query_file or not source_file:
        gr.Warning("Both query and source documents must be uploaded before processing.")
        return gr.update(visible=False), gr.Walkthrough(selected=1), None, None
    
    # Validate both files
    query_valid, query_msg = validate_csv(query_file)
    source_valid, source_msg = validate_csv(source_file)
    
    if not query_valid or not source_valid:
        gr.Warning("Please ensure both documents are valid before processing.")
        return gr.update(visible=False), gr.Walkthrough(selected=1), None, None
    
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
        
        # Return results and navigate to results step (Step 3, id=2)
        return (
            gr.update(visible=False),   # Hide processing status
            gr.Walkthrough(selected=2), # Navigate to Results step
            results,                    # Store results in state
            query_doc,                  # Store query doc in state
        )
        
    except Exception as e:
        print(f"Processing error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        gr.Error(f"Processing failed: {str(e)}")
        return (
            gr.update(visible=False),   # Hide processing status
            gr.Walkthrough(selected=1), # Stay on Configuration step
            None,                       # No results
            None,                       # No query doc
        )


def build_config_stage() -> tuple[gr.Step, dict]:
    """Build the configuration stage UI.
    
    Returns:
        Tuple of (Step component, dict of components for external access)
    """
    components = {}
    
    with gr.Step("Pipeline Configuration", id=1) as step:
        gr.Markdown("### ⚙️ Step 2: Pipeline Configuration")
        gr.Markdown(
            "Configure the two-stage pipeline. Stage 1 (Embedding): Quickly ranks all source segments by similarity to each query segment. "
            "Stage 2 (Classification): Examines the top-K candidates more carefully to identify true intertextual references. "
            "Higher K values catch more potential citations but increase computation time. The threshold filters results by classification confidence."
        )
        
        with gr.Row():
            # Left column: Model Selection
            with gr.Column():
                gr.Markdown("**🤖 Model Selection**")
                components["classification_model"] = gr.Dropdown(
                    label="Classification Model",
                    choices=["julian-schelb/PhilBerta-class-latin-intertext-v1"],
                    value="julian-schelb/PhilBerta-class-latin-intertext-v1",
                    interactive=True,
                    info="Model used to classify candidate pairs as intertextual or not",
                )
                components["embedding_model"] = gr.Dropdown(
                    label="Embedding Model",
                    choices=["julian-schelb/SPhilBerta-emb-lat-intertext-v1"],
                    value="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
                    interactive=True,
                    info="Model used to generate embeddings for candidate retrieval",
                )
            
            # Right column: Retrieval Parameters
            with gr.Column():
                gr.Markdown("**🛠️ Retrieval Parameters**")
                components["top_k"] = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Top K Candidates",
                    info="How many candidates to examine per query. Higher values find more references but take longer to process.",
                )
                components["threshold"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Classification Threshold",
                    info="Minimum confidence to count as a 'find'. Lower = more results but more false positives; Higher = fewer but more certain results.",
                )
        
        components["processing_status"] = gr.HTML(visible=False)
        
        with gr.Row():
            components["back_btn"] = gr.Button("← Back to Upload", size="lg")
            components["process_btn"] = gr.Button("Process Documents →", variant="primary", size="lg")
    
    return step, components


def setup_config_handlers(
    components: dict,
    file_states: dict,
    pipeline_states: dict,
    walkthrough: gr.Walkthrough,
    results_components: dict,
) -> None:
    """Set up event handlers for the configuration stage.
    
    Args:
        components: Dictionary of UI components from build_config_stage
        file_states: Dictionary with query_file_state and source_file_state
        pipeline_states: Dictionary with results_state and query_doc_state
        walkthrough: The Walkthrough component for navigation
        results_components: Components from results stage for updating
    """
    from .results_stage import update_results_display
    
    # Back button: Step 2 → Step 1
    components["back_btn"].click(
        fn=lambda: gr.Walkthrough(selected=0),
        outputs=walkthrough,
    )
    
    # Process button: Step 2 → Step 3
    components["process_btn"].click(
        fn=_show_processing_status,
        outputs=components["processing_status"],
    ).then(
        fn=_process_documents,
        inputs=[
            file_states["query_file_state"],
            file_states["source_file_state"],
            components["classification_model"],
            components["embedding_model"],
            components["top_k"],
            components["threshold"],
        ],
        outputs=[
            components["processing_status"],
            walkthrough,
            pipeline_states["results_state"],
            pipeline_states["query_doc_state"],
        ],
    ).then(
        fn=update_results_display,
        inputs=[
            pipeline_states["results_state"],
            pipeline_states["query_doc_state"],
            components["threshold"],
        ],
        outputs=[
            results_components["query_segments"],
            results_components["query_segments_state"],
            results_components["matches_dict_state"],
        ],
    )
