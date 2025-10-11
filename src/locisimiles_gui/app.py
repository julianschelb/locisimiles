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

from .results_page import build_results_page, update_results_display, _export_results_to_csv
from .utils import validate_csv, validate_and_notify, load_csv_preview

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


def build_interface() -> gr.Blocks:
    """Create the main Gradio Blocks interface."""
    # Custom theme matching the presentation color scheme
    # Colors extracted from the slide: warm beige background, blue accents, brown text
    theme = gr.themes.Soft(
        primary_hue="blue",      # Blue from the numbered circles (#6B9BD1 area)
        secondary_hue="orange",  # Warm accent color
        neutral_hue="stone",     # Warm neutral matching the beige/cream background
    ).set(
        # Primary buttons - blue accent color
        button_primary_background_fill="#6B9BD1",
        button_primary_background_fill_hover="#5A8BC0",
        button_primary_text_color="white",
        # Body styling - warm cream/beige background
        body_background_fill="#F5F3EF",
        body_text_color="#5B4636",
        # Blocks/panels - slightly lighter cream
        block_background_fill="white",
        block_border_color="#E5E3DF",
        # Input elements
        input_background_fill="white",
        input_border_color="#D4D2CE",
    )
    
    with gr.Blocks(title="Loci Similes Demo", theme=theme) as demo:
        # State to store pipeline results and files
        results_state = gr.State(value=None)
        query_doc_state = gr.State(value=None)
        query_file_state = gr.State(value=None)
        source_file_state = gr.State(value=None)
        
        gr.Markdown("# Loci Similes - Intertextuality Detection")
        gr.Markdown(
            "Find intertextual references in Latin documents using a two-stage pipeline with pre-trained language models. "
            "The first stage uses embedding similarity to quickly retrieve candidate passages from thousands of text segments. "
            "The second stage applies a classification model to accurately identify true intertextual references among the candidates. "
            "This approach balances computational efficiency with high-quality results. "
            "*Built with the [LociSimiles Python package](https://pypi.org/project/locisimiles/).*"
        )
        
        with gr.Walkthrough(selected=0) as walkthrough:
            # ========== STEP 1: Upload Files ==========
            with gr.Step("Upload Files", id=0):
                gr.Markdown("### 📄 Step 1: Upload Documents")
                gr.Markdown("Upload two CSV files containing Latin text segments. Each CSV must have two columns: `seg_id` and `text`.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🔍 Query Document**")
                        gr.Markdown("The document in which you want to find intertextual references.")
                        query_upload = gr.File(
                            label="Upload Query CSV",
                            file_types=[".csv"],
                            type="filepath",
                        )
                        query_preview = gr.Dataframe(
                            label="Query Document Preview",
                            interactive=False,
                            visible=False,
                            max_height=400,
                            wrap=True,
                        )
                    
                    with gr.Column():
                        gr.Markdown("**📖 Source Document**")
                        gr.Markdown("The document to search for potential references.")
                        source_upload = gr.File(
                            label="Upload Source CSV",
                            file_types=[".csv"],
                            type="filepath",
                        )
                        source_preview = gr.Dataframe(
                            label="Source Document Preview",
                            interactive=False,
                            visible=False,
                            max_height=400,
                            wrap=True,
                        )
                
                with gr.Row():
                    next_to_config_btn = gr.Button("Next: Configuration →", variant="primary", size="lg")
            
            # ========== STEP 2: Pipeline Configuration ==========
            with gr.Step("Pipeline Configuration", id=1):
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
                        classification_model = gr.Dropdown(
                            label="Classification Model",
                            choices=["julian-schelb/PhilBerta-class-latin-intertext-v1"],
                            value="julian-schelb/PhilBerta-class-latin-intertext-v1",
                            interactive=True,
                            info="Model used to classify candidate pairs as intertextual or not",
                        )
                        embedding_model = gr.Dropdown(
                            label="Embedding Model",
                            choices=["julian-schelb/SPhilBerta-emb-lat-intertext-v1", "bowphs/SPhilBerta", "distilbert-base-multilingual-cased"],
                            value="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
                            interactive=True,
                            info="Model used to generate embeddings for candidate retrieval",
                        )
                    
                    # Right column: Retrieval Parameters
                    with gr.Column():
                        gr.Markdown("**🛠️ Retrieval Parameters**")
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Top K Candidates",
                            info="How many candidates to examine per query. Higher values find more references but take longer to process.",
                        )
                        threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Classification Threshold",
                            info="Minimum confidence to count as a 'find'. Lower = more results but more false positives; Higher = fewer but more certain results.",
                        )
                
                processing_status = gr.HTML(visible=False)
                
                with gr.Row():
                    back_to_upload_btn = gr.Button("← Back to Upload", size="lg")
                    process_btn = gr.Button("Process Documents →", variant="primary", size="lg")
            
            # ========== STEP 3: Results ==========
            with gr.Step("Results", id=2):
                results_page, results_components = build_results_page()
                
                with gr.Row():
                    restart_btn = gr.Button("← Start Over", size="lg")
        
        # ========== Event Handlers ==========
        
        # File upload handlers
        query_upload.change(
            fn=lambda f: (validate_and_notify(f), load_csv_preview(f), f),
            inputs=query_upload,
            outputs=[query_upload, query_preview, query_file_state],
        )
        
        source_upload.change(
            fn=lambda f: (validate_and_notify(f), load_csv_preview(f), f),
            inputs=source_upload,
            outputs=[source_upload, source_preview, source_file_state],
        )
        
        # Navigation: Step 1 → Step 2
        next_to_config_btn.click(
            fn=lambda: gr.Walkthrough(selected=1),
            outputs=walkthrough,
        )
        
        # Navigation: Step 2 → Step 1
        back_to_upload_btn.click(
            fn=lambda: gr.Walkthrough(selected=0),
            outputs=walkthrough,
        )
        
        # Processing: Step 2 → Step 3
        process_btn.click(
            fn=_show_processing_status,
            outputs=processing_status,
        ).then(
            fn=_process_documents,
            inputs=[
                query_file_state,
                source_file_state,
                classification_model,
                embedding_model,
                top_k,
                threshold,
            ],
            outputs=[
                processing_status,
                walkthrough,
                results_state,
                query_doc_state,
            ],
        ).then(
            fn=update_results_display,
            inputs=[results_state, query_doc_state, threshold],
            outputs=[
                results_components["query_segments"],
                results_components["query_segments_state"],
                results_components["matches_dict_state"],
            ],
        )
        
        # Restart: Step 3 → Step 1
        restart_btn.click(
            fn=lambda: gr.Walkthrough(selected=0),
            outputs=walkthrough,
        )
        
        # Download results
        results_components["download_btn"].click(
            fn=lambda qs, md, t: _export_results_to_csv(qs, md, t) if qs and md else None,
            inputs=[
                results_components["query_segments_state"],
                results_components["matches_dict_state"],
                threshold,
            ],
            outputs=results_components["download_btn"],
        )
        
    return demo


def launch(**kwargs: Any) -> None:
    """Launch the Gradio app."""
    # Print startup banner
    print("\n" + "="*60)
    print("🚀 Starting Loci Similes Web Interface...")
    print("="*60)
    print("\n📦 Building interface components...")
    
    demo = build_interface()
    
    print("✅ Interface built successfully!")
    print("\n🌐 Starting web server...")
    print("-"*60)
    
    kwargs.setdefault("show_api", False)
    kwargs.setdefault("inbrowser", False)
    kwargs.setdefault("quiet", False)  # Changed to False to show URL
    
    try:
        demo.launch(share=False, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "shareable link must be created" in msg:
            print(
                "\n⚠️  Unable to open the Gradio UI because localhost is blocked "
                "in this environment. Exiting without starting the server.",
                file=sys.stderr,
            )
            return
        raise


def main() -> None:
    """Entry point for the ``locisimiles-gui`` console script."""
    print("\n" + "="*60)
    print("  LOCI SIMILES - Intertextuality Detection Web Interface")
    print("="*60)
    print("\n📚 A tool for finding intertextual links in Latin literature")
    print("   using pre-trained language models.\n")
    
    launch()
    
    print("\n" + "="*60)
    print("👋 Server stopped. Thank you for using Loci Similes!")
    print("="*60 + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
