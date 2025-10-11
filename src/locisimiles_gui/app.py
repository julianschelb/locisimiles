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

from .upload_stage import build_upload_stage, setup_upload_handlers
from .config_stage import build_config_stage, setup_config_handlers
from .results_stage import build_results_stage, setup_results_handlers, update_results_display, _export_results_to_csv


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
            # ========== Build All Stages ==========
            upload_step, upload_components = build_upload_stage()
            config_step, config_components = build_config_stage()
            results_step, results_components = build_results_stage()
        
        # ========== Setup Event Handlers ==========
        
        # File states for passing between stages
        file_states = {
            "query_file_state": query_file_state,
            "source_file_state": source_file_state,
        }
        
        # Pipeline states for passing between stages
        pipeline_states = {
            "results_state": results_state,
            "query_doc_state": query_doc_state,
        }
        
        # Setup handlers for each stage
        setup_upload_handlers(upload_components, file_states)
        setup_config_handlers(config_components, file_states, pipeline_states, walkthrough, results_components)
        setup_results_handlers(results_components, walkthrough)
        
        # Navigation: Step 1 → Step 2
        upload_components["next_btn"].click(
            fn=lambda: gr.Walkthrough(selected=1),
            outputs=walkthrough,
        )
        
        # Download results
        results_components["download_btn"].click(
            fn=lambda qs, md, t: _export_results_to_csv(qs, md, t) if qs and md else None,
            inputs=[
                results_components["query_segments_state"],
                results_components["matches_dict_state"],
                config_components["threshold"],
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
