"""Upload stage for the Loci Similes GUI."""

from __future__ import annotations

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

from .utils import validate_and_notify, load_csv_preview


def build_upload_stage() -> tuple[gr.Step, dict]:
    """Build the upload stage UI.
    
    Returns:
        Tuple of (Step component, dict of components for external access)
    """
    components = {}
    
    with gr.Step("Upload Files", id=0) as step:
        gr.Markdown("### 📄 Step 1: Upload Documents")
        gr.Markdown("Upload two CSV files containing Latin text segments. Each CSV must have two columns: `seg_id` and `text`.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**🔍 Query Document**")
                gr.Markdown("The document in which you want to find intertextual references.")
                components["query_upload"] = gr.File(
                    label="Upload Query CSV",
                    file_types=[".csv"],
                    type="filepath",
                )
                components["query_preview"] = gr.Dataframe(
                    label="Query Document Preview",
                    interactive=False,
                    visible=False,
                    max_height=400,
                    wrap=True,
                )
            
            with gr.Column():
                gr.Markdown("**📚 Source Document**")
                gr.Markdown("The document to search for potential references.")
                components["source_upload"] = gr.File(
                    label="Upload Source CSV",
                    file_types=[".csv"],
                    type="filepath",
                )
                components["source_preview"] = gr.Dataframe(
                    label="Source Document Preview",
                    interactive=False,
                    visible=False,
                    max_height=400,
                    wrap=True,
                )
        
        with gr.Row():
            components["next_btn"] = gr.Button("Next: Configuration →", variant="primary", size="lg")
    
    return step, components


def setup_upload_handlers(components: dict, file_states: dict) -> None:
    """Set up event handlers for the upload stage.
    
    Args:
        components: Dictionary of UI components from build_upload_stage
        file_states: Dictionary with query_file_state and source_file_state
    """
    # Query file upload handler
    components["query_upload"].change(
        fn=lambda f: (validate_and_notify(f), load_csv_preview(f), f),
        inputs=components["query_upload"],
        outputs=[
            components["query_upload"],
            components["query_preview"],
            file_states["query_file_state"],
        ],
    )
    
    # Source file upload handler
    components["source_upload"].change(
        fn=lambda f: (validate_and_notify(f), load_csv_preview(f), f),
        inputs=components["source_upload"],
        outputs=[
            components["source_upload"],
            components["source_preview"],
            file_states["source_file_state"],
        ],
    )
