"""Upload page component for the Gradio GUI."""

from __future__ import annotations

from typing import Any

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError("Gradio is required for upload page") from exc

from .utils import validate_and_notify, load_csv_preview


def build_upload_page() -> tuple[gr.Column, dict[str, Any]]:
    """Build the upload page interface.
    
    Returns:
        A tuple of (page_column, components_dict) where components_dict contains
        references to all interactive components that need to be accessed later.
    """
    with gr.Column(visible=True) as upload_page:
        gr.Markdown("# Loci Similes Demo")
        gr.Markdown(
            "Upload two Latin documents to search for intertextual links between them."
        )
        gr.Markdown(
            "**Required CSV format:** Each file must be a CSV with two columns:\n"
            "- `seg_id`: A unique identifier for each text segment (e.g., `verg. aen. 1.1`)\n"
            "- `text`: The Latin text content for that segment"
        )

        with gr.Row():
            with gr.Column():
                query_upload = gr.File(
                    label="Query document (single CSV file)",
                    file_types=[".csv"],
                    file_count="single",
                )
                query_preview = gr.Dataframe(
                    label="Query document preview",
                    headers=["seg_id", "text"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Column():
                source_upload = gr.File(
                    label="Source document (single CSV file)",
                    file_types=[".csv"],
                    file_count="single",
                )
                source_preview = gr.Dataframe(
                    label="Source document preview",
                    headers=["seg_id", "text"],
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown("### Configuration")
        gr.Markdown("Configure the parameters for detecting intertextual links:")
        
        with gr.Row():
            with gr.Column():
                classification_model = gr.Dropdown(
                    label="Classification Model",
                    choices=[
                        "julian-schelb/PhilBerta-class-latin-intertext-v1"
                    ],
                    value="julian-schelb/PhilBerta-class-latin-intertext-v1",
                    interactive=True,
                    info="Model used to classify intertextual links"
                )
                embedding_model = gr.Dropdown(
                    label="Embedding Model",
                    choices=[
                        "julian-schelb/SPhilBerta-emb-lat-intertext-v1",
                        "bowphs/SPhilBerta",
                        "distilbert-base-multilingual-cased",
                    ],
                    value="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
                    interactive=True,
                    info="Model used to generate embeddings for candidate generation"
                )
            with gr.Column():
                top_k = gr.Slider(
                    label="Top K Candidates",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    info="Number of top similar candidates to retrieve and classify"
                )
                threshold = gr.Slider(
                    label="Similarity Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    info="Minimum similarity score to consider a match"
                )

        process_btn = gr.Button("Process Files", variant="primary", size="lg")
        
        # Processing status indicator
        processing_status = gr.HTML(
            value="",
            visible=False,
        )

    # Set up event handlers
    query_upload.upload(
        fn=lambda x: validate_and_notify(x, "Query document"),
        inputs=[query_upload],
        outputs=[query_upload],
    )
    query_upload.change(
        fn=load_csv_preview,
        inputs=[query_upload],
        outputs=[query_preview],
    )
    source_upload.upload(
        fn=lambda x: validate_and_notify(x, "Source document"),
        inputs=[source_upload],
        outputs=[source_upload],
    )
    source_upload.change(
        fn=load_csv_preview,
        inputs=[source_upload],
        outputs=[source_preview],
    )

    # Return the page and all components that need to be accessed
    components = {
        "query_upload": query_upload,
        "source_upload": source_upload,
        "query_preview": query_preview,
        "source_preview": source_preview,
        "classification_model": classification_model,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "threshold": threshold,
        "process_btn": process_btn,
        "processing_status": processing_status,
    }
    
    return upload_page, components
