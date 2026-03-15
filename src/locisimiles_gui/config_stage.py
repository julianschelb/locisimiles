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

from locisimiles.document import Document
from locisimiles.pipeline import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    DEFAULT_WORD2VEC_MODEL_PATH,
    ExhaustiveClassificationPipeline,
    LatinBertRetrievalPipeline,
    LatinBertTwoStagePipeline,
    RetrievalPipeline,
    RuleBasedPipeline,
    TwoStagePipeline,
    Word2VecRetrievalPipeline,
)

from .utils import validate_csv

# Pipeline type constants
PIPELINE_TWO_STAGE = "Two-Stage (Embedding + Classification)"
PIPELINE_EXHAUSTIVE = "Exhaustive Classification"
PIPELINE_RETRIEVAL = "Retrieval Only (Embedding Similarity)"
PIPELINE_RULE_BASED = "Rule-Based (Lexical Matching)"
PIPELINE_WORD2VEC = "Word2Vec Retrieval (Burns-Style)"
PIPELINE_LATIN_BERT_RETRIEVAL = "Contextual Latin BERT Retrieval"
PIPELINE_LATIN_BERT_TWO_STAGE = "Contextual Latin BERT Two-Stage"

PIPELINE_CHOICES = [
    PIPELINE_TWO_STAGE,
    PIPELINE_EXHAUSTIVE,
    PIPELINE_RETRIEVAL,
    PIPELINE_RULE_BASED,
    PIPELINE_WORD2VEC,
    PIPELINE_LATIN_BERT_RETRIEVAL,
    PIPELINE_LATIN_BERT_TWO_STAGE,
]

PIPELINE_DESCRIPTIONS = {
    PIPELINE_TWO_STAGE: (
        "**Two-Stage Pipeline** — Stage 1 uses embeddings to quickly retrieve the "
        "top-K most similar source segments. Stage 2 applies a classification model "
        "to accurately identify true intertextual references among the candidates. "
        "Best balance of speed and accuracy for large corpora."
    ),
    PIPELINE_EXHAUSTIVE: (
        "**Exhaustive Classification** — Classifies *every* possible query-source pair "
        "with a fine-tuned model. Most thorough but slowest; best suited for small corpora."
    ),
    PIPELINE_RETRIEVAL: (
        "**Retrieval Pipeline** — Ranks source segments by embedding cosine similarity "
        "and returns the top-K candidates. Fast but does not apply a classification step."
    ),
    PIPELINE_RULE_BASED: (
        "**Rule-Based Pipeline** — Identifies textual reuse through lexical matching "
        "and linguistic filters (shared words, distance criteria, punctuation patterns). "
        "No neural models required for the base configuration."
    ),
    PIPELINE_WORD2VEC: (
        "**Word2Vec Retrieval Pipeline** — Uses a local pre-trained Word2Vec model "
        "with bigram-level pair-aware similarity scoring inspired by Burns et al. (2021). "
        "Fast retrieval-only option for lemmatized input."
    ),
    PIPELINE_LATIN_BERT_RETRIEVAL: (
        "**Contextual Latin BERT Retrieval** — Computes token-level contextual embeddings "
        "with a transformer model and retrieves source segments by max token similarity. "
        "Best for exploratory semantic search without a second-stage classifier."
    ),
    PIPELINE_LATIN_BERT_TWO_STAGE: (
        "**Contextual Latin BERT Two-Stage** — Uses contextual token retrieval first, "
        "then reranks candidates with a classification model. Better precision for "
        "intertextual detection workflows."
    ),
}


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


def _update_pipeline_visibility(pipeline_type: str) -> tuple:
    """Return visibility updates for parameter groups based on selected pipeline.

    Returns:
        Tuple of (description, embedding_group, classification_group,
                  retrieval_group, rule_based_group, word2vec_group,
                  contextual_group)
    """
    desc = PIPELINE_DESCRIPTIONS.get(pipeline_type, "")
    show_embedding = pipeline_type in (PIPELINE_TWO_STAGE, PIPELINE_RETRIEVAL)
    show_classification = pipeline_type in (
        PIPELINE_TWO_STAGE,
        PIPELINE_EXHAUSTIVE,
        PIPELINE_LATIN_BERT_TWO_STAGE,
    )
    show_retrieval = pipeline_type in (
        PIPELINE_TWO_STAGE,
        PIPELINE_RETRIEVAL,
        PIPELINE_WORD2VEC,
        PIPELINE_LATIN_BERT_RETRIEVAL,
        PIPELINE_LATIN_BERT_TWO_STAGE,
    )
    show_rule_based = pipeline_type == PIPELINE_RULE_BASED
    show_word2vec = pipeline_type == PIPELINE_WORD2VEC
    show_contextual = pipeline_type in (
        PIPELINE_LATIN_BERT_RETRIEVAL,
        PIPELINE_LATIN_BERT_TWO_STAGE,
    )
    return (
        gr.update(value=desc),
        gr.update(visible=show_embedding),
        gr.update(visible=show_classification),
        gr.update(visible=show_retrieval),
        gr.update(visible=show_rule_based),
        gr.update(visible=show_word2vec),
        gr.update(visible=show_contextual),
    )


def _detect_device() -> str:
    """Detect the best available torch device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _process_documents(
    query_file: str | None,
    source_file: str | None,
    pipeline_type: str,
    classification_model: str,
    embedding_model: str,
    top_k: int,
    threshold: float,
    min_shared_words: int,
    min_complura: int,
    max_distance: int,
    rb_similarity_threshold: float,
    use_htrg: bool,
    use_similarity: bool,
    word2vec_model_path: str,
    word2vec_interval: int,
    word2vec_order_free: bool,
    contextual_model_name: str,
    contextual_model_path: str,
    contextual_max_length: int,
    contextual_min_token_length: int,
    contextual_use_stopword_filter: bool,
) -> tuple:
    """Process the documents using the selected pipeline and navigate to results step.

    Args:
        query_file: Path to query CSV file.
        source_file: Path to source CSV file.
        pipeline_type: Selected pipeline type string.
        classification_model: HuggingFace classification model identifier.
        embedding_model: HuggingFace embedding model identifier.
        top_k: Number of top candidates to retrieve.
        threshold: Classification / similarity threshold for display filtering.
        min_shared_words: (Rule-based) Minimum shared non-stopwords.
        min_complura: (Rule-based) Minimum adjacent tokens for complura.
        max_distance: (Rule-based) Maximum distance between shared words.
        rb_similarity_threshold: (Rule-based) Semantic similarity threshold.
        use_htrg: (Rule-based) Whether to apply HTRG POS filter.
        use_similarity: (Rule-based) Whether to apply spaCy similarity filter.
        word2vec_model_path: (Word2Vec) Local path to gensim model file.
        word2vec_interval: (Word2Vec) Bigram token gap interval.
        word2vec_order_free: (Word2Vec) Whether bigrams are order-insensitive.
        contextual_model_name: (Contextual) HuggingFace model identifier.
        contextual_model_path: (Contextual) Optional local model path.
        contextual_max_length: (Contextual) Max sequence length for encoding.
        contextual_min_token_length: (Contextual) Minimum token length retained.
        contextual_use_stopword_filter: (Contextual) Whether stopword filtering is applied.

    Returns:
        Tuple of (processing_status_update, walkthrough_update, results_state, query_doc_state)
    """
    if not query_file or not source_file:
        gr.Warning("Both query and source documents must be uploaded before processing.")
        return gr.update(visible=False), gr.Walkthrough(selected=1), None, None

    # Validate both files
    query_valid, _query_msg = validate_csv(query_file)
    source_valid, _source_msg = validate_csv(source_file)

    if not query_valid or not source_valid:
        gr.Warning("Please ensure both documents are valid before processing.")
        return gr.update(visible=False), gr.Walkthrough(selected=1), None, None

    try:
        device = _detect_device()

        # Build the selected pipeline
        if pipeline_type == PIPELINE_TWO_STAGE:
            pipeline = TwoStagePipeline(
                classification_name=classification_model,
                embedding_model_name=embedding_model,
                device=device,
            )
        elif pipeline_type == PIPELINE_EXHAUSTIVE:
            pipeline = ExhaustiveClassificationPipeline(
                classification_name=classification_model,
                device=device,
            )
        elif pipeline_type == PIPELINE_RETRIEVAL:
            pipeline = RetrievalPipeline(
                embedding_model_name=embedding_model,
                device=device,
                top_k=top_k,
            )
        elif pipeline_type == PIPELINE_RULE_BASED:
            pipeline = RuleBasedPipeline(
                min_shared_words=int(min_shared_words),
                min_complura=int(min_complura),
                max_distance=int(max_distance),
                similarity_threshold=float(rb_similarity_threshold),
                use_htrg=bool(use_htrg),
                use_similarity=bool(use_similarity),
                device=device,
            )
        elif pipeline_type == PIPELINE_WORD2VEC:
            model_path = word2vec_model_path.strip() if word2vec_model_path else ""
            pipeline = Word2VecRetrievalPipeline(
                model_path=model_path or DEFAULT_WORD2VEC_MODEL_PATH,
                top_k=int(top_k),
                similarity_threshold=float(threshold),
                interval=int(word2vec_interval),
                order_free=bool(word2vec_order_free),
            )
        elif pipeline_type == PIPELINE_LATIN_BERT_RETRIEVAL:
            resolved_local_path = contextual_model_path.strip() if contextual_model_path else ""
            normalized_model_name = contextual_model_name.strip()
            if (
                resolved_local_path
                and normalized_model_name
                and normalized_model_name != DEFAULT_CONTEXTUAL_BERT_MODEL_NAME
            ):
                gr.Error(
                    "Provide either a contextual HF model name or a local model path, not both."
                )
                return gr.update(visible=False), gr.Walkthrough(selected=1), None, None

            pipeline = LatinBertRetrievalPipeline(
                model_name=normalized_model_name or DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
                model_path=resolved_local_path or None,
                device=device,
                top_k=int(top_k),
                similarity_threshold=float(threshold),
                max_length=int(contextual_max_length),
                min_token_length=int(contextual_min_token_length),
                use_stopword_filter=bool(contextual_use_stopword_filter),
            )
        elif pipeline_type == PIPELINE_LATIN_BERT_TWO_STAGE:
            resolved_local_path = contextual_model_path.strip() if contextual_model_path else ""
            normalized_model_name = contextual_model_name.strip()
            if (
                resolved_local_path
                and normalized_model_name
                and normalized_model_name != DEFAULT_CONTEXTUAL_BERT_MODEL_NAME
            ):
                gr.Error(
                    "Provide either a contextual HF model name or a local model path, not both."
                )
                return gr.update(visible=False), gr.Walkthrough(selected=1), None, None

            pipeline = LatinBertTwoStagePipeline(
                classification_name=classification_model,
                model_name=normalized_model_name or DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
                model_path=resolved_local_path or None,
                device=device,
                max_length=int(contextual_max_length),
                min_token_length=int(contextual_min_token_length),
                use_stopword_filter=bool(contextual_use_stopword_filter),
            )
        else:
            gr.Error(f"Unknown pipeline type: {pipeline_type}")
            return gr.update(visible=False), gr.Walkthrough(selected=1), None, None

        # Load documents
        query_doc = Document(query_file)
        source_doc = Document(source_file)

        # Build run kwargs — only pass top_k when the pipeline uses it
        run_kwargs: dict = {}
        if pipeline_type in (
            PIPELINE_TWO_STAGE,
            PIPELINE_RETRIEVAL,
            PIPELINE_WORD2VEC,
            PIPELINE_LATIN_BERT_RETRIEVAL,
            PIPELINE_LATIN_BERT_TWO_STAGE,
        ):
            run_kwargs["top_k"] = top_k

        # Run pipeline
        results = pipeline.run(query=query_doc, source=source_doc, **run_kwargs)

        # Store results
        num_queries = len(results)
        total_matches = sum(len(matches) for matches in results.values())

        print(
            f"Processing complete! Found matches for {num_queries} query segments "
            f"({total_matches} total matches)."
        )

        # Return results and navigate to results step (Step 3, id=2)
        return (
            gr.update(visible=False),  # Hide processing status
            gr.Walkthrough(selected=2),  # Navigate to Results step
            results,  # Store results in state
            query_doc,  # Store query doc in state
        )

    except Exception as e:
        print(f"Processing error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        gr.Error(f"Processing failed: {str(e)}")
        return (
            gr.update(visible=False),  # Hide processing status
            gr.Walkthrough(selected=1),  # Stay on Configuration step
            None,  # No results
            None,  # No query doc
        )


def build_config_stage() -> tuple[gr.Step, dict]:
    """Build the configuration stage UI.

    Returns:
        Tuple of (Step component, dict of components for external access)
    """
    components = {}

    with gr.Step("Pipeline Configuration", id=1) as step:
        gr.Markdown("### ⚙️ Step 2: Pipeline Configuration")
        gr.Markdown("Choose a pipeline and configure its parameters.")

        # ── Pipeline selector ──────────────────────────────────────────
        components["pipeline_type"] = gr.Radio(
            label="Pipeline Type",
            choices=PIPELINE_CHOICES,
            value=PIPELINE_TWO_STAGE,
            info="Select which detection pipeline to use.",
        )
        components["pipeline_description"] = gr.Markdown(
            value=PIPELINE_DESCRIPTIONS[PIPELINE_TWO_STAGE],
        )

        with gr.Row():
            # ── Embedding model (Two-Stage & Retrieval) ────────────────
            with gr.Column() as embedding_group:
                gr.Markdown("**🤖 Embedding Model**")
                components["embedding_model"] = gr.Dropdown(
                    label="Embedding Model",
                    choices=[
                        "julian-schelb/multilingual-e5-small-emb-lat-intertext-v1",
                        "julian-schelb/multilingual-e5-base-emb-lat-intertext-v1",
                        "julian-schelb/multilingual-e5-large-emb-lat-intertext-v1",
                        "julian-schelb/granite-embedding-107m-emb-lat-intertext-v1",
                        "julian-schelb/granite-embedding-278m-emb-lat-intertext-v1",
                        "julian-schelb/sphilberta-emb-lat-intertext-v1",
                        "julian-schelb/bge-m3-emb-lat-intertext-v1",
                    ],
                    value="julian-schelb/multilingual-e5-large-emb-lat-intertext-v1",
                    interactive=True,
                    info="Sentence-transformer model for candidate retrieval.",
                )
            components["embedding_group"] = embedding_group

            # ── Classification model (Two-Stage & Exhaustive) ──────────
            with gr.Column() as classification_group:
                gr.Markdown("**🤖 Classification Model**")
                components["classification_model"] = gr.Dropdown(
                    label="Classification Model",
                    choices=[
                        "julian-schelb/mmbert-small-class-lat-intertext-v1",
                        "julian-schelb/mmbert-base-class-lat-intertext-v1",
                        "julian-schelb/bert-romanian-class-lat-intertext-v1",
                        "julian-schelb/philberta-class-lat-intertext-v2",
                        "julian-schelb/laberta-class-lat-intertext-v1",
                        "julian-schelb/modernbert-large-class-lat-intertext-v1",
                        "julian-schelb/modernbert-base-class-lat-intertext-v1",
                        "julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
                        "julian-schelb/xlm-roberta-base-class-lat-intertext-v1",
                        "julian-schelb/roberta-base-latin-v2-class-lat-intertext-v1",
                    ],
                    value="julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
                    interactive=True,
                    info="Model used to classify candidate pairs.",
                )
            components["classification_group"] = classification_group

        # ── Retrieval parameters (Two-Stage & Retrieval) ───────────────
        with gr.Row(visible=True) as retrieval_group, gr.Column():
            gr.Markdown("**🛠️ Retrieval Parameters**")
            components["top_k"] = gr.Slider(
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                label="Top K Candidates",
                info="How many candidates to retrieve per query segment.",
            )
            components["threshold"] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.85,
                step=0.05,
                label="Display Threshold",
                info="Minimum confidence to highlight as a 'find' in the results view.",
            )
        components["retrieval_group"] = retrieval_group

        # ── Rule-based parameters (Rule-Based only) ────────────────────
        with gr.Row(visible=False) as rule_based_group:
            with gr.Column():
                gr.Markdown("**🛠️ Rule-Based Parameters**")
                components["min_shared_words"] = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Min Shared Words",
                    info="Minimum number of shared non-stopwords required.",
                )
                components["min_complura"] = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Min Complura",
                    info="Minimum adjacent tokens for complura detection.",
                )
                components["max_distance"] = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=3,
                    step=1,
                    label="Max Distance",
                    info="Maximum distance between shared words.",
                )
            with gr.Column():
                gr.Markdown("**🔬 Optional Filters**")
                components["rb_similarity_threshold"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Similarity Threshold",
                    info="Threshold for semantic similarity filter (if enabled).",
                )
                components["use_htrg"] = gr.Checkbox(
                    label="Use HTRG filter (POS-based)",
                    value=False,
                    info="Apply part-of-speech analysis. Requires torch.",
                )
                components["use_similarity"] = gr.Checkbox(
                    label="Use similarity filter",
                    value=False,
                    info="Apply word-embedding similarity check. Requires spaCy.",
                )
                components["rb_threshold"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    label="Display Threshold",
                    info="Minimum confidence to highlight as a 'find' in the results view.",
                )
        components["rule_based_group"] = rule_based_group

        with gr.Row(visible=False) as word2vec_group:
            with gr.Column():
                gr.Markdown("**🧠 Word2Vec Parameters**")
                components["word2vec_model_path"] = gr.Textbox(
                    label="Word2Vec Model Path",
                    value=str(DEFAULT_WORD2VEC_MODEL_PATH),
                    info="Path to a local gensim .model file.",
                )
                components["word2vec_interval"] = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=1,
                    label="Bigram Interval",
                    info="Maximum token gap between bigram words.",
                )
                components["word2vec_order_free"] = gr.Checkbox(
                    label="Order-Free Bigrams",
                    value=False,
                    info="Treat bigrams as order-insensitive.",
                )
        components["word2vec_group"] = word2vec_group

        with gr.Row(visible=False) as contextual_group:
            with gr.Column():
                gr.Markdown("**🧠 Contextual Latin BERT Parameters**")
                components["contextual_model_name"] = gr.Textbox(
                    label="Contextual Model Name (HF)",
                    value="xlm-roberta-base",
                    info="HuggingFace model id. Leave blank when using local path.",
                )
                components["contextual_model_path"] = gr.Textbox(
                    label="Contextual Model Path (Local)",
                    value="",
                    info="Optional local model directory. Leave blank to use HF model name.",
                )
                components["contextual_max_length"] = gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=256,
                    step=16,
                    label="Max Sequence Length",
                    info="Maximum tokenized length per segment.",
                )
                components["contextual_min_token_length"] = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="Min Token Length",
                    info="Ignore shorter tokens during contextual scoring.",
                )
                components["contextual_use_stopword_filter"] = gr.Checkbox(
                    label="Use Latin Stopword Filter",
                    value=True,
                    info="Filters common function words before token-level similarity.",
                )
        components["contextual_group"] = contextual_group

        components["processing_status"] = gr.HTML(visible=False)

        with gr.Row():
            components["back_btn"] = gr.Button("← Back to Upload", size="lg")
            components["process_btn"] = gr.Button(
                "Process Documents →", variant="primary", size="lg"
            )

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

    # Pipeline type change → toggle parameter visibility
    components["pipeline_type"].change(
        fn=_update_pipeline_visibility,
        inputs=[components["pipeline_type"]],
        outputs=[
            components["pipeline_description"],
            components["embedding_group"],
            components["classification_group"],
            components["retrieval_group"],
            components["rule_based_group"],
            components["word2vec_group"],
            components["contextual_group"],
        ],
    )

    # Back button: Step 2 → Step 1
    components["back_btn"].click(
        fn=lambda: gr.Walkthrough(selected=0),
        outputs=walkthrough,
    )

    # Helper to pick the right threshold component value
    def _get_effective_threshold(
        pipeline_type: str, threshold: float, rb_threshold: float
    ) -> float:
        """Return the display threshold relevant for the selected pipeline."""
        if pipeline_type == PIPELINE_RULE_BASED:
            return rb_threshold
        return threshold

    # Process button: Step 2 → Step 3
    components["process_btn"].click(
        fn=_show_processing_status,
        outputs=components["processing_status"],
    ).then(
        fn=_process_documents,
        inputs=[
            file_states["query_file_state"],
            file_states["source_file_state"],
            components["pipeline_type"],
            components["classification_model"],
            components["embedding_model"],
            components["top_k"],
            components["threshold"],
            components["min_shared_words"],
            components["min_complura"],
            components["max_distance"],
            components["rb_similarity_threshold"],
            components["use_htrg"],
            components["use_similarity"],
            components["word2vec_model_path"],
            components["word2vec_interval"],
            components["word2vec_order_free"],
            components["contextual_model_name"],
            components["contextual_model_path"],
            components["contextual_max_length"],
            components["contextual_min_token_length"],
            components["contextual_use_stopword_filter"],
        ],
        outputs=[
            components["processing_status"],
            walkthrough,
            pipeline_states["results_state"],
            pipeline_states["query_doc_state"],
        ],
    ).then(
        fn=lambda pt, t, rbt, *args: update_results_display(
            *args, threshold=_get_effective_threshold(pt, t, rbt)
        ),
        inputs=[
            components["pipeline_type"],
            components["threshold"],
            components["rb_threshold"],
            pipeline_states["results_state"],
            pipeline_states["query_doc_state"],
        ],
        outputs=[
            results_components["query_segments"],
            results_components["query_segments_state"],
            results_components["matches_dict_state"],
        ],
    )
