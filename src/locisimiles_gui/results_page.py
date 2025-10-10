"""Results page component for the Gradio GUI."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError("Gradio is required for results page") from exc

from locisimiles.document import Document, TextSegment

# Type aliases from pipeline
FullDict = Dict[str, List[Tuple[TextSegment, float, float]]]


def update_results_display(results: FullDict | None, query_doc: Document | None) -> tuple[dict, dict, dict]:
    """Update the results display with new data.
    
    Args:
        results: Pipeline results
        query_doc: Query document
    
    Returns:
        Tuple of (query_segments_update, query_segments_state, matches_dict_state)
    """
    query_segments, matches_dict = _convert_results_to_display(results, query_doc)
    
    return (
        gr.update(value=query_segments),  # Update query segments dataframe
        query_segments,                   # Update query segments state
        matches_dict,                     # Update matches dict state
    )


# Mock data for demonstration
MOCK_QUERY_SEGMENTS = [
    ["hier. adv. iovin. 1.1", "Furiosas Apollinis uates legimus; et illud Uirgilianum: Dat sine mente sonum."],
    ["hier. adv. iovin. 1.41", "O decus Italiae, uirgo!"],
    ["hier. adv. iovin. 2.36", "Uirgilianum consilium est: Coniugium uocat, hoc praetexit nomine culpam."],
    ["hier. adv. pelag. 1.23", "Hoc totum dico, quod non omnia possumus omnes."],
    ["hier. adv. pelag. 3.11", "Numquam hodie effugies, ueniam quocumque uocaris."],
]

MOCK_MATCHES = {
    "hier. adv. iovin. 1.1": [
        ["verg. aen. 10.636", "dat sine mente sonum gressusque effingit euntis", 0.92, 0.89],
        ["verg. aen. 6.50", "insanam uatem aspicies", 0.65, 0.54],
    ],
    "hier. adv. iovin. 1.41": [
        ["verg. aen. 11.508", "o decus Italiae uirgo, quas dicere grates", 0.95, 0.93],
        ["verg. aen. 7.473", "o germana mihi atque eadem gratissima nuper", 0.58, 0.42],
    ],
    "hier. adv. iovin. 2.36": [
        ["verg. aen. 4.172", "coniugium uocat, hoc praetexit nomine culpam.", 0.98, 0.96],
        ["verg. aen. 4.34", "anna fatebor enim", 0.43, 0.31],
    ],
    "hier. adv. pelag. 1.23": [
        ["verg. ecl. 8.63", "non omnia possumus omnes.", 0.99, 0.97],
        ["verg. georg. 2.109", "omnia fert aetas, animum quoque", 0.61, 0.48],
    ],
    "hier. adv. pelag. 3.11": [
        ["verg. ecl. 3.49", "Numquam hodie effugies; ueniam quocumque uocaris.", 0.97, 0.95],
        ["verg. aen. 6.388", "ibimus, haud uanum patimur te ducere", 0.52, 0.39],
    ],
}


def _convert_results_to_display(results: FullDict | None, query_doc: Document | None) -> tuple[list[list], dict]:
    """Convert pipeline results to display format.
    
    Args:
        results: Pipeline results (FullDict format)
        query_doc: Query document
    
    Returns:
        Tuple of (query_segments_list, matches_dict)
    """
    if results is None or query_doc is None:
        # Return mock data if no results
        return MOCK_QUERY_SEGMENTS, MOCK_MATCHES
    
    # Convert query document to list format
    # Document is iterable and returns TextSegments in order
    query_segments = []
    for segment in query_doc:
        query_segments.append([segment.id, segment.text])
    
    # Convert results to matches dictionary
    matches_dict = {}
    for query_id, match_list in results.items():
        # Sort by probability (descending) to show most likely matches first
        sorted_matches = sorted(match_list, key=lambda x: x[2], reverse=True)  # x[2] is probability
        matches_dict[query_id] = [
            [source_seg.id, source_seg.text, similarity, probability]
            for source_seg, similarity, probability in sorted_matches
        ]
    
    return query_segments, matches_dict


def _on_query_select(evt: gr.SelectData, query_segments: list, matches_dict: dict) -> tuple[dict, dict]:
    """Handle query segment selection and return matching source segments.
    
    Note: evt.index[0] gives the row number when clicking anywhere in that row.
    
    Args:
        evt: Selection event data
        query_segments: List of query segments
        matches_dict: Dictionary mapping query IDs to matches
    
    Returns:
        A tuple of (prompt_visibility_update, dataframe_update_with_data)
    """
    if evt.index is None or len(evt.index) < 1:
        return gr.update(visible=True), gr.update(visible=False)
    
    row_index = evt.index[0]
    if row_index >= len(query_segments):
        return gr.update(visible=True), gr.update(visible=False)
    
    segment_id = query_segments[row_index][0]
    matches = matches_dict.get(segment_id, [])
    
    # Hide prompt, show dataframe with results
    return gr.update(visible=False), gr.update(value=matches, visible=True)


def build_results_page() -> tuple[gr.Column, dict[str, Any]]:
    """Build the results page interface.
    
    Returns:
        A tuple of (page_column, components_dict) where components_dict contains
        references to all interactive components that need to be accessed later.
    """
    # State to hold current query segments and matches
    query_segments_state = gr.State(value=MOCK_QUERY_SEGMENTS)
    matches_dict_state = gr.State(value=MOCK_MATCHES)
    
    with gr.Column(visible=False) as results_page:
        gr.Markdown("# Processing Results")
        gr.Markdown("Select a query segment on the left to view potential intertextual references from the source document.")
        
        back_btn = gr.Button("← Back to Upload", size="sm")
        
        with gr.Row():
            # Left column: Query segments
            with gr.Column(scale=1):
                gr.Markdown("### Query Document Segments")
                query_segments = gr.Dataframe(
                    value=MOCK_QUERY_SEGMENTS,
                    headers=["Segment ID", "Text"],
                    interactive=False,
                    show_label=False,
                    label="Query Document Segments",
                    wrap=True,
                    height=500,
                    row_count=(len(MOCK_QUERY_SEGMENTS), "fixed"),
                    col_count=(2, "fixed"),
                )
            
            # Right column: Matching source segments
            with gr.Column(scale=1):
                gr.Markdown("### Potential Intertextual References")
                
                # Prompt shown initially
                selection_prompt = gr.Markdown(
                    """
                    <div style="display: flex; align-items: center; justify-content: center; height: 400px; font-size: 18px; color: #666;">
                        <div style="text-align: center;">
                            <div style="font-size: 48px; margin-bottom: 20px;">←</div>
                            <div>Select a query segment to view</div>
                            <div>potential intertextual references</div>
                        </div>
                    </div>
                    """,
                    visible=True
                )
                
                # Dataframe hidden initially
                source_matches = gr.Dataframe(
                    headers=["Source ID", "Source Text", "Cosine Similarity", "Classification Probability"],
                    interactive=False,
                    show_label=False,
                    label="Potential Intertextual References from Source Document",
                    wrap=True,
                    height=500,
                    visible=False,
                )
        
        # Set up selection handler
        query_segments.select(
            fn=_on_query_select,
            inputs=[query_segments_state, matches_dict_state],
            outputs=[selection_prompt, source_matches],
        )

    # Return the page and all components that need to be accessed
    components = {
        "query_segments": query_segments,
        "query_segments_state": query_segments_state,
        "matches_dict_state": matches_dict_state,
        "source_matches": source_matches,
        "selection_prompt": selection_prompt,
        "back_btn": back_btn,
    }
    
    return results_page, components
