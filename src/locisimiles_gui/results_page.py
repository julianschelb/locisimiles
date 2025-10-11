"""Results page component for the Gradio GUI."""

from __future__ import annotations

import csv
import tempfile
from typing import Any, Dict, List, Tuple

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError("Gradio is required for results page") from exc

from locisimiles.document import Document, TextSegment

# Type aliases from pipeline
FullDict = Dict[str, List[Tuple[TextSegment, float, float]]]


def update_results_display(results: FullDict | None, query_doc: Document | None, threshold: float = 0.5) -> tuple[dict, dict, dict]:
    """Update the results display with new data.
    
    Args:
        results: Pipeline results
        query_doc: Query document
        threshold: Classification probability threshold for counting finds
    
    Returns:
        Tuple of (query_segments_update, query_segments_state, matches_dict_state)
    """
    query_segments, matches_dict = _convert_results_to_display(results, query_doc, threshold)
    
    return (
        gr.update(value=query_segments),  # Update query segments dataframe
        query_segments,                   # Update query segments state
        matches_dict,                     # Update matches dict state
    )


# Mock data for demonstration
MOCK_QUERY_SEGMENTS = [
    ["hier. adv. iovin. 1.1", "Furiosas Apollinis uates legimus; et illud Uirgilianum: Dat sine mente sonum.", 2],
    ["hier. adv. iovin. 1.41", "O decus Italiae, uirgo!", 1],
    ["hier. adv. iovin. 2.36", "Uirgilianum consilium est: Coniugium uocat, hoc praetexit nomine culpam.", 1],
    ["hier. adv. pelag. 1.23", "Hoc totum dico, quod non omnia possumus omnes.", 2],
    ["hier. adv. pelag. 3.11", "Numquam hodie effugies, ueniam quocumque uocaris.", 1],
]


def _format_metric_with_bar(value: float, is_above_threshold: bool = False) -> str:
    """Format a metric value with a visual progress bar.
    
    Args:
        value: Metric value between 0 and 1
        is_above_threshold: Whether to highlight this value
    
    Returns:
        HTML string with progress bar
    """
    percentage = int(value * 100)
    
    # Choose color based on threshold
    if is_above_threshold:
        bar_color = "#6B9BD1"  # Blue accent for findings
        bg_color = "#E3F2FD"   # Light blue background
    else:
        bar_color = "#B0B0B0"  # Gray for below threshold
        bg_color = "#F5F5F5"   # Light gray background
    
    html = f'''
    <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
        <div style="flex: 1; background-color: {bg_color}; border-radius: 4px; overflow: hidden; height: 20px; position: relative;">
            <div style="background-color: {bar_color}; width: {percentage}%; height: 100%; transition: width 0.3s;"></div>
        </div>
        <span style="min-width: 45px; text-align: right; font-weight: {'bold' if is_above_threshold else 'normal'};">{value:.3f}</span>
    </div>
    '''
    return html


def _create_mock_matches(threshold: float = 0.5) -> dict:
    """Create mock matches data with formatted visualizations."""
    raw_mock = {
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
    
    # Format with progress bars
    formatted = {}
    for query_id, matches in raw_mock.items():
        formatted[query_id] = [
            [
                seg_id,
                text,
                _format_metric_with_bar(sim, prob >= threshold),
                _format_metric_with_bar(prob, prob >= threshold)
            ]
            for seg_id, text, sim, prob in matches
        ]
    return formatted

MOCK_MATCHES = _create_mock_matches()


def _convert_results_to_display(results: FullDict | None, query_doc: Document | None, threshold: float = 0.5) -> tuple[list[list], dict]:
    """Convert pipeline results to display format.
    
    Args:
        results: Pipeline results (FullDict format)
        query_doc: Query document
        threshold: Classification probability threshold for counting finds
    
    Returns:
        Tuple of (query_segments_list, matches_dict)
    """
    if results is None or query_doc is None:
        # Return mock data if no results
        return MOCK_QUERY_SEGMENTS, MOCK_MATCHES
    
    # First pass: Create raw matches dictionary and count finds
    raw_matches = {}
    find_counts = {}
    
    for query_id, match_list in results.items():
        # Sort by probability (descending) to show most likely matches first
        sorted_matches = sorted(match_list, key=lambda x: x[2], reverse=True)  # x[2] is probability
        
        # Store raw numeric values
        raw_matches[query_id] = sorted_matches
        
        # Count finds above threshold
        find_counts[query_id] = sum(1 for _, _, prob in sorted_matches if prob >= threshold)
    
    # Convert query document to list format with find counts
    # Document is iterable and returns TextSegments in order
    query_segments = []
    for segment in query_doc:
        find_count = find_counts.get(segment.id, 0)
        query_segments.append([segment.id, segment.text, find_count])
    
    # Second pass: Format matches with HTML progress bars
    matches_dict = {}
    for query_id, match_list in raw_matches.items():
        matches_dict[query_id] = [
            [
                source_seg.id,
                source_seg.text,
                _format_metric_with_bar(round(similarity, 3), probability >= threshold),
                _format_metric_with_bar(round(probability, 3), probability >= threshold)
            ]
            for source_seg, similarity, probability in match_list
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


def _extract_numeric_from_html(html_str: str) -> float:
    """Extract numeric value from HTML formatted metric string.
    
    Args:
        html_str: HTML string with embedded numeric value
    
    Returns:
        Extracted numeric value
    """
    import re
    # Extract the number from the span at the end: <span ...>0.XXX</span>
    match = re.search(r'<span[^>]*>([\d.]+)</span>', html_str)
    if match:
        return float(match.group(1))
    # Fallback: if it's already a number
    try:
        return float(html_str)
    except (ValueError, TypeError):
        return 0.0


def _export_results_to_csv(query_segments: list, matches_dict: dict, threshold: float) -> str:
    """Export results to a CSV file.
    
    Args:
        query_segments: List of query segments with find counts
        matches_dict: Dictionary mapping query IDs to matches
        threshold: Classification probability threshold
    
    Returns:
        Path to the temporary CSV file
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='', encoding='utf-8')
    
    with temp_file as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Query_Segment_ID",
            "Query_Text",
            "Source_Segment_ID",
            "Source_Text",
            "Similarity",
            "Probability",
            "Above_Threshold"
        ])
        
        # Write data for each query segment
        for query_row in query_segments:
            query_id = query_row[0]
            query_text = query_row[1]
            
            # Get matches for this query segment
            matches = matches_dict.get(query_id, [])
            
            if matches:
                for match in matches:
                    source_id = match[0]
                    source_text = match[1]
                    # Extract numeric values from HTML formatted strings
                    similarity = _extract_numeric_from_html(match[2]) if isinstance(match[2], str) else match[2]
                    probability = _extract_numeric_from_html(match[3]) if isinstance(match[3], str) else match[3]
                    above_threshold = "Yes" if probability >= threshold else "No"
                    
                    writer.writerow([
                        query_id,
                        query_text,
                        source_id,
                        source_text,
                        similarity,
                        probability,
                        above_threshold
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
    
    return temp_file.name


def build_results_page() -> tuple[gr.Column, dict[str, Any]]:
    """Build the results page interface.
    
    Returns:
        A tuple of (page_column, components_dict) where components_dict contains
        references to all interactive components that need to be accessed later.
    """
    # State to hold current query segments and matches
    query_segments_state = gr.State(value=MOCK_QUERY_SEGMENTS)
    matches_dict_state = gr.State(value=MOCK_MATCHES)
    
    with gr.Column() as results_page:
        gr.Markdown("### 📊 Step 3: View Results")
        gr.Markdown(
            "Select a query segment on the left to view potential intertextual references from the source document. "
            "Similarity measures the cosine similarity between embeddings (0-1, higher = more similar). "
            "Probability is the classifier's confidence that the pair represents an intertextual reference (0-1, higher = more likely)."
        )
        
        # Download button
        with gr.Row():
            download_btn = gr.DownloadButton("Download Results as CSV", variant="primary")
        
        with gr.Row():
            # Left column: Query segments
            with gr.Column(scale=1):
                gr.Markdown("### Query Document Segments")
                query_segments = gr.Dataframe(
                    value=MOCK_QUERY_SEGMENTS,
                    headers=["Segment ID", "Text", "Finds"],
                    interactive=False,
                    show_label=False,
                    label="Query Document Segments",
                    wrap=True,
                    max_height=600,
                    row_count=(len(MOCK_QUERY_SEGMENTS), "fixed"),
                    col_count=(3, "fixed"),
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
                    headers=["Source ID", "Source Text", "Similarity", "Probability"],
                    interactive=False,
                    show_label=False,
                    label="Potential Intertextual References from Source Document",
                    wrap=True,
                    max_height=600,
                    visible=False,
                    datatype=["str", "str", "html", "html"],  # Enable HTML rendering for metric columns
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
        "download_btn": download_btn,
    }
    
    return results_page, components
