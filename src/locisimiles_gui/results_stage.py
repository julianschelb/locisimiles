"""Results stage for the Loci Similes GUI."""

from __future__ import annotations

import csv
import io
import re
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from locisimiles.document import Document, TextSegment

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
        # Return empty data if no results
        return [], {}
    
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


def build_results_stage() -> tuple[gr.Step, dict[str, Any]]:
    """Build the results stage UI.
    
    Returns:
        A tuple of (Step component, components_dict) where components_dict contains
        references to all interactive components that need to be accessed later.
    """
    with gr.Step("Results", id=2) as step:
        # State to hold current query segments and matches
        query_segments_state = gr.State(value=[])
        matches_dict_state = gr.State(value={})
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
                    value=[],
                    headers=["Segment ID", "Text", "Finds"],
                    interactive=False,
                    show_label=False,
                    label="Query Document Segments",
                    wrap=True,
                    max_height=600,
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
        
        with gr.Row():
            restart_btn = gr.Button("← Start Over", size="lg")

    # Return the step and all components that need to be accessed
    components = {
        "query_segments": query_segments,
        "query_segments_state": query_segments_state,
        "matches_dict_state": matches_dict_state,
        "source_matches": source_matches,
        "selection_prompt": selection_prompt,
        "download_btn": download_btn,
        "restart_btn": restart_btn,
    }
    
    return step, components


def setup_results_handlers(components: dict, walkthrough: gr.Walkthrough) -> None:
    """Set up event handlers for the results stage.
    
    Args:
        components: Dictionary of UI components from build_results_stage
        walkthrough: The Walkthrough component for navigation
    """
    # Selection handler for query segments
    components["query_segments"].select(
        fn=_on_query_select,
        inputs=[components["query_segments_state"], components["matches_dict_state"]],
        outputs=[components["selection_prompt"], components["source_matches"]],
    )
    
    # Restart button: Step 3 → Step 1
    components["restart_btn"].click(
        fn=lambda: gr.Walkthrough(selected=0),
        outputs=walkthrough,
    )
