from collections import defaultdict
from io import BytesIO
from pathlib import Path
import colorsys
import html
import textwrap

import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pm4py.visualization.ocel.ocpn.variants import wo_decoration

from ..discovery.subprocess_detection import (
    _arc_key,
    _build_component_colors,
    _place_key,
    _transition_key,
    collapse_sub_processes,
)
from ..discovery.region_detection import build_object_centric_region_highlight


def print_tuple_dict_matrices(push, pull, decimals=4):
    """
    Print two tuple-key dictionaries as matrices in the terminal.

    Each input must look like:
        {('row_name', 'col_name'): value, ...}
    """

    def to_matrix(d):
        if not isinstance(d, dict):
            raise TypeError("Each input must be a dictionary.")
        if not all(isinstance(k, tuple) and len(k) == 2 for k in d):
            raise ValueError("All keys must be 2-item tuples like ('row', 'col').")

        df = pd.Series(d).unstack()

        # Keep a stable row/column order based on appearance in the dict
        rows = []
        cols = []
        for r, c in d.keys():
            if r not in rows:
                rows.append(r)
            if c not in cols:
                cols.append(c)

        return df.reindex(index=rows, columns=cols)

    m1 = to_matrix(push).round(decimals)
    m2 = to_matrix(pull).round(decimals)

    print("\nPush:")
    print(m1.to_string())

    print("\nPull:")
    print(m2.to_string())

    return m1, m2


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def visualize_layers_boxed(solution, title="Hierarchy Layers", output_path=None):
    layers = defaultdict(list)
    for node, layer in solution.items():
        layers[layer].append(str(node))

    sorted_layers = sorted(layers.keys(), reverse=True)
    if not sorted_layers:
        return

    lines = []
    for layer in sorted_layers:
        members = ", ".join(sorted(layers[layer]))
        lines.append(f"Layer {layer}: {members}")

    fig, ax = plt.subplots(figsize=(12, max(2, 0.8 * len(lines) + 1)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, line in enumerate(lines):
        y = len(lines) - i
        ax.text(
            0.02, y, line,
            transform=ax.get_yaxis_transform(),
            fontsize=14,
            ha="left",
            va="center",
            color="black",
            family="monospace"
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, len(lines) + 0.5)
    ax.axis("off")
    ax.set_title(title, color="black")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def _load_font(size):
    font_candidates = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue

    return ImageFont.load_default()


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def _fallback_unique_color(index, used_colors):
    color_value = (index * 0x9E3779B1 + 0x12345) & 0xFFFFFF

    while True:
        color = f"#{color_value:06x}"
        if color not in used_colors:
            return color
        color_value = (color_value + 1) & 0xFFFFFF


def _build_object_type_colors(object_types):
    unique_object_types = sorted(set(object_types))
    if not unique_object_types:
        return {}

    used_colors = set()
    object_type_colors = {}
    golden_ratio = 0.618033988749895

    for index, object_type in enumerate(unique_object_types):
        color = None

        # Use a golden-ratio hue sequence so neighboring assignments stay well
        # separated, then vary saturation/lightness if rounding to RGB would
        # otherwise reuse a color.
        for variant in range(12):
            hue = (index * golden_ratio + variant * (1 / 36)) % 1.0
            saturation = 0.55 + 0.1 * (variant % 3)
            lightness = 0.45 + 0.08 * ((variant // 3) % 3)
            candidate = _rgb_to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))
            if candidate not in used_colors:
                color = candidate
                break

        if color is None:
            color = _fallback_unique_color(index, used_colors)

        used_colors.add(color)
        object_type_colors[object_type] = color

    return object_type_colors


def _build_activity_label(activity, resource_types, object_type_colors, marker=None):
    resource_types = resource_types or []
    escaped_activity = html.escape(activity)
    if not resource_types and marker is None:
        return escaped_activity

    dots_row = ""
    if resource_types:
        dots = "&#8201;".join(
            (
                f'<FONT COLOR="{object_type_colors.get(object_type, "#000000")}" '
                'POINT-SIZE="16">&#9679;</FONT>'
            )
            for object_type in resource_types
        )
        dots_row = f'<TR><TD ALIGN="RIGHT">{dots}</TD></TR>'

    marker_row = ""
    if marker is not None:
        marker_row = (
            f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="18">{html.escape(marker)}</FONT></TD></TR>'
        )

    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f"{dots_row}"
        f'<TR><TD ALIGN="CENTER">{escaped_activity}</TD></TR>'
        f"{marker_row}"
        '</TABLE>>'
    )


def _unique_colors(colors):
    return tuple(dict.fromkeys(colors))


def _build_subprocess_style_index(subprocess_components):
    transition_colors = defaultdict(list)
    transition_fillcolors = defaultdict(list)
    transition_markers = defaultdict(list)
    place_colors = defaultdict(list)
    place_fillcolors = defaultdict(list)
    arc_colors = defaultdict(list)

    for component in subprocess_components or []:
        color = component.get("color")
        fillcolor = component.get("fillcolor")
        marker = component.get("marker")
        if not color:
            color = fillcolor
        if not color and not fillcolor and marker is None:
            continue

        for transition_key in component.get("transition_keys", ()):
            if color:
                transition_colors[transition_key].append(color)
            if fillcolor:
                transition_fillcolors[transition_key].append(fillcolor)
            if marker is not None:
                transition_markers[transition_key].append(marker)
        for place_key in component.get("place_keys", ()):
            if color:
                place_colors[place_key].append(color)
            if fillcolor:
                place_fillcolors[place_key].append(fillcolor)
        for arc_key in component.get("arc_keys", ()):
            if color:
                arc_colors[arc_key].append(color)

    return {
        "transitions": {
            key: _unique_colors(colors)
            for key, colors in transition_colors.items()
        },
        "transition_fillcolors": {
            key: _unique_colors(colors)
            for key, colors in transition_fillcolors.items()
        },
        "transition_markers": {
            key: tuple(dict.fromkeys(markers))
            for key, markers in transition_markers.items()
        },
        "places": {
            key: _unique_colors(colors)
            for key, colors in place_colors.items()
        },
        "place_fillcolors": {
            key: _unique_colors(colors)
            for key, colors in place_fillcolors.items()
        },
        "arcs": {
            key: _unique_colors(colors)
            for key, colors in arc_colors.items()
        },
    }


def _node_border_attributes(colors):
    unique_colors = _unique_colors(colors)
    if not unique_colors:
        return {}

    border_attributes = {
        "color": ":".join(unique_colors),
        "penwidth": "2.5",
    }
    if len(unique_colors) > 1:
        border_attributes["peripheries"] = str(len(unique_colors))
    return border_attributes


def _apply_node_border(node_kwargs, border_attributes):
    if not border_attributes:
        return

    node_kwargs.update(border_attributes)
    if node_kwargs.get("style") is None:
        node_kwargs["style"] = "solid"


def _build_ocpn_graphviz(
        ocpn,
        object_type_colors=None,
        activity_resource_types=None,
        highlighted_activities=None,
        subprocess_components=None,
        parameters=None,
):
    if parameters is None:
        parameters = {}

    object_type_colors = object_type_colors or {}
    activity_resource_types = activity_resource_types or {}
    highlighted_activities = set(highlighted_activities or [])
    subprocess_styles = _build_subprocess_style_index(subprocess_components)
    parameters_enum = wo_decoration.Parameters
    image_format = wo_decoration.exec_utils.get_param_value(parameters_enum.FORMAT, parameters, "png")
    bgcolor = wo_decoration.exec_utils.get_param_value(
        parameters_enum.BGCOLOR,
        parameters,
        wo_decoration.constants.DEFAULT_BGCOLOR,
    )
    rankdir = wo_decoration.exec_utils.get_param_value(
        parameters_enum.RANKDIR,
        parameters,
        wo_decoration.constants.DEFAULT_RANKDIR_GVIZ,
    )
    enable_graph_title = wo_decoration.exec_utils.get_param_value(
        parameters_enum.ENABLE_GRAPH_TITLE,
        parameters,
        wo_decoration.constants.DEFAULT_ENABLE_GRAPH_TITLES,
    )
    graph_title = wo_decoration.exec_utils.get_param_value(
        parameters_enum.GRAPH_TITLE,
        parameters,
        "Object-Centric Petri net",
    )

    filename = wo_decoration.tempfile.NamedTemporaryFile(suffix=".gv")
    filename.close()

    viz = wo_decoration.Digraph(
        "ocpn",
        filename=filename.name,
        engine="dot",
        graph_attr={"bgcolor": bgcolor},
    )
    viz.attr("node", shape="ellipse", fixedsize="false")

    if enable_graph_title:
        viz.attr(
            label='<<FONT POINT-SIZE="20">' + graph_title + "</FONT>>",
            labelloc="top",
        )

    activities_map = {}
    transition_map = {}
    places = {}

    for activity in ocpn["activities"]:
        activities_map[activity] = str(wo_decoration.uuid.uuid4())
        activity_markers = subprocess_styles["transition_markers"].get(("activity", activity), ())
        label = _build_activity_label(
            activity,
            activity_resource_types.get(activity, []),
            object_type_colors,
            marker=activity_markers[0] if activity_markers else None,
        )
        activity_colors = subprocess_styles["transitions"].get(("activity", activity), ())
        activity_fillcolors = subprocess_styles["transition_fillcolors"].get(("activity", activity), ())
        node_kwargs = {
            "label": label,
            "shape": "box",
        }
        if activity in highlighted_activities:
            node_kwargs["style"] = "filled"
            node_kwargs["fillcolor"] = "#f7d7a6"
        if activity_fillcolors:
            node_kwargs["style"] = "filled"
            node_kwargs["fillcolor"] = activity_fillcolors[0]
        _apply_node_border(node_kwargs, _node_border_attributes(activity_colors))
        viz.node(activities_map[activity], **node_kwargs)

    for object_type in ocpn["petri_nets"]:
        object_type_color = object_type_colors.get(object_type, wo_decoration.ot_to_color(object_type))
        net, initial_marking, final_marking = ocpn["petri_nets"][object_type]
        place_diagnostics = {}
        transition_diagnostics = {}
        if object_type in ocpn["tbr_results"]:
            place_diagnostics = ocpn["tbr_results"][object_type][0]
            transition_diagnostics = ocpn["tbr_results"][object_type][1]

        for place in net.places:
            place_id = str(wo_decoration.uuid.uuid4())
            places[place] = place_id
            place_label = " "
            place_shape = "circle"
            place_fontcolor = None
            place_fillcolor = object_type_color
            place_key = _place_key(object_type, place)
            place_colors = subprocess_styles["places"].get(place_key, ())
            place_highlight_fillcolors = subprocess_styles["place_fillcolors"].get(place_key, ())

            if place in initial_marking:
                place_label = object_type
                place_shape = "ellipse"
            elif place in final_marking:
                place_label = object_type
                place_shape = "underline"
                place_fontcolor = object_type_color
                place_fillcolor = None

            if place in place_diagnostics and place_shape == "circle":
                diagnostics = place_diagnostics[place]
                place_shape = "ellipse"
                place_label = "p=%d m=%d\nc=%d r=%d" % (
                    diagnostics["p"],
                    diagnostics["m"],
                    diagnostics["c"],
                    diagnostics["r"],
                )

            border_attributes = _node_border_attributes(place_colors)
            if place_highlight_fillcolors:
                place_fillcolor = place_highlight_fillcolors[0]

            node_kwargs = {
                "label": place_label,
                "shape": place_shape,
                "style": "filled" if place_fillcolor is not None else None,
                "fillcolor": place_fillcolor,
                "fontcolor": place_fontcolor,
            }
            _apply_node_border(node_kwargs, border_attributes)
            viz.node(place_id, **node_kwargs)

        for transition in net.transitions:
            if transition.label is not None:
                transition_map[transition] = activities_map[transition.label]
            else:
                transition_key = _transition_key(object_type, transition)
                transition_colors = subprocess_styles["transitions"].get(transition_key, ())
                transition_fillcolors = subprocess_styles["transition_fillcolors"].get(transition_key, ())
                transition_map[transition] = str(wo_decoration.uuid.uuid4())
                node_kwargs = {
                    "label": " ",
                    "shape": "box",
                    "style": "filled",
                    "fillcolor": transition_fillcolors[0] if transition_fillcolors else object_type_color,
                }
                _apply_node_border(node_kwargs, _node_border_attributes(transition_colors))
                viz.node(
                    transition_map[transition],
                    **node_kwargs,
                )

        for arc in net.arcs:
            arc_label = " "
            arc_key = _arc_key(object_type, arc)
            arc_colors = subprocess_styles["arcs"].get(arc_key, ())
            arc_color = ":".join(arc_colors) if arc_colors else object_type_color
            if isinstance(arc.source, wo_decoration.PetriNet.Place):
                is_double = (
                    arc.target.label in ocpn["double_arcs_on_activity"][object_type]
                    and ocpn["double_arcs_on_activity"][object_type][arc.target.label]
                )
                penwidth = 4.0 if is_double else 1.0
                if arc_colors:
                    penwidth = max(penwidth, 3.0)
                if arc.target in transition_diagnostics:
                    arc_label = str(transition_diagnostics[arc.target])
                viz.edge(
                    places[arc.source],
                    transition_map[arc.target],
                    color=arc_color,
                    penwidth=str(penwidth),
                    label=arc_label,
                )
            elif isinstance(arc.source, wo_decoration.PetriNet.Transition):
                is_double = (
                    arc.source.label in ocpn["double_arcs_on_activity"][object_type]
                    and ocpn["double_arcs_on_activity"][object_type][arc.source.label]
                )
                penwidth = 4.0 if is_double else 1.0
                if arc_colors:
                    penwidth = max(penwidth, 3.0)
                if arc.source in transition_diagnostics:
                    arc_label = str(transition_diagnostics[arc.source])
                viz.edge(
                    transition_map[arc.source],
                    places[arc.target],
                    color=arc_color,
                    penwidth=str(penwidth),
                    label=arc_label,
                )

    viz.attr("graph", nodesep="0.1", ranksep="0.2")
    viz.attr(rankdir=rankdir)
    viz.format = image_format.replace("html", "plain-ext")
    return viz


def _render_ocpn_image(
        ocpn,
        max_size=(1400, 700),
        object_type_colors=None,
        activity_resource_types=None,
        highlighted_activities=None,
        subprocess_components=None,
):
    if ocpn is None:
        return None

    gviz = _build_ocpn_graphviz(
        ocpn,
        object_type_colors=object_type_colors,
        activity_resource_types=activity_resource_types,
        highlighted_activities=highlighted_activities,
        subprocess_components=subprocess_components,
    )
    image = Image.open(BytesIO(gviz.pipe(format="png"))).convert("RGBA")
    image = ImageOps.contain(image, max_size)
    return image


def render_collapsed_sub_processes(
        ocpn,
        subprocess_components,
        highlight_color="#f7d7a6",
        max_size=(1400, 700),
        object_type_colors=None,
        activity_resource_types=None,
        highlighted_activities=None,
):
    collapsed_ocpn, inserted_transitions = collapse_sub_processes(ocpn, subprocess_components) #ocpn, []#
    if collapsed_ocpn is None:
        return None

    highlight_component = {
        "id": "collapsed_sub_processes",
        "color": highlight_color,
        "fillcolor": highlight_color,
        "marker": "+",
        "transition_keys": frozenset(
            _transition_key(object_type, transition)
            for object_type, transition in inserted_transitions
        ),
        "place_keys": frozenset(),
        "arc_keys": frozenset(),
    }
    rendered_subprocess_components = list(subprocess_components or [])
    if inserted_transitions:
        rendered_subprocess_components = [highlight_component]

    return _render_ocpn_image(
        collapsed_ocpn,
        max_size=max_size,
        object_type_colors=object_type_colors,
        activity_resource_types=activity_resource_types,
        highlighted_activities=highlighted_activities,
        subprocess_components=rendered_subprocess_components,
    )


def render_object_centric_region(
        ocpn,
        region,
        highlight_color="#f7d7a6",
        fillcolor="#f7d7a6",
        marker=None,
        max_size=(1400, 700),
        object_type_colors=None,
        activity_resource_types=None,
        highlighted_activities=None,
):
    if ocpn is None or region is None:
        return None

    highlight_component = build_object_centric_region_highlight(
        region,
        border_color=highlight_color,
        fillcolor=fillcolor,
        marker=marker,
    )
    return _render_ocpn_image(
        ocpn,
        max_size=max_size,
        object_type_colors=object_type_colors,
        activity_resource_types=activity_resource_types,
        highlighted_activities=highlighted_activities,
        subprocess_components=[highlight_component],
    )


def render_object_centric_regions(
        ocpn,
        regions,
        max_size=(1400, 700),
        object_type_colors=None,
        activity_resource_types=None,
        highlighted_activities=None,
):
    if ocpn is None:
        return None

    regions = list(regions or [])
    region_colors = _build_component_colors(len(regions))
    highlight_components = [
        build_object_centric_region_highlight(
            region,
            border_color=color,
            fillcolor=None,
        )
        for region, color in zip(regions, region_colors)
    ]
    return _render_ocpn_image(
        ocpn,
        max_size=max_size,
        object_type_colors=object_type_colors,
        activity_resource_types=activity_resource_types,
        highlighted_activities=highlighted_activities,
        subprocess_components=highlight_components,
    )


def _render_model_image_for_hierarchy_row(model_data, object_type_colors, max_size=(1400, 700)):
    if model_data.get("subprocess_components"):
        return render_collapsed_sub_processes(
            model_data.get("ocpn"),
            model_data.get("subprocess_components"),
            max_size=max_size,
            object_type_colors=object_type_colors,
            activity_resource_types=model_data.get("activity_resources"),
            highlighted_activities=model_data.get("highlighted_activities"),
        )

    return _render_ocpn_image(
        model_data.get("ocpn"),
        max_size=max_size,
        object_type_colors=object_type_colors,
        activity_resource_types=model_data.get("activity_resources"),
        highlighted_activities=model_data.get("highlighted_activities"),
        subprocess_components=model_data.get("subprocess_components"),
    )


def render_pruning_candidate_debug(
        with_component_ocpn,
        without_component_ocpn,
        *,
        title="Pruning Candidate Debug",
        with_component_title="With component",
        without_component_title="Without component",
        with_component_metrics=None,
        without_component_metrics=None,
        summary_metrics=None,
        max_model_size=(900, 420),
        output_path=None,
        show=False,
):
    with_component_metrics = with_component_metrics or {}
    without_component_metrics = without_component_metrics or {}
    summary_metrics = summary_metrics or {}

    title_font = _load_font(28)
    heading_font = _load_font(22)
    body_font = _load_font(18)

    with_image = _render_ocpn_image(with_component_ocpn, max_size=max_model_size)
    without_image = _render_ocpn_image(without_component_ocpn, max_size=max_model_size)

    if with_image is None:
        with_image = _placeholder_model_image("No process model", size=(max_model_size[0], 220))
    if without_image is None:
        without_image = _placeholder_model_image("No process model", size=(max_model_size[0], 220))

    dummy_image = Image.new("RGB", (10, 10), "white")
    dummy_draw = ImageDraw.Draw(dummy_image)

    def _metrics_lines(metrics):
        lines = []
        for label, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{label}: {value:.4f}")
            else:
                lines.append(f"{label}: {value}")
        return lines or ["No metrics"]

    with_lines = _metrics_lines(with_component_metrics)
    without_lines = _metrics_lines(without_component_metrics)
    summary_lines = _metrics_lines(summary_metrics)

    def _text_block_height(lines, font, spacing):
        if not lines:
            return 0
        sample_bbox = dummy_draw.textbbox((0, 0), "Ag", font=font)
        line_height = sample_bbox[3] - sample_bbox[1]
        return len(lines) * line_height + max(0, len(lines) - 1) * spacing

    outer_padding = 24
    panel_gap = 24
    section_gap = 18
    metrics_spacing = 8
    header_gap = 10
    panel_padding = 16
    panel_width = max(with_image.width, without_image.width) + 2 * panel_padding
    summary_width = 360

    title_bbox = dummy_draw.textbbox((0, 0), title, font=title_font)
    title_height = title_bbox[3] - title_bbox[1]
    heading_bbox = dummy_draw.textbbox((0, 0), with_component_title, font=heading_font)
    heading_height = heading_bbox[3] - heading_bbox[1]
    body_bbox = dummy_draw.textbbox((0, 0), "Ag", font=body_font)
    body_height = body_bbox[3] - body_bbox[1]

    with_panel_height = (
        panel_padding + heading_height + header_gap + with_image.height + section_gap
        + _text_block_height(with_lines, body_font, metrics_spacing) + panel_padding
    )
    without_panel_height = (
        panel_padding + heading_height + header_gap + without_image.height + section_gap
        + _text_block_height(without_lines, body_font, metrics_spacing) + panel_padding
    )
    summary_panel_height = (
        panel_padding + heading_height + header_gap
        + _text_block_height(summary_lines, body_font, metrics_spacing) + panel_padding
    )

    content_height = max(with_panel_height, without_panel_height, summary_panel_height)
    canvas_width = outer_padding * 2 + panel_width * 2 + summary_width + panel_gap * 2
    canvas_height = outer_padding * 2 + title_height + section_gap + content_height

    canvas = Image.new("RGBA", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((outer_padding, outer_padding), title, fill="black", font=title_font)

    panel_y = outer_padding + title_height + section_gap

    def _draw_panel(x, heading, image, lines, width):
        panel_bottom = panel_y + content_height
        draw.rounded_rectangle(
            [(x, panel_y), (x + width, panel_bottom)],
            radius=18,
            outline="black",
            width=2,
            fill="#f7f7f7",
        )
        cursor_y = panel_y + panel_padding
        draw.text((x + panel_padding, cursor_y), heading, fill="black", font=heading_font)
        cursor_y += heading_height + header_gap
        image_x = x + (width - image.width) // 2
        canvas.alpha_composite(image, (image_x, cursor_y))
        cursor_y += image.height + section_gap
        for line in lines:
            draw.text((x + panel_padding, cursor_y), line, fill="black", font=body_font)
            cursor_y += body_height + metrics_spacing

    _draw_panel(outer_padding, with_component_title, with_image, with_lines, panel_width)
    _draw_panel(
        outer_padding + panel_width + panel_gap,
        without_component_title,
        without_image,
        without_lines,
        panel_width,
    )

    summary_x = outer_padding + 2 * (panel_width + panel_gap)
    summary_bottom = panel_y + content_height
    draw.rounded_rectangle(
        [(summary_x, panel_y), (summary_x + summary_width, summary_bottom)],
        radius=18,
        outline="black",
        width=2,
        fill="#fff8eb",
    )
    cursor_y = panel_y + panel_padding
    draw.text((summary_x + panel_padding, cursor_y), "Summary", fill="black", font=heading_font)
    cursor_y += heading_height + header_gap
    for line in summary_lines:
        draw.text((summary_x + panel_padding, cursor_y), line, fill="black", font=body_font)
        cursor_y += body_height + metrics_spacing

    result = canvas.convert("RGB")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
    if show:
        plt.figure(figsize=(max(12, result.width / 150), max(7, result.height / 150)))
        plt.imshow(result)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
    return result


def _placeholder_model_image(text, size=(900, 220)):
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(28)
    body_font = _load_font(20)

    draw.rounded_rectangle(
        [(10, 10), (size[0] - 10, size[1] - 10)],
        radius=18,
        outline="black",
        width=2,
        fill="white",
    )

    title_bbox = draw.textbbox((0, 0), "No Process Model", font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(
        ((size[0] - title_width) / 2, 65),
        "No Process Model",
        fill="black",
        font=title_font,
    )

    body_bbox = draw.textbbox((0, 0), text, font=body_font)
    body_width = body_bbox[2] - body_bbox[0]
    draw.text(
        ((size[0] - body_width) / 2, 120),
        text,
        fill="black",
        font=body_font,
    )

    return image


def _wrap_object_types(object_types, width=18):
    parts = [part.strip() for part in object_types if part and part.strip()]
    if not parts:
        return [["None"]]

    wrapped_lines = []
    current_line = []
    current_length = 0

    for part in parts:
        part_options = [part] if len(part) <= width else textwrap.wrap(part, width=width)
        for option in part_options:
            separator_length = 2 if current_line else 0
            candidate_length = current_length + separator_length + len(option)
            if current_line and candidate_length > width:
                wrapped_lines.append(current_line)
                current_line = [option]
                current_length = len(option)
            else:
                current_line.append(option)
                current_length = candidate_length

    if current_line:
        wrapped_lines.append(current_line)

    return wrapped_lines


def _measure_object_type_boxes(
        draw,
        object_type_lines,
        font,
        spacing=14,
        box_padding_x=16,
        box_padding_y=10,
        box_gap_x=12,
):
    if not object_type_lines:
        return 0, 0

    sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
    text_height = sample_bbox[3] - sample_bbox[1]
    box_height = text_height + 2 * box_padding_y
    max_width = 0

    for line in object_type_lines:
        line_width = 0
        for index, object_type in enumerate(line):
            text_bbox = draw.textbbox((0, 0), object_type, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            line_width += text_width + 2 * box_padding_x
            if index < len(line) - 1:
                line_width += box_gap_x
        max_width = max(max_width, line_width)

    total_height = len(object_type_lines) * box_height + max(0, len(object_type_lines) - 1) * spacing
    return max_width, total_height


def _draw_object_type_boxes(
        draw,
        position,
        object_type_lines,
        font,
        object_type_colors,
        spacing=14,
        box_padding_x=16,
        box_padding_y=10,
        box_gap_x=12,
        box_radius=16,
):
    x_start, y = position
    sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
    text_height = sample_bbox[3] - sample_bbox[1]
    box_height = text_height + 2 * box_padding_y

    for line in object_type_lines:
        x = x_start
        for object_type in line:
            text_bbox = draw.textbbox((0, 0), object_type, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            box_width = text_width + 2 * box_padding_x
            box_fill = object_type_colors.get(object_type, "#dddddd")

            draw.rounded_rectangle(
                [(x, y), (x + box_width, y + box_height)],
                radius=box_radius,
                fill=box_fill,
                outline="black",
                width=2,
            )
            draw.text(
                (x + box_padding_x, y + box_padding_y),
                object_type,
                fill="black",
                font=font,
            )
            x += box_width + box_gap_x

        y += box_height + spacing


def visualize_hierarchy_with_models(
        solution,
        discovered_models,
        title="Hierarchy with Process Models",
        output_path=None,
):
    if not solution:
        return None

    discovered_models = discovered_models or {}

    title_font = _load_font(34)
    heading_font = _load_font(34)
    label_font = _load_font(28)
    object_font = _load_font(52)

    layers = sorted(solution.values(), reverse=True)
    layers = list(dict.fromkeys(layers))
    all_object_types = set(solution.keys())
    for model_data in discovered_models.values():
        all_object_types.update(model_data.get("object_types", []))
    object_type_colors = _build_object_type_colors(all_object_types)

    text_panel_width = 900
    outer_padding = 30
    row_padding = 24
    panel_gap = 28
    separator_gap = 10

    row_data = []
    max_model_width = 0

    dummy_image = Image.new("RGB", (10, 10), "white")
    dummy_draw = ImageDraw.Draw(dummy_image)

    for layer in layers:
        model_data = discovered_models.get(layer, {})
        object_types = model_data.get("object_types", [])
        object_type_lines = _wrap_object_types(object_types)
        label_bbox = dummy_draw.textbbox((0, 0), "Object types:", font=label_font)
        _, objects_height = _measure_object_type_boxes(
            dummy_draw,
            object_type_lines,
            font=object_font,
            spacing=14,
        )
        label_height = label_bbox[3] - label_bbox[1]
        text_height = 40 + label_height + 22 + objects_height + 50

        try:
            model_image = _render_model_image_for_hierarchy_row(
                model_data,
                object_type_colors,
                max_size=(1400, 700),
            )
        except Exception:
            model_image = None

        if model_image is None:
            model_image = _placeholder_model_image("No events for this layer")

        max_model_width = max(max_model_width, model_image.width)
        row_height = max(text_height, model_image.height)
        row_data.append({
            "layer": layer,
            "object_type_lines": object_type_lines,
            "model_image": model_image,
            "row_height": row_height,
        })

    title_bbox = dummy_draw.textbbox((0, 0), title, font=title_font)
    title_height = title_bbox[3] - title_bbox[1]

    canvas_width = outer_padding * 2 + text_panel_width + panel_gap + max_model_width
    canvas_height = outer_padding * 2 + title_height + 20
    canvas_height += sum(row["row_height"] + row_padding for row in row_data)

    canvas = Image.new("RGBA", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((outer_padding, outer_padding), title, fill="black", font=title_font)

    y = outer_padding + title_height + 20
    for row in row_data:
        row_top = y
        row_bottom = y + row["row_height"]

        draw.rounded_rectangle(
            [(outer_padding, row_top), (outer_padding + text_panel_width, row_bottom)],
            radius=18,
            outline="black",
            width=2,
            fill="#f6f6f6",
        )

        heading_y = row_top + 18
        draw.text(
            (outer_padding + 18, heading_y),
            f"Layer {row['layer']}",
            fill="black",
            font=heading_font,
        )
        draw.multiline_text(
            (outer_padding + 18, heading_y + 52),
            "Object types:",
            fill="black",
            font=label_font,
            spacing=12,
        )
        _draw_object_type_boxes(
            draw,
            (outer_padding + 18, heading_y + 92),
            row["object_type_lines"],
            object_font,
            object_type_colors,
            spacing=14,
        )

        model_x = outer_padding + text_panel_width + panel_gap
        model_y = row_top + (row["row_height"] - row["model_image"].height) // 2
        canvas.alpha_composite(row["model_image"], (model_x, model_y))

        draw.line(
            [(outer_padding, row_bottom + separator_gap), (canvas_width - outer_padding, row_bottom + separator_gap)],
            fill="#cccccc",
            width=1,
        )

        y = row_bottom + row_padding

    result = canvas.convert("RGB")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
    else:
        plt.figure(figsize=(max(10, result.width / 150), max(6, result.height / 150)))
        plt.imshow(result)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

    return result
