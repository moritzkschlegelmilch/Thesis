from collections import defaultdict
from io import BytesIO
from pathlib import Path
import textwrap

import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pm4py.visualization.ocel.ocpn import visualizer as ocpn_visualizer


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


def _render_ocpn_image(ocpn, max_size=(1400, 700)):
    if ocpn is None:
        return None

    gviz = ocpn_visualizer.apply(ocpn)
    image = Image.open(BytesIO(gviz.pipe(format="png"))).convert("RGBA")
    image = ImageOps.contain(image, max_size)
    return image


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


def _wrap_object_types_text(text, width=18):
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return "None"

    wrapped_lines = []
    current_line = ""
    for part in parts:
        candidate = part if not current_line else f"{current_line}, {part}"
        if len(candidate) <= width:
            current_line = candidate
        else:
            if current_line:
                wrapped_lines.append(current_line)
            if len(part) <= width:
                current_line = part
            else:
                wrapped_lines.extend(textwrap.wrap(part, width=width))
                current_line = ""

    if current_line:
        wrapped_lines.append(current_line)

    return "\n".join(wrapped_lines)


def visualize_hierarchy_with_models(
        solution,
        discovered_models,
        title="Hierarchy with Process Models",
        output_path=None,
):
    if not solution:
        return None

    title_font = _load_font(34)
    heading_font = _load_font(34)
    label_font = _load_font(28)
    object_font = _load_font(52)

    layers = sorted(solution.values(), reverse=True)
    layers = list(dict.fromkeys(layers))

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
        object_types_text = _wrap_object_types_text(", ".join(object_types) if object_types else "None")
        label_bbox = dummy_draw.textbbox((0, 0), "Object types:", font=label_font)
        objects_bbox = dummy_draw.multiline_textbbox(
            (0, 0),
            object_types_text,
            font=object_font,
            spacing=14,
        )
        label_height = label_bbox[3] - label_bbox[1]
        objects_height = objects_bbox[3] - objects_bbox[1]
        text_height = 40 + label_height + 22 + objects_height + 50

        try:
            model_image = _render_ocpn_image(model_data.get("ocpn"))
        except Exception:
            model_image = None

        if model_image is None:
            model_image = _placeholder_model_image("No events for this layer")

        max_model_width = max(max_model_width, model_image.width)
        row_height = max(text_height, model_image.height)
        row_data.append({
            "layer": layer,
            "object_types_text": object_types_text,
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
        draw.multiline_text(
            (outer_padding + 18, heading_y + 92),
            row["object_types_text"],
            fill="black",
            font=object_font,
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
