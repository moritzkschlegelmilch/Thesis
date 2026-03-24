import pandas as pd

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


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_layers_boxed(solution, title="Hierarchy Layers"):
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
    plt.show()