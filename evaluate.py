#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
REPO_PARENT = REPO_ROOT.parent

if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))


def _configure_runtime():
    os.environ.setdefault("MPLBACKEND", "Agg")

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "repo-mpl-cache"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


def _build_summary(input_path, output_path, solution, discovered_models):
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "solution": {
            str(object_type): int(layer)
            for object_type, layer in solution.items()
        },
        "models": {
            str(layer): {
                "object_types": list(model_data.get("object_types", [])),
                "activities": list(model_data.get("activities", [])),
                "highlighted_activities": list(model_data.get("highlighted_activities", [])),
                "activity_resources": {
                    str(activity): list(resource_types)
                    for activity, resource_types in model_data.get("activity_resources", {}).items()
                },
                "subprocess_component_ids": [
                    component.get("id")
                    for component in model_data.get("subprocess_components", [])
                    if component.get("id") is not None
                ],
            }
            for layer, model_data in discovered_models.items()
        },
    }


def evaluate_log(
    input_path,
    output_path,
    *,
    title=None,
    layer_context=None,
    layers_output=None,
    summary_output=None,
    show_progress=True,
):
    _configure_runtime()

    from totem_lib import import_ocel

    from repo.discovery.algorithm import ProcessAreaDiscovery
    from repo.discovery.totem import clear_totem_cache

    input_path = Path(input_path)
    output_path = Path(output_path)
    ocel = None
    stage_bar = None

    try:
        if show_progress:
            stage_bar = tqdm(
                total=4,
                desc="Evaluating log",
                file=sys.stdout,
            )

        ocel = import_ocel(str(input_path))
        if stage_bar is not None:
            stage_bar.set_postfix_str("imported")
            stage_bar.update(1)

        discovery = ProcessAreaDiscovery(ocel)

        solution = discovery.get_layers(show_progress=show_progress)
        if stage_bar is not None:
            stage_bar.set_postfix_str("layers")
            stage_bar.update(1)

        discovery.discover_models(layer_context=layer_context, show_progress=show_progress)
        if stage_bar is not None:
            stage_bar.set_postfix_str("models")
            stage_bar.update(1)

        if layers_output is not None:
            discovery.visualize_layers(
                title=f"{title or input_path.stem} Layers",
                output_path=layers_output,
            )

        discovery.visualize(
            title=title or input_path.stem,
            output_path=output_path,
        )
        if stage_bar is not None:
            stage_bar.set_postfix_str("visualized")
            stage_bar.update(1)

        summary = _build_summary(
            input_path,
            output_path,
            solution,
            discovery.discovered_models or {},
        )

        if summary_output is not None:
            summary_output = Path(summary_output)
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return summary
    finally:
        if stage_bar is not None:
            stage_bar.close()
        if ocel is not None:
            clear_totem_cache(ocel)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run process-area discovery for one OCEL log and save the visualization.",
    )
    parser.add_argument("input_path", help="Path to the input OCEL log.")
    parser.add_argument("output_path", help="Path to the output visualization image.")
    parser.add_argument(
        "--title",
        help="Optional title for the visualization. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--layer-context",
        type=int,
        nargs="*",
        help="Optional per-layer context values passed to discover_models().",
    )
    parser.add_argument(
        "--layers-output",
        help="Optional path for a separate layer-only visualization image.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional path for a JSON summary of the discovered hierarchy.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = evaluate_log(
        args.input_path,
        args.output_path,
        title=args.title,
        layer_context=args.layer_context,
        layers_output=args.layers_output,
        summary_output=args.summary_output,
        show_progress=not args.no_progress,
    )

    print(f"Processed log: {summary['input_path']}")
    print(f"Visualization: {summary['output_path']}")
    print("Layers:")
    for object_type, layer in sorted(summary["solution"].items(), key=lambda item: (item[1], item[0])):
        print(f"  {object_type}: {layer}")


if __name__ == "__main__":
    main()
