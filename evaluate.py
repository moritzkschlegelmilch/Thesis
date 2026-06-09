#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
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


def _count_rows(frame):
    if frame is None:
        return None

    try:
        return int(len(frame))
    except TypeError:
        return None


def _format_duration(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"

    total_minutes, remainder = divmod(seconds, 60)
    if total_minutes < 60:
        return f"{int(total_minutes)}m {remainder:04.1f}s"

    total_hours, minutes = divmod(int(total_minutes), 60)
    return f"{total_hours}h {minutes:02d}m"


def _format_status_metrics(metrics):
    if not metrics:
        return ""

    return ", ".join(
        f"{key}={value}"
        for key, value in metrics.items()
        if value is not None
    )


def _summarize_ocel(ocel):
    return {
        "events": _count_rows(getattr(ocel, "events", None)),
        "objects": _count_rows(getattr(ocel, "objects", None)),
        "relations": _count_rows(getattr(ocel, "relations", None)),
        "object_types": len(getattr(ocel, "object_types", ()) or ()),
    }


def _summarize_solution(solution):
    distinct_layers = sorted(set(solution.values()))
    return {
        "layers": len(distinct_layers),
        "object_types": len(solution),
    }


def _summarize_discovered_models(discovered_models):
    discovered_models = discovered_models or {}
    return {
        "models": len(discovered_models),
        "activities": sum(
            len(model_data.get("activities", ()))
            for model_data in discovered_models.values()
        ),
        "highlighted": sum(
            len(model_data.get("highlighted_activities", ()))
            for model_data in discovered_models.values()
        ),
        "subprocesses": sum(
            len(model_data.get("subprocess_components", ()))
            for model_data in discovered_models.values()
        ),
    }


class _EvaluationStatusDisplay:
    def __init__(self, run_label, *, total_stages, enabled):
        self.enabled = enabled
        self.run_started_at = perf_counter()
        self.current_stage_name = None
        self.current_stage_started_at = None
        self.stage_records = []
        self.stage_bar = None

        if enabled:
            self.stage_bar = tqdm(
                total=total_stages,
                desc=f"Evaluate {run_label}",
                dynamic_ncols=True,
                file=sys.stdout,
            )

    def _write(self, message):
        if self.stage_bar is not None:
            tqdm.write(
                f"[{_format_duration(perf_counter() - self.run_started_at)}] {message}",
                file=sys.stdout,
            )

    def start_stage(self, name, *, detail=None):
        self.current_stage_name = name
        self.current_stage_started_at = perf_counter()

        if self.stage_bar is not None:
            self.stage_bar.set_postfix_str(
                f"{name}: {detail or 'running'}",
                refresh=False,
            )

    def complete_stage(self, *, metrics=None):
        if self.current_stage_name is None or self.current_stage_started_at is None:
            return

        elapsed_seconds = perf_counter() - self.current_stage_started_at
        record = {
            "name": self.current_stage_name,
            "seconds": elapsed_seconds,
            "metrics": {
                str(key): value
                for key, value in (metrics or {}).items()
                if value is not None
            },
        }
        self.stage_records.append(record)

        if self.stage_bar is not None:
            self.stage_bar.update(1)
            metrics_text = _format_status_metrics(record["metrics"])
            self.stage_bar.set_postfix_str(
                f"{record['name']} completed in {_format_duration(elapsed_seconds)}",
                refresh=False,
            )
            message = f"{record['name']} completed in {_format_duration(elapsed_seconds)}"
            if metrics_text:
                message += f" | {metrics_text}"
            self._write(message)

        self.current_stage_name = None
        self.current_stage_started_at = None

    def note(self, message):
        self._write(message)

    def fail(self, exc):
        stage_name = self.current_stage_name or "Evaluation"
        self._write(
            f"{stage_name} failed with {type(exc).__name__}: {exc}"
        )

    def finalize(self, *, stats=None):
        total_seconds = perf_counter() - self.run_started_at
        evaluation = {
            "total_seconds": total_seconds,
            "stages": self.stage_records,
        }
        if stats:
            evaluation["stats"] = {
                str(key): value
                for key, value in stats.items()
                if value is not None
            }

        if self.stage_bar is not None:
            self.stage_bar.set_postfix_str(
                f"done in {_format_duration(total_seconds)}",
                refresh=False,
            )
            summary_text = _format_status_metrics(evaluation.get("stats"))
            message = f"Completed evaluation in {_format_duration(total_seconds)}"
            if summary_text:
                message += f" | {summary_text}"
            self._write(message)

        return evaluation

    def close(self):
        if self.stage_bar is not None:
            self.stage_bar.close()


def _build_summary(
    input_path,
    output_path,
    solution,
    discovered_models,
    *,
    layers_output=None,
    subprocess_output=None,
    summary_output=None,
    evaluation=None,
):
    summary = {
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
    if layers_output is not None:
        summary["layers_output"] = str(layers_output)
    if subprocess_output is not None:
        summary["subprocess_output"] = str(subprocess_output)
    if summary_output is not None:
        summary["summary_output"] = str(summary_output)
    if evaluation is not None:
        summary["evaluation"] = evaluation
    return summary


def evaluate_log(
    input_path,
    output_path,
    *,
    title=None,
    layer_context=None,
    precision_context_sample_size=512,
    precision_context_depth=5,
    precision_context_sample_seed=None,
    layers_output=None,
    subprocess_output=None,
    summary_output=None,
    show_progress=True,
):
    _configure_runtime()

    from totem_lib import import_ocel

    from repo.discovery.algorithm import ProcessAreaDiscovery
    from repo.discovery.totem import clear_totem_cache

    input_path = Path(input_path)
    output_path = Path(output_path)
    layers_output = Path(layers_output) if layers_output is not None else None
    subprocess_output = Path(subprocess_output) if subprocess_output is not None else None
    summary_output = Path(summary_output) if summary_output is not None else None
    ocel = None
    status_display = _EvaluationStatusDisplay(
        input_path.name,
        total_stages=4,
        enabled=show_progress,
    )

    try:
        status_display.start_stage("Import log", detail=input_path.name)
        ocel = import_ocel(str(input_path))
        log_stats = _summarize_ocel(ocel)
        status_display.complete_stage(metrics=log_stats)

        discovery = ProcessAreaDiscovery(ocel)

        status_display.start_stage(
            "Discover layers",
            detail=f"{log_stats.get('object_types', 0)} object types",
        )
        solution = discovery.get_layers(show_progress=show_progress)
        layer_stats = _summarize_solution(solution)
        status_display.complete_stage(metrics=layer_stats)

        status_display.start_stage(
            "Discover models",
            detail=f"{layer_stats.get('layers', 0)} layers",
        )
        discovery.discover_models(
            layer_context=layer_context,
            show_progress=show_progress,
            precision_context_sample_size=precision_context_sample_size,
            precision_context_depth=precision_context_depth,
            precision_context_sample_seed=precision_context_sample_seed,
        )
        model_stats = _summarize_discovered_models(discovery.discovered_models or {})
        status_display.complete_stage(metrics=model_stats)

        status_display.start_stage("Render outputs", detail=output_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if layers_output is not None:
            layers_output.parent.mkdir(parents=True, exist_ok=True)
            discovery.visualize_layers(
                title=f"{title or input_path.stem} Layers",
                output_path=layers_output,
            )
        if subprocess_output is not None:
            subprocess_output.parent.mkdir(parents=True, exist_ok=True)
            discovery.visualize_subprocesses(
                title=f"{title or input_path.stem} Indexed Subprocesses",
                output_path=subprocess_output,
            )

        discovery.visualize(
            title=title or input_path.stem,
            output_path=output_path,
        )
        status_display.complete_stage(
            metrics={
                "artifacts": (
                    1
                    + int(layers_output is not None)
                    + int(subprocess_output is not None)
                )
            }
        )
        status_display.note(f"Visualization saved to {output_path}")
        if layers_output is not None:
            status_display.note(f"Layer visualization saved to {layers_output}")
        if subprocess_output is not None:
            status_display.note(f"Indexed subprocess visualization saved to {subprocess_output}")

        evaluation = status_display.finalize(
            stats={
                **log_stats,
                **layer_stats,
                **model_stats,
            }
        )

        summary = _build_summary(
            input_path,
            output_path,
            solution,
            discovery.discovered_models or {},
            layers_output=layers_output,
            subprocess_output=subprocess_output,
            summary_output=summary_output,
            evaluation=evaluation,
        )

        if summary_output is not None:
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            status_display.note(f"Summary JSON saved to {summary_output}")

        return summary
    except Exception as exc:
        status_display.fail(exc)
        raise
    finally:
        status_display.close()
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
        "--precision-context-sample-size",
        type=int,
        default=512,
        help="PPS-with-replacement sample size for precision contexts during pruning. Defaults to 512.",
    )
    parser.add_argument(
        "--precision-context-depth",
        type=int,
        default=5,
        help="Maximum predecessor depth in the event-object graph when preparing precision contexts. Defaults to 5.",
    )
    parser.add_argument(
        "--precision-context-sample-seed",
        type=int,
        help="Optional random seed for precision context sampling.",
    )
    parser.add_argument(
        "--layers-output",
        help="Optional path for a separate layer-only visualization image.",
    )
    parser.add_argument(
        "--subprocess-output",
        help="Optional path for a separate indexed-subprocess visualization image.",
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
        precision_context_sample_size=args.precision_context_sample_size,
        precision_context_depth=args.precision_context_depth,
        precision_context_sample_seed=args.precision_context_sample_seed,
        layers_output=args.layers_output,
        subprocess_output=args.subprocess_output,
        summary_output=args.summary_output,
        show_progress=not args.no_progress,
    )

    print(f"Processed log: {summary['input_path']}")
    print(f"Visualization: {summary['output_path']}")
    if "layers_output" in summary:
        print(f"Layer visualization: {summary['layers_output']}")
    if "subprocess_output" in summary:
        print(f"Indexed subprocess visualization: {summary['subprocess_output']}")
    if "summary_output" in summary:
        print(f"Summary JSON: {summary['summary_output']}")
    evaluation = summary.get("evaluation")
    if evaluation is not None:
        print(f"Total runtime: {_format_duration(evaluation.get('total_seconds'))}")
        print("Stage timings:")
        for stage in evaluation.get("stages", ()):
            stage_line = (
                f"  {stage.get('name')}: "
                f"{_format_duration(stage.get('seconds'))}"
            )
            metrics_text = _format_status_metrics(stage.get("metrics"))
            if metrics_text:
                stage_line += f" | {metrics_text}"
            print(stage_line)
    print("Layers:")
    for object_type, layer in sorted(summary["solution"].items(), key=lambda item: (item[1], item[0])):
        print(f"  {object_type}: {layer}")


if __name__ == "__main__":
    main()
