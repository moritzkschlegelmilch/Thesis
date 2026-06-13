#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import signal
import tempfile
from typing import Any, Callable, Iterable

from discovery import (
    CheckSet,
    CheckpointManager,
    CollapsedNetBuilder,
    GreedyOptimization,
    HierarchyQualityEvaluator,
    LayerAssignment,
    LayerAssignmentMiner,
    ModelDiscovery,
    PM4PyOCPNDiscovery,
    PrecisionCalculator,
    PrecisionParameters,
    QualityScores,
    SubprocessMiner,
)
from discovery.scorables import (
    CardinalityRelationScorer,
    DivergenceScorer,
    TimeRelationScorer,
)
from discovery.totem import clear_totem_cache
from helpers.vorbose import (
    visualize_hierarchy_with_indexed_subprocesses,
    visualize_hierarchy_with_models,
    visualize_layers_boxed,
)
from totem_lib import import_ocel


SUPPORTED_LOG_EXTENSIONS = (".sqlite", ".xml", ".xmlocel", ".json", ".jsonocel")
DEFAULT_LAYER_CONTEXT = 1
DEFAULT_PRECISION_CONTEXT_SAMPLE_SIZE = 256
DEFAULT_PRECISION_CONTEXT_DEPTH = 5
DEFAULT_PRECISION_CONTEXT_SAMPLE_SEED = None
DEFAULT_MAX_NODES_PER_REPLAY = 100


LayerContext = int | Iterable[int] | Callable[[LayerAssignment, Any, Path], Iterable[int]]


def _configure_runtime() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "repo-mpl-cache"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


def build_default_layer_miner(
    checkpoint: CheckpointManager | None = None,
    *,
    verbose: bool = False,
) -> LayerAssignmentMiner:
    return LayerAssignmentMiner(
        [
            TimeRelationScorer(1),
            CardinalityRelationScorer(1),
            DivergenceScorer(1),
        ],
        checkpoint=checkpoint,
        verbose=verbose,
    )


def build_default_model_discovery(
    checkpoint: CheckpointManager | None = None,
    *,
    verbose: bool = False,
    precision_context_sample_size: int | None = DEFAULT_PRECISION_CONTEXT_SAMPLE_SIZE,
    precision_context_depth: int | None = DEFAULT_PRECISION_CONTEXT_DEPTH,
    precision_context_sample_seed: int | None = DEFAULT_PRECISION_CONTEXT_SAMPLE_SEED,
    max_nodes_per_replay: int | None = DEFAULT_MAX_NODES_PER_REPLAY,
) -> ModelDiscovery:
    check_set = CheckSet(
        PM4PyOCPNDiscovery(checkpoint=checkpoint, verbose=verbose),
        SubprocessMiner(checkpoint=checkpoint, verbose=verbose),
        CollapsedNetBuilder(checkpoint=checkpoint, verbose=verbose),
        PrecisionCalculator(
            PrecisionParameters(
                d=precision_context_depth,
                replay_budget=max_nodes_per_replay,
                sample_size=precision_context_sample_size,
                random_seed=precision_context_sample_seed,
            ),
            checkpoint=checkpoint,
            verbose=verbose,
        ),
        use_delta=True,
        checkpoint=checkpoint,
        verbose=verbose,
    )

    return ModelDiscovery(
        PM4PyOCPNDiscovery(checkpoint=checkpoint, verbose=verbose),
        GreedyOptimization(check_set, checkpoint=checkpoint, verbose=verbose),
        subprocess_miner=SubprocessMiner(checkpoint=checkpoint, verbose=verbose),
        alpha_decision_parameter=0.5,
        checkpoint=checkpoint,
        verbose=verbose,
    )


def evaluate_event_log_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    layer_miner: LayerAssignmentMiner,
    hierarchy_miner: ModelDiscovery,
    *,
    layer_context: LayerContext = DEFAULT_LAYER_CONTEXT,
    quality_evaluator: HierarchyQualityEvaluator | None = None,
    recursive: bool = False,
    extensions: Iterable[str] = SUPPORTED_LOG_EXTENSIONS,
    verbose: bool = True,
    checkpoint_factory: Callable[[], CheckpointManager] | None = None,
    log_importer: Callable[[str], Any] = import_ocel,
    continue_on_error: bool = True,
    per_log_timeout_seconds: int | float | None = None,
) -> dict[str, Any]:
    """Evaluate all event logs in a folder with configured layer and hierarchy miners.

    For every log, this writes:
    - hierarchy.png: hierarchy visualization without indexed subprocess annotations
    - hierarchy_subprocesses.png: hierarchy visualization with indexed subprocesses
    - hierarchy_layers.png: compact layer assignment visualization
    - checkpoints.json: raw checkpoint timing records
    - quality.json: hierarchy quality metrics
    - summary.json: per-log metadata and artifact paths

    The folder itself receives summary.json, quality.csv, and checkpoint_times.csv.
    """
    _configure_runtime()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_paths = _collect_input_paths(
        input_dir,
        recursive=recursive,
        extensions=_normalize_extensions(extensions),
    )
    if not input_paths:
        raise ValueError(f"No event logs found in {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    folder_checkpoint = CheckpointManager(verbose=verbose, collect=True)
    summaries: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    checkpoint_factory = checkpoint_factory or (
        lambda: CheckpointManager(verbose=verbose, collect=True)
    )

    with folder_checkpoint.section(
        "Evaluate event log folder",
        total=len(input_paths),
        metadata={"logs": len(input_paths)},
        unit="log",
    ) as folder_span:
        for input_path in input_paths:
            log_output_dir = _log_output_dir(input_dir, output_dir, input_path)
            checkpoint = checkpoint_factory()
            try:
                with _log_timeout(per_log_timeout_seconds, input_path):
                    summary = evaluate_log(
                        input_path,
                        log_output_dir,
                        layer_miner,
                        hierarchy_miner,
                        layer_context=layer_context,
                        quality_evaluator=quality_evaluator,
                        verbose=verbose,
                        checkpoint=checkpoint,
                        log_importer=log_importer,
                    )
            except Exception as exc:
                if not continue_on_error:
                    raise
                summary = _failure_summary(
                    input_path,
                    log_output_dir,
                    exc,
                    checkpoint,
                    timeout_seconds=per_log_timeout_seconds,
                )
                _write_json(log_output_dir / "summary.json", summary)
                _write_json(log_output_dir / "checkpoints.json", summary["checkpoints"])
            summaries.append(summary)
            quality_rows.append(_quality_csv_row(summary))
            checkpoint_rows.extend(_checkpoint_csv_rows(summary))
            folder_span.update(postfix=f"{input_path.name}: {summary['status']}")

    folder_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "status": "ok" if all(summary["status"] == "ok" for summary in summaries) else "partial",
        "logs": summaries,
        "folder_checkpoints": folder_checkpoint.as_dicts(),
    }
    _write_json(output_dir / "summary.json", folder_summary)
    _write_csv(output_dir / "quality.csv", quality_rows)
    _write_csv(output_dir / "checkpoint_times.csv", checkpoint_rows)
    _write_json(output_dir / "folder_checkpoints.json", folder_checkpoint.as_dicts())

    return folder_summary


def evaluate_log(
    input_path: str | Path,
    output_dir: str | Path,
    layer_miner: LayerAssignmentMiner,
    hierarchy_miner: ModelDiscovery,
    *,
    layer_context: LayerContext = DEFAULT_LAYER_CONTEXT,
    quality_evaluator: HierarchyQualityEvaluator | None = None,
    verbose: bool = True,
    checkpoint: CheckpointManager | None = None,
    log_importer: Callable[[str], Any] = import_ocel,
) -> dict[str, Any]:
    """Evaluate a single event log and write all thesis evaluation artifacts."""
    _configure_runtime()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint or CheckpointManager(verbose=verbose, collect=True)
    _attach_checkpoint(layer_miner, checkpoint)
    _attach_checkpoint(hierarchy_miner, checkpoint)
    if quality_evaluator is not None:
        _attach_checkpoint(quality_evaluator, checkpoint)

    ocel = None
    try:
        with checkpoint.section(
            "Evaluate event log",
            total=6,
            metadata={"log": input_path.name},
        ) as span:
            with span.child("Import event log", metadata={"path": str(input_path)}):
                ocel = log_importer(str(input_path))
            log_stats = _summarize_ocel(ocel)
            span.update(postfix=f"events={log_stats.get('events')}")

            layer_assignment = layer_miner.mine(ocel)
            layer_stats = _summarize_layer_assignment(layer_assignment)
            span.update(postfix=f"layers={layer_stats.get('layers')}")

            delta = _resolve_layer_context(layer_context, layer_assignment, ocel, input_path)
            hierarchy = hierarchy_miner.mine(ocel, layer_assignment, delta)
            hierarchy_stats = _summarize_hierarchy(hierarchy)
            span.update(postfix=f"areas={hierarchy_stats.get('areas')}")

            evaluator = quality_evaluator or _derive_quality_evaluator(
                hierarchy_miner,
                checkpoint=checkpoint,
                verbose=verbose,
            )
            quality_log = _quality_log(ocel)
            quality = evaluator.evaluate(quality_log, hierarchy)
            quality_data = _quality_to_dict(quality)
            span.update(postfix=f"quality={quality.quality:.3f}")

            artifacts = _render_artifacts(input_path, output_dir, hierarchy)
            span.update(postfix="visualizations saved")

            summary = {
                "status": "ok",
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "log": log_stats,
                "layer_assignment": {
                    "object_to_layer": layer_assignment.as_object_to_layer(),
                    **layer_stats,
                },
                "layer_context": list(delta),
                "hierarchy": hierarchy_stats,
                "quality": quality_data,
                "artifacts": {
                    key: str(value)
                    for key, value in artifacts.items()
                },
            }
            span.update(postfix="summary prepared")

        checkpoints = checkpoint.as_dicts()
        summary["checkpoints"] = checkpoints

        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "quality.json", summary["quality"])
        _write_json(output_dir / "checkpoints.json", checkpoints)
        _write_json(output_dir / "layer_assignment.json", summary["layer_assignment"])

        return summary
    finally:
        if ocel is not None:
            clear_totem_cache(ocel)


def _render_artifacts(input_path: Path, output_dir: Path, hierarchy) -> dict[str, Path]:
    title = input_path.stem
    artifacts = {
        "layers": output_dir / "hierarchy_layers.png",
        "hierarchy": output_dir / "hierarchy.png",
        "hierarchy_subprocesses": output_dir / "hierarchy_subprocesses.png",
    }
    visualize_layers_boxed(
        hierarchy,
        title=f"{title} Layers",
        output_path=artifacts["layers"],
    )
    visualize_hierarchy_with_models(
        hierarchy,
        title=f"{title} Hierarchy",
        output_path=artifacts["hierarchy"],
    )
    visualize_hierarchy_with_indexed_subprocesses(
        hierarchy,
        title=f"{title} Hierarchy with Subprocesses",
        output_path=artifacts["hierarchy_subprocesses"],
    )
    return artifacts


def _quality_log(log):
    from discovery.discovery_preparation import (
        _build_ocel_filtering_context,
        _build_ocel_from_filtering_context,
    )

    return _build_ocel_from_filtering_context(_build_ocel_filtering_context(log))


def _failure_summary(
    input_path: Path,
    output_dir: Path,
    exc: Exception,
    checkpoint: CheckpointManager,
    *,
    timeout_seconds: int | float | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "failed",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "checkpoints": checkpoint.as_dicts(),
    }
    if timeout_seconds is not None:
        summary["timeout_seconds"] = timeout_seconds
    return summary


@contextmanager
def _log_timeout(seconds: int | float | None, input_path: Path):
    if seconds is None or seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    seconds = float(seconds)
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    except ValueError:
        yield
        return

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"Event log {input_path.name!r} exceeded {seconds:g} seconds.")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _derive_quality_evaluator(
    hierarchy_miner: ModelDiscovery,
    *,
    checkpoint: CheckpointManager,
    verbose: bool,
) -> HierarchyQualityEvaluator:
    optimization_function = getattr(hierarchy_miner, "optimization_function", None)
    check_set = getattr(optimization_function, "check_set", None)
    if check_set is None:
        raise ValueError(
            "quality_evaluator is required when the hierarchy miner does not expose "
            "a check_set through its optimization function."
        )

    evaluator = HierarchyQualityEvaluator(
        check_set.discovery_technique,
        check_set.collapsed_net_builder,
        check_set.precision_calculator,
        checkpoint=checkpoint,
        verbose=verbose,
    )
    _attach_checkpoint(evaluator, checkpoint)
    return evaluator


def _attach_checkpoint(component: Any, checkpoint: CheckpointManager, seen: set[int] | None = None) -> None:
    if component is None:
        return
    if seen is None:
        seen = set()
    component_id = id(component)
    if component_id in seen:
        return
    seen.add(component_id)

    if hasattr(component, "checkpoint"):
        component.checkpoint = checkpoint

    for attribute in (
        "resource_indicators",
        "discovery_technique",
        "optimization_function",
        "check_set",
        "subprocess_miner",
        "collapsed_net_builder",
        "precision_calculator",
    ):
        if not hasattr(component, attribute):
            continue
        value = getattr(component, attribute)
        if _is_simple_value(value):
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                _attach_checkpoint(nested, checkpoint, seen)
        else:
            _attach_checkpoint(value, checkpoint, seen)


def _is_simple_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, bytes, int, float, bool, Path))


def _normalize_extensions(extensions: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
        if extension
    }))


def _collect_input_paths(
    input_dir: Path,
    *,
    recursive: bool,
    extensions: tuple[str, ...],
) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def _log_output_dir(input_dir: Path, output_dir: Path, input_path: Path) -> Path:
    try:
        relative = input_path.relative_to(input_dir)
    except ValueError:
        relative = Path(input_path.name)

    parts = list(relative.with_suffix("").parts)
    safe_name = "__".join(_safe_path_part(part) for part in parts)
    return output_dir / safe_name


def _safe_path_part(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _resolve_layer_context(
    layer_context: LayerContext,
    layer_assignment: LayerAssignment,
    ocel,
    input_path: Path,
) -> list[int]:
    if callable(layer_context):
        resolved = list(layer_context(layer_assignment, ocel, input_path))
    elif isinstance(layer_context, int):
        resolved = [layer_context] * len(layer_assignment.layers)
    else:
        resolved = list(layer_context)

    if len(resolved) != len(layer_assignment.layers):
        raise ValueError(
            "layer_context must contain exactly one value per discovered layer "
            f"({len(layer_assignment.layers)} expected, {len(resolved)} given)."
        )
    return [int(value) for value in resolved]


def _summarize_ocel(ocel) -> dict[str, Any]:
    return {
        "events": _count_rows(getattr(ocel, "events", None)),
        "objects": _count_rows(getattr(ocel, "objects", None)),
        "relations": _count_rows(getattr(ocel, "relations", None)),
        "object_types": len(getattr(ocel, "object_types", ()) or ()),
    }


def _count_rows(frame) -> int | None:
    if frame is None:
        return None
    try:
        return int(len(frame))
    except TypeError:
        return None


def _summarize_layer_assignment(layer_assignment: LayerAssignment) -> dict[str, Any]:
    return {
        "layers": len(layer_assignment.layers),
        "object_types": sum(len(layer) for layer in layer_assignment.layers),
    }


def _summarize_hierarchy(hierarchy) -> dict[str, Any]:
    return {
        "areas": len(hierarchy.areas),
        "object_types": len(hierarchy.object_types()),
        "activities": len(hierarchy.activities()),
        "subprocesses": sum(len(area.subprocesses or ()) for area in hierarchy.areas),
        "areas_detail": [
            {
                "layer": layer,
                "object_types": sorted(area.object_types),
                "activities": sorted(area.activities),
                "activity_count": len(area.activities),
                "subprocess_count": len(area.subprocesses or ()),
            }
            for layer, area in enumerate(hierarchy.areas, start=1)
        ],
    }


def _quality_to_dict(quality: QualityScores) -> dict[str, float]:
    return {
        "simplicity_gain": float(quality.simplicity_gain),
        "information_loss": float(quality.information_loss),
        "quality": float(quality.quality),
        "precision": float(quality.precision),
        "complexity": float(quality.complexity),
    }


def _quality_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    hierarchy = summary.get("hierarchy", {})
    return {
        "status": summary.get("status"),
        "timeout_seconds": summary.get("timeout_seconds"),
        "error_type": summary.get("error", {}).get("type"),
        "error_message": summary.get("error", {}).get("message"),
        "log": Path(summary["input_path"]).name,
        "input_path": summary["input_path"],
        **summary.get("log", {}),
        "areas": hierarchy.get("areas"),
        "hierarchy_object_types": hierarchy.get("object_types"),
        "activities": hierarchy.get("activities"),
        "subprocesses": hierarchy.get("subprocesses"),
        **summary.get("quality", {}),
    }


def _checkpoint_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in summary.get("checkpoints", ()):
        rows.append({
            "log": Path(summary["input_path"]).name,
            "input_path": summary["input_path"],
            "name": record.get("name"),
            "parent": record.get("parent"),
            "status": record.get("status"),
            "duration_seconds": record.get("duration_seconds"),
            "count": record.get("count"),
            "total": record.get("total"),
            "metadata": json.dumps(record.get("metadata", {}), sort_keys=True, default=str),
        })
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every OCEL event log in a folder and save hierarchy images, "
            "checkpoint timings, and paper-style quality metrics."
        ),
    )
    parser.add_argument("input_dir", help="Directory containing event logs.")
    parser.add_argument("output_dir", help="Directory for evaluation artifacts.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input_dir.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(SUPPORTED_LOG_EXTENSIONS),
        help="Included file extensions.",
    )
    parser.add_argument(
        "--layer-context",
        type=int,
        default=DEFAULT_LAYER_CONTEXT,
        help="Single delta/context value used for every discovered layer.",
    )
    parser.add_argument(
        "--precision-context-sample-size",
        type=int,
        default=DEFAULT_PRECISION_CONTEXT_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--precision-context-depth",
        type=int,
        default=DEFAULT_PRECISION_CONTEXT_DEPTH,
    )
    parser.add_argument(
        "--precision-context-sample-seed",
        type=int,
        default=DEFAULT_PRECISION_CONTEXT_SAMPLE_SEED,
    )
    parser.add_argument(
        "--max-nodes-per-replay",
        type=int,
        default=DEFAULT_MAX_NODES_PER_REPLAY,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the folder evaluation on the first failing log.",
    )
    parser.add_argument(
        "--per-log-timeout-minutes",
        type=float,
        help="Mark one log as failed after this many minutes and continue with the next log.",
    )
    parser.add_argument(
        "--per-log-timeout-seconds",
        type=float,
        help="Mark one log as failed after this many seconds and continue with the next log.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    verbose = not args.no_progress
    checkpoint = CheckpointManager(verbose=verbose, collect=True)
    timeout_seconds = _resolve_timeout_seconds(
        args.per_log_timeout_seconds,
        args.per_log_timeout_minutes,
    )
    return evaluate_event_log_folder(
        args.input_dir,
        args.output_dir,
        build_default_layer_miner(checkpoint=checkpoint, verbose=verbose),
        build_default_model_discovery(
            checkpoint=checkpoint,
            verbose=verbose,
            precision_context_sample_size=args.precision_context_sample_size,
            precision_context_depth=args.precision_context_depth,
            precision_context_sample_seed=args.precision_context_sample_seed,
            max_nodes_per_replay=args.max_nodes_per_replay,
        ),
        layer_context=args.layer_context,
        recursive=args.recursive,
        extensions=args.extensions,
        verbose=verbose,
        continue_on_error=not args.fail_fast,
        per_log_timeout_seconds=timeout_seconds,
    )


def _resolve_timeout_seconds(seconds: float | None, minutes: float | None) -> float | None:
    if seconds is not None and minutes is not None:
        raise ValueError("Use either --per-log-timeout-seconds or --per-log-timeout-minutes, not both.")
    if seconds is not None:
        return seconds
    if minutes is not None:
        return minutes * 60
    return None


if __name__ == "__main__":
    main()
