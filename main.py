import json
from pathlib import Path

from discovery import (
    CheckSet,
    CheckpointManager,
    CollapsedNetBuilder,
    GreedyOptimization,
    HierarchyQualityEvaluator,
    LayerAssignmentMiner,
    ModelDiscovery,
    PM4PyOCPNDiscovery,
    PrecisionCalculator,
    PrecisionParameters,
    SubprocessMiner, SubprocessMoveUpOptimization,
)
from discovery.scorables import (
    CardinalityRelationScorer,
    DivergenceScorer,
    TimeRelationScorer,
)
from helpers.vorbose import (
    visualize_hierarchy_with_indexed_subprocesses,
    visualize_hierarchy_with_models,
    visualize_layers_boxed,
)
from totem_lib import import_ocel
from discovery.discovery_preparation import (
    _build_ocel_filtering_context,
    _build_ocel_from_filtering_context,
)


DEFAULT_INPUT_PATH = "all_logs/ContainerLogistics.sqlite"
DEFAULT_LAYER_CONTEXT = 4
PRECISION_CONTEXT_SAMPLE_SIZE = 512
PRECISION_CONTEXT_DEPTH = None
PRECISION_CONTEXT_SAMPLE_SEED = None
MAX_NODES_PER_REPLAY = 128
DEFAULT_LAYERS_OUTPUT = "output/hierarchy_layers.png"
DEFAULT_MODEL_OUTPUT = "output/hierarchy_with_models.png"
DEFAULT_SUBPROCESS_OUTPUT = "output/hierarchy_with_subprocesses.png"
DEFAULT_QUALITY_OUTPUT = "output/quality.json"


def build_model_discovery(checkpoint, *, verbose=False):
    check_set = CheckSet(
        PM4PyOCPNDiscovery(checkpoint=checkpoint, verbose=verbose),
        SubprocessMiner(checkpoint=checkpoint, verbose=verbose),
        CollapsedNetBuilder(checkpoint=checkpoint, verbose=verbose),
        PrecisionCalculator(
            PrecisionParameters(
                d=PRECISION_CONTEXT_DEPTH,
                replay_budget=MAX_NODES_PER_REPLAY,
                sample_size=PRECISION_CONTEXT_SAMPLE_SIZE,
                random_seed=PRECISION_CONTEXT_SAMPLE_SEED,
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


def build_quality_evaluator(checkpoint, *, verbose=False):
    return HierarchyQualityEvaluator(
        PM4PyOCPNDiscovery(checkpoint=checkpoint, verbose=verbose),
        CollapsedNetBuilder(checkpoint=checkpoint, verbose=verbose),
        PrecisionCalculator(
            PrecisionParameters(
                d=PRECISION_CONTEXT_DEPTH,
                replay_budget=MAX_NODES_PER_REPLAY,
                sample_size=PRECISION_CONTEXT_SAMPLE_SIZE,
                random_seed=PRECISION_CONTEXT_SAMPLE_SEED,
            ),
            checkpoint=checkpoint,
            verbose=verbose,
        ),
        checkpoint=checkpoint,
        verbose=verbose,
    )


def quality_to_dict(quality):
    return {
        "simplicity_gain": float(quality.simplicity_gain),
        "information_loss": float(quality.information_loss),
        "quality": float(quality.quality),
        "precision": float(quality.precision),
        "complexity": float(quality.complexity),
    }


def write_quality(path, quality_data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(quality_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def quality_log(log):
    return _build_ocel_from_filtering_context(_build_ocel_filtering_context(log))


def main():
    verbose = True
    checkpoint = CheckpointManager(verbose=verbose, collect=True)
    ocel = import_ocel(DEFAULT_INPUT_PATH)
    layer_assignment = LayerAssignmentMiner(
        [
            TimeRelationScorer(1),
            CardinalityRelationScorer(1),
            DivergenceScorer(1),
        ],
        checkpoint=checkpoint,
        verbose=verbose,
        debug_matrices=True,
        alpha=1.0,
        beta=0.5
    ).mine(ocel)
    hierarchy = build_model_discovery(
        checkpoint,
        verbose=verbose,
    ).mine(
        ocel,
        layer_assignment,
        [DEFAULT_LAYER_CONTEXT] * len(layer_assignment.layers),
    )

    visualize_layers_boxed(hierarchy, title="Hierarchy Layers", output_path=DEFAULT_LAYERS_OUTPUT)
    visualize_hierarchy_with_models(
        hierarchy,
        title="Hierarchy with Process Models",
        output_path=DEFAULT_MODEL_OUTPUT,
    )
    visualize_hierarchy_with_indexed_subprocesses(
        hierarchy,
        title="Hierarchy with Indexed Subprocesses",
        output_path=DEFAULT_SUBPROCESS_OUTPUT,
    )
    quality = build_quality_evaluator(
        checkpoint,
        verbose=verbose,
    ).evaluate(
        quality_log(ocel),
        hierarchy,
    )
    quality_data = quality_to_dict(quality)
    write_quality(DEFAULT_QUALITY_OUTPUT, quality_data)
    print(
        "Quality: "
        f"{quality_data['quality']:.3f} "
        f"(precision={quality_data['precision']:.3f}, "
        f"simplicity_gain={quality_data['simplicity_gain']:.3f}, "
        f"information_loss={quality_data['information_loss']:.3f}, "
        f"complexity={quality_data['complexity']:.1f})"
    )

    return hierarchy


if __name__ == "__main__":
    main()
