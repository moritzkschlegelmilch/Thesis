from discovery import (
    CheckSet,
    CheckpointManager,
    CollapsedNetBuilder,
    GreedyOptimization,
    LayerAssignmentMiner,
    ModelDiscovery,
    PM4PyOCPNDiscovery,
    PrecisionCalculator,
    PrecisionParameters,
    SubprocessMiner,
    SubprocessMoveUpOptimization,
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


DEFAULT_INPUT_PATH = "example_data/ContainerLogistics.sqlite"
DEFAULT_LAYER_CONTEXT = 1
PRECISION_CONTEXT_SAMPLE_SIZE = 256
PRECISION_CONTEXT_DEPTH = 5
PRECISION_CONTEXT_SAMPLE_SEED = None
MAX_NODES_PER_REPLAY = 100
DEFAULT_LAYERS_OUTPUT = "output/hierarchy_layers.png"
DEFAULT_MODEL_OUTPUT = "output/hierarchy_with_models.png"
DEFAULT_SUBPROCESS_OUTPUT = "output/hierarchy_with_subprocesses.png"


def build_model_discovery(checkpoint, *, verbose=False, optimization="greedy"):
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
    optimizer = {
        "greedy": GreedyOptimization,
        "subprocess-move-up": SubprocessMoveUpOptimization,
    }[optimization]

    return ModelDiscovery(
        PM4PyOCPNDiscovery(checkpoint=checkpoint, verbose=verbose),
        optimizer(check_set, checkpoint=checkpoint, verbose=verbose),
        alpha_decision_parameter=0.5,
        subprocess_miner=SubprocessMiner(checkpoint=checkpoint, verbose=verbose),
        checkpoint=checkpoint,
        verbose=verbose,
    )


def visualization_data(layer_assignment, hierarchy):
    discovered_models = {}
    subprocess_index = 1

    for layer, area in enumerate(hierarchy.areas, start=1):
        subprocess_components = []
        for subprocess in area.subprocesses or ():
            raw_subprocess = getattr(subprocess, "raw", subprocess)
            if raw_subprocess is None:
                continue
            try:
                raw_subprocess["index"] = subprocess_index
                raw_subprocess["global_id"] = f"subprocess_{subprocess_index}"
            except Exception:
                pass
            subprocess_components.append(raw_subprocess)
            subprocess_index += 1

        discovered_models[layer] = {
            "object_types": sorted(area.object_types),
            "activities": sorted(area.activities),
            "activity_resources": {
                activity: sorted(resource_types)
                for activity, resource_types in area.resources.items()
            },
            "highlighted_activities": [],
            "ocpn": area.net.raw if area.net is not None else None,
            "subprocess_components": tuple(subprocess_components),
        }

    return layer_assignment.as_object_to_layer(), discovered_models


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
    ).mine(ocel)
    hierarchy = build_model_discovery(
        checkpoint,
        verbose=verbose,
        optimization="greedy",
    ).mine(
        ocel,
        layer_assignment,
        [DEFAULT_LAYER_CONTEXT] * len(layer_assignment.layers),
    )

    solution, discovered_models = visualization_data(layer_assignment, hierarchy)
    visualize_layers_boxed(solution, title="Hierarchy Layers", output_path=DEFAULT_LAYERS_OUTPUT)
    visualize_hierarchy_with_models(
        solution,
        discovered_models,
        title="Hierarchy with Process Models",
        output_path=DEFAULT_MODEL_OUTPUT,
    )
    visualize_hierarchy_with_indexed_subprocesses(
        solution,
        discovered_models,
        title="Hierarchy with Indexed Subprocesses",
        output_path=DEFAULT_SUBPROCESS_OUTPUT,
    )

    return layer_assignment, hierarchy


if __name__ == "__main__":
    main()
