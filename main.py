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


DEFAULT_INPUT_PATH = "simple_logs/01_o2c.xml"
DEFAULT_LAYER_CONTEXT = 1
PRECISION_CONTEXT_SAMPLE_SIZE = 256
PRECISION_CONTEXT_DEPTH = 5
PRECISION_CONTEXT_LENGTH = 5
PRECISION_CONTEXT_SAMPLE_SEED = None
MAX_NODES_PER_REPLAY = 100
DEFAULT_LAYERS_OUTPUT = "output/hierarchy_layers.png"
DEFAULT_MODEL_OUTPUT = "output/hierarchy_with_models.png"
DEFAULT_SUBPROCESS_OUTPUT = "output/hierarchy_with_subprocesses.png"


def build_model_discovery(checkpoint, *, verbose=False):
    check_set = CheckSet(
        PM4PyOCPNDiscovery(checkpoint=checkpoint, verbose=verbose),
        SubprocessMiner(checkpoint=checkpoint, verbose=verbose),
        CollapsedNetBuilder(checkpoint=checkpoint, verbose=verbose),
        PrecisionCalculator(
            PrecisionParameters(
                d=PRECISION_CONTEXT_DEPTH,
                l=PRECISION_CONTEXT_LENGTH,
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

    return hierarchy


if __name__ == "__main__":
    main()
