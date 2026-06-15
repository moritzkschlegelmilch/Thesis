from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .checkpoints import CheckpointManager
from .framework import LayerAssignment, ResourceForces
from .ilp import solve


class ResourceIndicator(ABC):
    """Base class for paper-style resource indicators."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def prepare(self, log) -> None:
        pass

    @abstractmethod
    def score(self, source_type: str, target_type: str) -> ResourceForces:
        pass


class ScorableResourceIndicator(ResourceIndicator):
    """Adapter for the existing scorer interface."""

    def __init__(self, scorable):
        super().__init__(getattr(scorable, "weight", getattr(scorable, "eps", 1.0)))
        self.scorable = scorable

    def prepare(self, log) -> None:
        self.scorable.prepare(log)

    def score(self, source_type: str, target_type: str) -> ResourceForces:
        return ResourceForces(
            push=float(self.scorable.assign_score_push(source_type, target_type)),
            pull=float(self.scorable.assign_score_pull(source_type, target_type)),
        )


class LayerAssignmentMiner:
    def __init__(
        self,
        resource_indicators: Sequence[ResourceIndicator],
        alpha: float = 1.0,
        beta: float = 1.0,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
        debug_matrices: bool = False,
    ):
        self.resource_indicators = tuple(
            indicator
            if isinstance(indicator, ResourceIndicator)
            else ScorableResourceIndicator(indicator)
            for indicator in resource_indicators
        )
        self.alpha = alpha
        self.beta = beta
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)
        self.debug_matrices = debug_matrices
        self.layer_assignment: LayerAssignment | None = None
        self.scores_push: dict[tuple[str, str], float] = {}
        self.scores_pull: dict[tuple[str, str], float] = {}

    def prepare(self, log) -> None:
        with self.checkpoint.section(
            "Prepare resource indicators",
            total=len(self.resource_indicators),
            metadata={"indicators": len(self.resource_indicators)},
            unit="indicator",
        ) as span:
            for indicator in self.resource_indicators:
                indicator_name = type(indicator).__name__
                with span.child(
                    f"Prepare {indicator_name}",
                    metadata={"indicator": indicator_name},
                ):
                    indicator.prepare(log)
                span.update(postfix=f"last={indicator_name}")

    def assign_scores(self, log) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        object_types = tuple(log.object_types)
        total_weight = sum(indicator.weight for indicator in self.resource_indicators)
        if total_weight <= 0:
            raise ValueError("At least one resource indicator with positive weight is required.")

        scores_push = {}
        scores_pull = {}
        with self.checkpoint.section(
            "Assign resource scores",
            total=len(object_types) * len(object_types),
            metadata={"object_types": len(object_types)},
            unit="pair",
        ) as span:
            for source_type in object_types:
                for target_type in object_types:
                    if source_type == target_type:
                        scores_push[(source_type, target_type)] = 0.0
                        scores_pull[(source_type, target_type)] = 0.0
                        span.update()
                        continue

                    push = 0.0
                    pull = 0.0
                    for indicator in self.resource_indicators:
                        forces = indicator.score(source_type, target_type)
                        push += indicator.weight * forces.push
                        pull += indicator.weight * forces.pull

                    scores_push[(source_type, target_type)] = push / total_weight
                    scores_pull[(source_type, target_type)] = pull / total_weight
                    span.update(postfix=f"{source_type} -> {target_type}")

        self.scores_push = scores_push
        self.scores_pull = scores_pull
        if self.debug_matrices:
            self.print_debug_matrices(object_types, scores_push, scores_pull)
        return scores_push, scores_pull

    def print_debug_matrices(
        self,
        object_types: Sequence[str],
        scores_push: dict[tuple[str, str], float],
        scores_pull: dict[tuple[str, str], float],
    ) -> None:
        try:
            from ..helpers.vorbose import print_tuple_dict_matrices
        except ImportError:
            from helpers.vorbose import print_tuple_dict_matrices

        object_types = tuple(object_types)
        print("\n=== Layer scorer debug matrices ===")
        for indicator in self.resource_indicators:
            indicator_push = {}
            indicator_pull = {}
            for source_type in object_types:
                for target_type in object_types:
                    if source_type == target_type:
                        indicator_push[(source_type, target_type)] = 0.0
                        indicator_pull[(source_type, target_type)] = 0.0
                        continue
                    forces = indicator.score(source_type, target_type)
                    indicator_push[(source_type, target_type)] = float(forces.push)
                    indicator_pull[(source_type, target_type)] = float(forces.pull)

            print(f"\n--- {type(indicator).__name__} (weight={indicator.weight}) ---")
            print_tuple_dict_matrices(indicator_push, indicator_pull)
            if hasattr(indicator, "print_debug_details"):
                indicator.print_debug_details(object_types)

        print("\n--- Combined weighted scores ---")
        print_tuple_dict_matrices(scores_push, scores_pull)

    def mine(self, log) -> LayerAssignment:
        with self.checkpoint.section(
            "Layer assignment mining",
            total=4,
            metadata={"object_types": len(tuple(log.object_types))},
        ) as span:
            self.prepare(log)
            span.update(postfix="indicators prepared")

            scores_push, scores_pull = self.assign_scores(log)
            span.update(postfix="scores assigned")

            with span.child(
                "Solve layer ILP",
                metadata={"object_types": len(tuple(log.object_types))},
            ):
                solution = solve(
                    tuple(log.object_types),
                    scores_push,
                    scores_pull,
                    alpha=self.alpha,
                    beta=self.beta,
                )
            span.update(postfix="ILP solved")

            with span.child("Normalize layer assignment"):
                self.layer_assignment = LayerAssignment.from_mapping(solution)
            span.update(postfix="assignment normalized")

        return self.layer_assignment
