from abc import ABC
from math import sqrt
import sys

from .Scorable import Scorable
from .ilp import solve
from .discovery_preparation import discover_models_for_hierarchy
try:
    from ..helpers.vorbose import (
        visualize_hierarchy_with_indexed_subprocesses,
        visualize_hierarchy_with_models,
        visualize_layers_boxed,
    )
except ImportError:
    from helpers.vorbose import (
        visualize_hierarchy_with_indexed_subprocesses,
        visualize_hierarchy_with_models,
        visualize_layers_boxed,
    )
from tqdm import tqdm


class ProcessAreaDiscoveryFramework(ABC):

    def __init__(self, ocel, scorables: list[Scorable]):
        self.scorables = scorables
        self.scores_push: dict[tuple[str, str], float] = dict()
        self.scores_pull: dict[tuple[str, str], float] = dict()
        self.ocel = ocel
        self.solution = None
        self.activity_to_layer = None
        self.discovered_models = None

        self.overall_weight: int = 0
        for scorable in self.scorables:
            self.overall_weight += scorable.eps

    def prepare(self, show_progress=False):
        scorables = self.scorables
        if show_progress:
            scorables = tqdm(
                self.scorables,
                total=len(self.scorables),
                desc="Preparing scorers",
                file=sys.stdout,
            )

        for scorable in scorables:
            scorable.prepare(self.ocel)

    def assign_scores(self, show_progress=False):
        object_types = tuple(self.ocel.object_types)
        total_pairs = len(object_types) * len(object_types)
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=total_pairs,
                desc="Assigning scores",
                file=sys.stdout,
            )

        for o_1 in object_types:
            for o_2 in object_types:
                if o_1 == o_2:
                    self.scores_pull[o_1, o_2] = 0
                    self.scores_push[o_1, o_2] = 0
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue

                score_push: float = 0
                score_pull: float = 0

                for scorable in self.scorables:
                    score_push += scorable.assign_score_push(o_1, o_2) * scorable.eps
                    score_pull += scorable.assign_score_pull(o_1, o_2) * scorable.eps

                self.scores_pull[o_1, o_2] = score_pull / self.overall_weight
                self.scores_push[o_1, o_2] = score_push / self.overall_weight

                if progress_bar is not None:
                    progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    def solve_ilp(self):
        self.solution = solve(self.ocel.object_types, self.scores_push, self.scores_pull)
        return self.solution

    def visualize_layers(self, title="Hierarchy Layers", output_path=None):
        return visualize_layers_boxed(self.solution, title=title, output_path=output_path)

    def visualize(self, title="Hierarchy with Process Models", output_path=None):
        return visualize_hierarchy_with_models(
            self.solution,
            self.discovered_models,
            title=title,
            output_path=output_path,
        )

    def visualize_indexed_subprocesses(
        self,
        title="Hierarchy with Indexed Subprocesses",
        output_path=None,
    ):
        return visualize_hierarchy_with_indexed_subprocesses(
            self.solution,
            self.discovered_models,
            title=title,
            output_path=output_path,
        )

    def visualize_subprocesses(
        self,
        title="Hierarchy with Indexed Subprocesses",
        output_path=None,
    ):
        return self.visualize_indexed_subprocesses(
            title=title,
            output_path=output_path,
        )

    def discover_models(
        self,
        layer_context=None,
        show_progress=False,
        *,
        precision_context_sample_size=512,
        precision_context_depth=5,
        precision_context_length=5,
        precision_context_sample_seed=None,
    ):
        self.activity_to_layer, self.discovered_models = discover_models_for_hierarchy(
            self.ocel,
            self.solution,
            layer_context=layer_context,
            show_progress=show_progress,
            precision_context_sample_size=precision_context_sample_size,
            precision_context_depth=precision_context_depth,
            precision_context_length=precision_context_length,
            precision_context_sample_seed=precision_context_sample_seed,
        )

    def get_layers(self, show_progress=False):
        self.prepare(show_progress=show_progress)
        self.assign_scores(show_progress=show_progress)
        return self.solve_ilp()

    def run(
        self,
        layer_context=None,
        show_progress=False,
        *,
        precision_context_sample_size=512,
        precision_context_depth=5,
        precision_context_length=5,
        precision_context_sample_seed=None,
    ):
        self.get_layers(show_progress=show_progress)
        self.discover_models(
            layer_context=layer_context,
            show_progress=show_progress,
            precision_context_sample_size=precision_context_sample_size,
            precision_context_depth=precision_context_depth,
            precision_context_length=precision_context_length,
            precision_context_sample_seed=precision_context_sample_seed,
        )
        return self.solution, self.discovered_models
