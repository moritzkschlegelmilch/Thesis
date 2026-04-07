from abc import ABC
from math import sqrt

from .Scorable import Scorable
from .ilp import solve
from .discovery_preparation import discover_models_for_hierarchy
from ..helpers.vorbose import visualize_hierarchy_with_models, visualize_layers_boxed


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

    def prepare(self):
        for scorable in self.scorables:
            scorable.prepare(self.ocel)

    def assign_scores(self):
        for o_1 in self.ocel.object_types:
            for o_2 in self.ocel.object_types:
                if o_1 == o_2:
                    self.scores_pull[o_1, o_2] = 0
                    self.scores_push[o_1, o_2] = 0
                    continue

                score_push: float = 0
                score_pull: float = 0

                for scorable in self.scorables:
                    score_push += scorable.assign_score_push(o_1, o_2) * scorable.eps
                    score_pull += scorable.assign_score_pull(o_1, o_2) * scorable.eps

                self.scores_pull[o_1, o_2] = score_pull / self.overall_weight
                self.scores_push[o_1, o_2] = score_push / self.overall_weight

    def solve_ilp(self):
        self.solution = solve(self.ocel.object_types, self.scores_push, self.scores_pull)

    def visualize_layers(self, title="Hierarchy Layers", output_path=None):
        return visualize_layers_boxed(self.solution, title=title, output_path=output_path)

    def visualize(self, title="Hierarchy with Process Models", output_path=None):
        return visualize_hierarchy_with_models(
            self.solution,
            self.discovered_models,
            title=title,
            output_path=output_path,
        )

    def discover_models(self):
        self.activity_to_layer, self.discovered_models = discover_models_for_hierarchy(
            self.ocel,
            self.solution,
        )

    def run(self):
        self.prepare()
        self.assign_scores()
        self.solve_ilp()
        self.discover_models()
