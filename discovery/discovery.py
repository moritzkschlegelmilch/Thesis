from abc import ABC
from math import sqrt

from .Scorable import DataSource
from .ilp import solve
from .totem import totemDiscovery


class ProcessAreaDiscoveryFramework(ABC):

    def __init__(self, ocel, data_sources: list[DataSource]):
        self.data_sources = data_sources
        self.scores_push: dict[tuple[str, str], float] = dict()
        self.scores_pull: dict[tuple[str, str], float] = dict()
        self.ocel = ocel

        self.overall_weight: int = 0
        for data_source in self.data_sources:
            self.overall_weight += data_source.weight()

    def prepare(self):
        for data_source in self.data_sources:
            data_source.prepare(self.ocel)

    def assign_scores(self):
        for o_1 in self.ocel.object_types:
            for o_2 in self.ocel.object_types:
                score_push: float = 0
                score_pull: float = 0

                for data_source in self.data_sources:
                    score_push += data_source.assign_score_push(o_1, o_2)
                    score_pull += data_source.assign_score_pull(o_1, o_2)

                self.scores_pull[o_1, o_2] = score_pull / self.overall_weight
                self.scores_push[o_1, o_2] = score_push / self.overall_weight

    def solve_ilp(self):
        solve(self.ocel.object_types, self.scores_push, self.scores_pull, 0)

    def run(self):
        self.prepare()
        self.assign_scores()
        self.solve_ilp()
