from abc import ABC
from math import sqrt

from .Scorable import Scorable
from .ilp import solve


class ProcessAreaDiscoveryFramework(ABC):

    def __init__(self, ocel, scorables: list[Scorable]):
        self.scorables = scorables
        self.scores_push: dict[tuple[str, str], float] = dict()
        self.scores_pull: dict[tuple[str, str], float] = dict()
        self.ocel = ocel

        self.overall_weight: int = 0
        for scorable in self.scorables:
            self.overall_weight += scorable.eps

    def prepare(self):
        for scorable in self.scorables:
            scorable.prepare(self.ocel)

    def assign_scores(self):
        for o_1 in self.ocel.object_types:
            for o_2 in self.ocel.object_types:
                score_push: float = 0
                score_pull: float = 0

                for scorable in self.scorables:
                    score_push += scorable.assign_score_push(o_1, o_2)*scorable.eps
                    score_pull += scorable.assign_score_pull(o_1, o_2)*scorable.eps

                self.scores_pull[o_1, o_2] = score_pull / self.overall_weight
                self.scores_push[o_1, o_2] = score_push / self.overall_weight

    def solve_ilp(self):
        solve(self.ocel.object_types, self.scores_push, self.scores_pull)

    def run(self):
        self.prepare()
        self.assign_scores()
        self.solve_ilp()
