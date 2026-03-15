from math import sqrt

from .ilp import solve
from .totem import totemDiscovery


class ProcessAreaDiscovery:

    def __init__(self, ocel):
        self.temporal_relations = None
        self.scores_push: dict[tuple[str, str], int] = dict()
        self.scores_pull: dict[tuple[str, str], int] = dict()
        self.ocel = ocel

    def prepare(self):
        self.temporal_relations = totemDiscovery(self.ocel)

    def assign_scores(self):
        for o_1 in self.ocel.object_types:
            for o_2 in self.ocel.object_types:
                self.scores_push[o_1, o_2] = self.assign_score_push(o_1, o_2)
                self.scores_pull[o_1, o_2] = self.assign_score_pull(o_1, o_2)

    def assign_score_push(self, o_1, o_2):
        temp_r = self.temporal_relations[o_1, o_2]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("Di", 0)
        temp_r.setdefault("D", 0)

        return (temp_r["Di"] - temp_r["D"]) / temp_r["total"]

    def assign_score_pull(self, o_1, o_2):
        temp_r = self.temporal_relations[o_1, o_2]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("Ii", 0)
        temp_r.setdefault("I", 0)

        return (temp_r["Ii"] + temp_r["I"]) / temp_r["total"]

    def solve_ilp(self):
        solve(self.ocel.object_types, self.scores_push, self.scores_pull, 0)
