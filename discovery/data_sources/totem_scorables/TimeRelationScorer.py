from repo.discovery.Scorable import Scorable


class TimeRelationScorer(Scorable):
    def assign_score_pull(self, o_1, o_2, data) -> float:
        temp_r = data.temporal_relations[o_1, o_2]
        temp_r_reverse = data.temporal_relations[o_2, o_1]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("I", 0)
        temp_r_reverse.setdefault("I", 0)

        return (temp_r_reverse["I"] + temp_r["I"]) / temp_r["total"]

    def assign_score_push(self, o_1, o_2, data) -> float:
        temp_r = data.temporal_relations[o_1, o_2]
        temp_r_reverse = data.temporal_relations[o_2, o_1]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("D", 0)
        temp_r_reverse.setdefault("D", 0)

        return (temp_r_reverse["D"] - temp_r["D"]) / temp_r["total"]