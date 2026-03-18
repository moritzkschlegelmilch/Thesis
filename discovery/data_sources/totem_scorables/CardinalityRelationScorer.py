from repo.discovery.Scorable import Scorable


class CardinalityRelationScorer(Scorable):
    def assign_score_pull(self, o_1, o_2, data) -> float:
        temp_r = data.cardinality_relations[o_1, o_2]
        temp_r_reverse = data.cardinality_relations[o_2, o_1]

        # if they are not related from o_1's perspective, then also not from o_2's perspective
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("1", 0)
        temp_r_reverse.setdefault("1", 0)

        return (temp_r["1"] + temp_r_reverse["1"]) / (temp_r["total"] + temp_r_reverse["total"])

    def assign_score_push(self, o_1, o_2, data) -> float:
        temp_r = data.cardinality_relations[o_1, o_2]
        temp_r_reverse = data.cardinality_relations[o_2, o_1]

        # if they are not related from o_1's perspective, then also not from o_2's perspective
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("1..*", 0)
        temp_r_reverse.setdefault("1..*", 0)

        return (temp_r["1..*"] / temp_r["total"]) - (temp_r_reverse["1..*"] / temp_r_reverse["total"])