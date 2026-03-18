from repo.discovery.Scorable import Scorable


class CardinalityRelationScorer(Scorable):
    def assign_score_pull(self, o_1, o_2, data) -> float:
        print(data.temporal_relations)

        return 0

    def assign_score_push(self, o_1, o_2, data) -> float:
        temp_r = data.temporal_relations[o_1, o_2]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("Di", 0)
        temp_r.setdefault("D", 0)

        return 0