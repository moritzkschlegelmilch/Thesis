from repo.discovery.Scorable import Scorable
from repo.discovery.totem import _prepare_totem_data

LC_TOTAL = "total"
LC_ONE = "1"
LC_MANY = "1..*"


class CardinalityRelationScorer(Scorable):

    def __init__(self, eps):
        super().__init__(eps)
        self.cardinality_relations = None

    def prepare(self, ocel):
        _, _, o2o, type_to_object = _prepare_totem_data(ocel)

        h_log_cardinalities: dict[tuple[str, str], dict[str, int]] = {}

        for source_type in ocel.object_types:
            for target_type in ocel.object_types:
                pair = (source_type, target_type)
                h_log_cardinalities.setdefault(pair, {})
                h_log_cardinalities[pair][LC_TOTAL] = 0

                for source_obj in type_to_object.get(source_type, set()):
                    h_log_cardinalities[pair][LC_TOTAL] += 1

                    target_objects = o2o.get(source_obj, {}).get(target_type, set())
                    cardinality = len(target_objects)

                    if cardinality == 1:
                        target_obj = next(iter(target_objects))
                        reverse_links = o2o.get(target_obj, {}).get(source_type, set())
                        if len(reverse_links) == 1:
                            h_log_cardinalities[pair].setdefault(LC_ONE, 0)
                            h_log_cardinalities[pair][LC_ONE] += 1

                    elif cardinality > 1:
                        h_log_cardinalities[pair].setdefault(LC_MANY, 0)
                        h_log_cardinalities[pair][LC_MANY] += 1

        self.cardinality_relations = h_log_cardinalities

    def assign_score_pull(self, o_1, o_2) -> float:
        temp_r = self.cardinality_relations[o_1, o_2]
        temp_r_reverse = self.cardinality_relations[o_2, o_1]

        # if they are not related from o_1's perspective, then also not from o_2's perspective
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("1", 0)
        temp_r_reverse.setdefault("1", 0)

        return (temp_r["1"] + temp_r_reverse["1"]) / (temp_r["total"] + temp_r_reverse["total"])

    def assign_score_push(self, o_1, o_2) -> float:
        temp_r = self.cardinality_relations[o_1, o_2]
        temp_r_reverse = self.cardinality_relations[o_2, o_1]

        # if they are not related from o_1's perspective, then also not from o_2's perspective
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("1..*", 0)
        temp_r_reverse.setdefault("1..*", 0)

        return (temp_r["1..*"] / temp_r["total"]) - (temp_r_reverse["1..*"] / temp_r_reverse["total"])
