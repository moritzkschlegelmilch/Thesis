from collections import defaultdict

from repo.discovery.Scorable import Scorable
from repo.discovery.totem import _prepare_totem_data

LC_TOTAL = "total"
LC_CONSTANT = "constant"
LC_MANY = "other"


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
                bilateral_cardinality_counts = defaultdict(int)
                total = 0

                for source_obj in type_to_object.get(source_type, set()):
                    total += 1
                    target_objects = o2o.get(source_obj, {}).get(target_type, set())
                    forward_cardinality = len(target_objects)
                    if forward_cardinality > 0:
                        reverse_cardinalities = tuple(sorted(
                            len(o2o.get(target_obj, {}).get(source_type, set()))
                            for target_obj in target_objects
                        ))
                        bilateral_signature = (forward_cardinality, reverse_cardinalities)
                        bilateral_cardinality_counts[bilateral_signature] += 1

                h_log_cardinalities[pair] = {LC_TOTAL: total}
                if total == 0 or not bilateral_cardinality_counts:
                    h_log_cardinalities[pair][LC_CONSTANT] = 0
                    h_log_cardinalities[pair][LC_MANY] = total
                    continue

                dominant_signature = max(
                    bilateral_cardinality_counts,
                    key=lambda signature: (
                        bilateral_cardinality_counts[signature],
                        -signature[0],
                        tuple(-value for value in signature[1]),
                    ),
                )
                constant_count = bilateral_cardinality_counts[dominant_signature]
                h_log_cardinalities[pair][LC_CONSTANT] = constant_count
                h_log_cardinalities[pair][LC_MANY] = total - constant_count

        self.cardinality_relations = h_log_cardinalities

    def assign_score_pull(self, o_1, o_2) -> float:
        temp_r = self.cardinality_relations[o_1, o_2]
        temp_r_reverse = self.cardinality_relations[o_2, o_1]

        # if they are not related from o_1's perspective, then also not from o_2's perspective
        if LC_TOTAL not in temp_r or temp_r[LC_TOTAL] == 0:
            return 0

        temp_r.setdefault(LC_CONSTANT, 0)
        temp_r_reverse.setdefault(LC_CONSTANT, 0)

        return (
            temp_r[LC_CONSTANT] + temp_r_reverse[LC_CONSTANT]
        ) / (temp_r[LC_TOTAL] + temp_r_reverse[LC_TOTAL])

    def assign_score_push(self, o_1, o_2) -> float:
        temp_r = self.cardinality_relations[o_1, o_2]
        temp_r_reverse = self.cardinality_relations[o_2, o_1]

        # if they are not related from o_1's perspective, then also not from o_2's perspective
        if LC_TOTAL not in temp_r or temp_r[LC_TOTAL] == 0:
            return 0

        temp_r.setdefault(LC_MANY, 0)
        temp_r_reverse.setdefault(LC_MANY, 0)

        return (temp_r[LC_MANY] / temp_r[LC_TOTAL]) - (temp_r_reverse[LC_MANY] / temp_r_reverse[LC_TOTAL])
