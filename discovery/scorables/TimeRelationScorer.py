from repo.discovery.Scorable import Scorable
from repo.discovery.totem import _prepare_totem_data

TR_TOTAL = "total"
TR_DEPENDENT = "D"
TR_INITIATING = "I"


class TimeRelationScorer(Scorable):

    def __init__(self, eps):
        super().__init__(eps)
        self.temporal_relations = None

    def prepare(self, ocel):
        o_min_times, o_max_times, o2o, type_to_object = _prepare_totem_data(ocel)

        h_temporal_relations: dict[tuple[str, str], dict[str, int]] = {}

        for source_type in ocel.object_types:
            for target_type in ocel.object_types:
                pair = (source_type, target_type)
                h_temporal_relations.setdefault(pair, {})

                for source_obj in type_to_object.get(source_type, set()):
                    target_objects = o2o.get(source_obj, {}).get(target_type, set())

                    for target_obj in target_objects:
                        if (
                                source_obj not in o_min_times
                                or source_obj not in o_max_times
                                or target_obj not in o_min_times
                                or target_obj not in o_max_times
                        ):
                            continue

                        h_temporal_relations[pair].setdefault(TR_TOTAL, 0)
                        h_temporal_relations[pair][TR_TOTAL] += 1

                        if (
                                o_min_times[target_obj]
                                <= o_min_times[source_obj]
                                <= o_max_times[source_obj]
                                <= o_max_times[target_obj]
                        ):
                            h_temporal_relations[pair].setdefault(TR_DEPENDENT, 0)
                            h_temporal_relations[pair][TR_DEPENDENT] += 1

                        if (
                                o_min_times[source_obj]
                                <= o_max_times[source_obj]
                                <= o_min_times[target_obj]
                                <= o_max_times[target_obj]
                        ) or (
                                o_min_times[source_obj]
                                < o_min_times[target_obj]
                                <= o_max_times[source_obj]
                                < o_max_times[target_obj]
                        ):
                            h_temporal_relations[pair].setdefault(TR_INITIATING, 0)
                            h_temporal_relations[pair][TR_INITIATING] += 1

        self.temporal_relations = h_temporal_relations

    def assign_score_pull(self, o_1, o_2) -> float:
        temp_r = self.temporal_relations[o_1, o_2]
        temp_r_reverse = self.temporal_relations[o_2, o_1]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("I", 0)
        temp_r_reverse.setdefault("I", 0)

        return (temp_r_reverse["I"] + temp_r["I"]) / temp_r["total"]

    def assign_score_push(self, o_1, o_2) -> float:
        temp_r = self.temporal_relations[o_1, o_2]
        temp_r_reverse = self.temporal_relations[o_2, o_1]
        if "total" not in temp_r or temp_r["total"] == 0:
            return 0

        temp_r.setdefault("D", 0)
        temp_r_reverse.setdefault("D", 0)

        return (temp_r_reverse["D"] - temp_r["D"]) / temp_r["total"]
