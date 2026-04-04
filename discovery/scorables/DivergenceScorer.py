from collections import defaultdict

from repo.discovery.Scorable import Scorable
from repo.discovery.totem import _prepare_totem_data, get_all_event_objects


class DivergenceScorer(Scorable):

    def __init__(self, eps):
        super().__init__(eps)
        self.divergence_relations = {}
        self.type_to_type = {}
        self.type_to_object = {}
        self.object_types = list()

    def prepare(self, ocel):
        _, _, o2o, self.type_to_object = _prepare_totem_data(ocel)
        self.object_types = list(ocel.object_types)
        self.type_to_type = {
            (source_type, target_type): 0
            for source_type in self.object_types
            for target_type in self.object_types
        }

        for source_type in self.object_types:
            for source_obj in self.type_to_object.get(source_type, ()):
                for target_type, target_objects in o2o.get(source_obj, {}).items():
                    if target_objects:
                        self.type_to_type[(source_type, target_type)] += 1

        object_to_type = {}
        for obj_type, objects in self.type_to_object.items():
            for obj in objects:
                object_to_type[obj] = obj_type

        targets_by_pair_activity_source = defaultdict(set)
        for event_id in ocel.events["_eventId"]:
            activity = ocel.get_event_activity(event_id)
            objects_by_type = defaultdict(set)
            for obj in get_all_event_objects(ocel, event_id):
                obj_type = object_to_type.get(obj)
                if obj_type is not None:
                    objects_by_type[obj_type].add(obj)

            frozen_objects_by_type = {
                obj_type: frozenset(objects)
                for obj_type, objects in objects_by_type.items()
                if objects
            }

            for source_type, source_objects in frozen_objects_by_type.items():
                for target_type, target_objects in frozen_objects_by_type.items():
                    key_prefix = (source_type, target_type), activity
                    for source_obj in source_objects:
                        targets_by_pair_activity_source[key_prefix + (source_obj,)].add(target_objects)

        divergences = defaultdict(set)
        for (pair, _, _), target_sets in targets_by_pair_activity_source.items():
            target_sets = list(target_sets)
            if len(target_sets) < 2:
                continue

            for index, target_objects in enumerate(target_sets):
                for other_target_objects in target_sets[index + 1:]:
                    diff = target_objects ^ other_target_objects
                    if diff:
                        divergences[pair].update(diff)

        self.divergence_relations = dict(divergences)

    def assign_score_pull(self, o_1, o_2) -> float:
        object_interactions = self.type_to_type.get((o_1, o_2), 0) / len(self.type_to_object.get(o_1, ()))
        object_interactions_reverse = self.type_to_type.get((o_2, o_1), 0) / len(self.type_to_object.get(o_2, ()))

        return (
                (1 - self.get_divergence_ratio(o_1, o_2)) * (1 - self.get_divergence_ratio(o_2, o_1)) *
                max(object_interactions, object_interactions_reverse)
        )

    def assign_score_push(self, o_1, o_2) -> float:
        return self.get_divergence_ratio(o_1, o_2) - self.get_divergence_ratio(o_2, o_1)

    def get_divergence_ratio(self, o_1, o_2) -> float:
        total_targets = len(self.type_to_object.get(o_2, ()))
        return len(self.divergence_relations.get((o_1, o_2), set())) / total_targets
