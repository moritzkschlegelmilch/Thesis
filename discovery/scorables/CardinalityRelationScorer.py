from collections import defaultdict
from math import log2

from ..Scorable import Scorable
from ..framework import ResourceForces
from ..layer_assignment import ResourceIndicator
from ..totem import get_all_event_objects


class CardinalityRelationScorer(Scorable, ResourceIndicator):

    def __init__(self, eps):
        Scorable.__init__(self, eps)
        ResourceIndicator.__init__(self, weight=eps)
        self.normalized_entropy = {}
        self.cardinality_counts = {}
        self.type_to_object = {}

    def prepare(self, ocel):
        o2o, type_to_object = _build_event_derived_o2o(ocel)
        self.type_to_object = type_to_object
        normalized_entropy = {}
        cardinality_counts_by_pair = {}

        for source_type in ocel.object_types:
            source_objects = type_to_object.get(source_type, set())
            source_count = len(source_objects)
            for target_type in ocel.object_types:
                cardinality_counts = defaultdict(int)
                for source_obj in source_objects:
                    cardinality = len(o2o.get(source_obj, {}).get(target_type, set()))
                    cardinality_counts[cardinality] += 1

                normalized_entropy[(source_type, target_type)] = _normalized_entropy(
                    cardinality_counts,
                    source_count,
                )
                cardinality_counts_by_pair[(source_type, target_type)] = dict(cardinality_counts)

        self.normalized_entropy = normalized_entropy
        self.cardinality_counts = cardinality_counts_by_pair

    def assign_score_pull(self, o_1, o_2) -> float:
        entropy_forward = self.normalized_entropy.get((o_1, o_2), 0.0)
        entropy_reverse = self.normalized_entropy.get((o_2, o_1), 0.0)
        if entropy_forward + entropy_reverse == 0:
            return 0

        harmonic_mean = (
            2 * entropy_forward * entropy_reverse
        ) / (
            entropy_forward + entropy_reverse
        )
        return harmonic_mean ** 2

    def assign_score_push(self, o_1, o_2) -> float:
        return (
            self.normalized_entropy.get((o_1, o_2), 0.0)
            - self.normalized_entropy.get((o_2, o_1), 0.0)
        )

    def score(self, source_type: str, target_type: str) -> ResourceForces:
        return ResourceForces(
            push=self.assign_score_push(source_type, target_type),
            pull=self.assign_score_pull(source_type, target_type),
        )

    def print_debug_details(self, object_types) -> None:
        print("\nCardinality details:")
        print(
            "source -> target | source_objects | target_objects | "
            "histogram | entropy | reverse_entropy | push | pull"
        )
        for source_type in object_types:
            for target_type in object_types:
                if source_type == target_type:
                    continue
                source_count = len(self.type_to_object.get(source_type, ()))
                target_count = len(self.type_to_object.get(target_type, ()))
                histogram = self.cardinality_counts.get((source_type, target_type), {})
                entropy = self.normalized_entropy.get((source_type, target_type), 0.0)
                reverse_entropy = self.normalized_entropy.get((target_type, source_type), 0.0)
                push = self.assign_score_push(source_type, target_type)
                pull = self.assign_score_pull(source_type, target_type)
                histogram_text = "{" + ", ".join(
                    f"{cardinality}: {count}"
                    for cardinality, count in sorted(histogram.items())
                ) + "}"
                print(
                    f"{source_type} -> {target_type} | "
                    f"{source_count} | {target_count} | "
                    f"{histogram_text} | "
                    f"{entropy:.4f} | {reverse_entropy:.4f} | "
                    f"{push:.4f} | {pull:.4f}"
                )


def _normalized_entropy(cardinality_counts, object_count):
    if object_count <= 1:
        return 0.0

    entropy = 0.0
    for count in cardinality_counts.values():
        probability = count / object_count
        if probability > 0:
            entropy -= probability * log2(probability)

    upper_entropy = log2(object_count)
    if upper_entropy == 0:
        return 0.0
    return entropy / upper_entropy


def _build_event_derived_o2o(ocel):
    object_to_type = _extract_object_to_type(ocel)
    type_to_object = defaultdict(set)
    o2o = defaultdict(lambda: defaultdict(set))

    event_ids = list(_event_ids(ocel))
    for event_id in event_ids:
        event_objects = [
            obj
            for obj in _event_objects(ocel, event_id)
            if obj in object_to_type
        ]
        objects_by_type = defaultdict(set)
        for obj in event_objects:
            object_type = object_to_type[obj]
            type_to_object[object_type].add(obj)
            objects_by_type[object_type].add(obj)

        for source_obj in event_objects:
            source_type = object_to_type[source_obj]
            type_to_object[source_type].add(source_obj)
            for target_type, target_objects in objects_by_type.items():
                o2o[source_obj][target_type].update(
                    target_obj
                    for target_obj in target_objects
                    if target_obj != source_obj
                )

    return {
        source_obj: dict(targets_by_type)
        for source_obj, targets_by_type in o2o.items()
    }, {
        object_type: set(objects)
        for object_type, objects in type_to_object.items()
    }


def _extract_object_to_type(ocel):
    objects_df = getattr(ocel, "objects", None)
    if objects_df is not None:
        if {"ocel:oid", "ocel:type"} <= set(getattr(objects_df, "columns", ())):
            return dict(zip(objects_df["ocel:oid"], objects_df["ocel:type"]))
        if {"_objId", "_objType"} <= set(getattr(objects_df, "columns", ())):
            return dict(zip(objects_df["_objId"], objects_df["_objType"]))

    object_to_type = {}
    for event_id in _event_ids(ocel):
        for object_type in ocel.object_types:
            for obj in ocel.get_event_objects_by_type(event_id, object_type):
                object_to_type[obj] = object_type
    return object_to_type


def _event_ids(ocel):
    events_df = getattr(ocel, "events", None)
    if events_df is not None:
        if "_eventId" in events_df.columns:
            return events_df["_eventId"]
        if "ocel:eid" in events_df.columns:
            return events_df["ocel:eid"]
        event_id_column = getattr(ocel, "event_id_column", None)
        if event_id_column is not None and event_id_column in events_df.columns:
            return events_df[event_id_column]
    return ()


def _event_objects(ocel, event_id):
    relations_df = getattr(ocel, "relations", None)
    if relations_df is not None and {"ocel:eid", "ocel:oid"} <= set(getattr(relations_df, "columns", ())):
        return tuple(relations_df.loc[relations_df["ocel:eid"] == event_id, "ocel:oid"])

    try:
        return tuple(get_all_event_objects(ocel, event_id))
    except Exception:
        return ()
