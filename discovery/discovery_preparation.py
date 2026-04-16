from collections import defaultdict
from numbers import Integral

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL

from .totem import _prepare_totem_data, get_all_event_objects


def _normalize_layer_context(discovered_layers, layer_context):
    if layer_context is None:
        return {layer: 1 for layer in discovered_layers}

    if len(layer_context) != len(discovered_layers):
        raise ValueError(
            "layer_context must have one entry per discovered layer "
            "(ordered from the lowest to the highest layer)."
        )

    normalized_layer_context = {}
    for index, (layer, context_value) in enumerate(zip(discovered_layers, layer_context)):
        if isinstance(context_value, bool) or not isinstance(context_value, Integral) or context_value < 0:
            raise ValueError(
                "Each layer_context entry must be a non-negative integer. "
                f"Invalid value at index {index}: {context_value!r}"
            )
        normalized_layer_context[layer] = int(context_value)

    return normalized_layer_context


def _discover_activity_resources(event_records, object_to_type, solution, reference_layer):
    activity_event_counts = defaultdict(int)
    activity_resource_counts = defaultdict(lambda: defaultdict(int))

    for _, activity, _, event_objects in event_records:
        activity_event_counts[activity] += 1
        higher_layer_types = {
            object_to_type[obj]
            for obj in event_objects
            if solution[object_to_type[obj]] > reference_layer
        }

        for object_type in higher_layer_types:
            activity_resource_counts[activity][object_type] += 1

    activity_resources = {}
    for activity, total_count in activity_event_counts.items():
        qualifying_object_types = [
            object_type
            for object_type, occurrence_count in activity_resource_counts[activity].items()
            if occurrence_count / total_count >= 0.5
        ]
        qualifying_object_types.sort(key=lambda object_type: (-solution[object_type], object_type))
        activity_resources[activity] = qualifying_object_types

    return activity_resources


def discover_models_for_hierarchy(ocel, solution, layer_context=None):
    _, _, _, type_to_object = _prepare_totem_data(ocel)

    object_to_type = {}
    for obj_type, objects in type_to_object.items():
        for obj in objects:
            object_to_type[obj] = obj_type

    event_records = []
    activity_to_layer = {}
    layer_to_object_types = defaultdict(set)

    for obj_type, layer in solution.items():
        layer_to_object_types[layer].add(obj_type)

    for event_id in ocel.events["_eventId"]:
        activity = ocel.get_event_activity(event_id)
        timestamp = ocel.get_event_timestamp(event_id)
        event_objects = [
            obj for obj in get_all_event_objects(ocel, event_id)
            if obj in object_to_type
        ]
        if not event_objects:
            continue

        event_layer = min(solution[object_to_type[obj]] for obj in event_objects)
        current_layer = activity_to_layer.get(activity)
        if current_layer is None or event_layer < current_layer:
            activity_to_layer[activity] = event_layer

        event_records.append((event_id, activity, timestamp, event_objects))

    discovered_layers = sorted(layer_to_object_types)
    layer_context_by_layer = _normalize_layer_context(discovered_layers, layer_context)

    discovered_models = {}
    for layer in discovered_layers:
        selected_object_types = layer_to_object_types[layer]
        selected_activities = {
            activity
            for activity, activity_layer in activity_to_layer.items()
            if activity_layer <= layer and layer - activity_layer <= layer_context_by_layer[layer]
        }

        event_rows = []
        relation_rows = []
        used_objects = set()
        included_event_records = []
        included_activities = set()

        for event_id, activity, timestamp, event_objects in event_records:
            if activity not in selected_activities:
                continue

            selected_event_objects = [
                obj for obj in event_objects
                if object_to_type[obj] in selected_object_types
            ]
            if not selected_event_objects:
                continue

            included_event_records.append((event_id, activity, timestamp, event_objects))
            included_activities.add(activity)
            event_rows.append({
                "ocel:eid": event_id,
                "ocel:activity": activity,
                "ocel:timestamp": timestamp,
            })

            for obj in dict.fromkeys(selected_event_objects):
                used_objects.add(obj)
                relation_rows.append({
                    "ocel:eid": event_id,
                    "ocel:activity": activity,
                    "ocel:timestamp": timestamp,
                    "ocel:oid": obj,
                    "ocel:type": object_to_type[obj],
                    "ocel:qualifier": None,
                })

        object_rows = [
            {
                "ocel:oid": obj,
                "ocel:type": object_to_type[obj],
            }
            for obj in sorted(used_objects)
        ]

        o2o_rows = [
            {
                "ocel:oid": source_obj,
                "ocel:oid_2": target_obj,
                "ocel:qualifier": None,
            }
            for source_obj, target_obj in ocel.o2o_graph_edges
            if source_obj in used_objects and target_obj in used_objects
        ]

        activity_resources = _discover_activity_resources(
            included_event_records,
            object_to_type,
            solution,
            layer,
        )
        included_activities = sorted(included_activities)
        highlighted_activities = sorted(
            activity
            for activity in included_activities
            if activity_to_layer[activity] == layer - 1
        )

        layer_ocel = OCEL(
            events=pd.DataFrame(
                event_rows,
                columns=["ocel:eid", "ocel:activity", "ocel:timestamp"],
            ),
            objects=pd.DataFrame(
                object_rows,
                columns=["ocel:oid", "ocel:type"],
            ),
            relations=pd.DataFrame(
                relation_rows,
                columns=[
                    "ocel:eid",
                    "ocel:activity",
                    "ocel:timestamp",
                    "ocel:oid",
                    "ocel:type",
                    "ocel:qualifier",
                ],
            ),
            o2o=pd.DataFrame(
                o2o_rows,
                columns=["ocel:oid", "ocel:oid_2", "ocel:qualifier"],
            ),
        )

        ocpn = None
        if not layer_ocel.events.empty and not layer_ocel.relations.empty:
            ocpn = pm4py.discover_oc_petri_net(layer_ocel)

        discovered_models[layer] = {
            "object_types": sorted(selected_object_types),
            "activities": included_activities,
            "activity_resources": {
                activity: activity_resources.get(activity, [])
                for activity in included_activities
            },
            "highlighted_activities": highlighted_activities,
            "ocel": layer_ocel,
            "ocpn": ocpn,
        }

    return dict(activity_to_layer), discovered_models
