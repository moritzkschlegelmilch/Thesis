from collections import defaultdict

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL

from .totem import _prepare_totem_data, get_all_event_objects


def discover_models_for_hierarchy(ocel, solution):
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

    layer_to_activities = defaultdict(set)
    for activity, layer in activity_to_layer.items():
        layer_to_activities[layer].add(activity)

    discovered_models = {}
    for layer in sorted(layer_to_object_types):
        selected_object_types = layer_to_object_types[layer]
        selected_activities = layer_to_activities.get(layer, set())

        event_rows = []
        relation_rows = []
        used_objects = set()

        for event_id, activity, timestamp, event_objects in event_records:
            if activity not in selected_activities:
                continue

            selected_event_objects = [
                obj for obj in event_objects
                if object_to_type[obj] in selected_object_types
            ]
            if not selected_event_objects:
                continue

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
            "activities": sorted(selected_activities),
            "ocel": layer_ocel,
            "ocpn": ocpn,
        }

    return dict(activity_to_layer), discovered_models
