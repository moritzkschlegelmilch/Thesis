from collections import defaultdict
from itertools import count

import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL

from .totem import _prepare_totem_data, get_all_event_objects

SERVE_AS_RESOURCE_ACTIVITY = "serve_as_resource"
START_SERVING_ACTIVITY = "start_serving"
END_SERVING_ACTIVITY = "end_serving"

EVENT_COLUMNS = ["ocel:eid", "ocel:activity", "ocel:timestamp"]
RELATION_COLUMNS = [
    "ocel:eid",
    "ocel:activity",
    "ocel:timestamp",
    "ocel:oid",
    "ocel:type",
    "ocel:qualifier",
]
OBJECT_COLUMNS = ["ocel:oid", "ocel:type"]
O2O_COLUMNS = ["ocel:oid", "ocel:oid_2", "ocel:qualifier"]


def _build_event_object_graph(relation_rows):
    graph = nx.Graph()

    for relation in relation_rows:
        graph.add_edge(
            ("event", relation["ocel:eid"]),
            ("object", relation["ocel:oid"]),
        )

    return graph


def _collapse_serving_events_in_connected_components(event_rows, relation_rows):
    serve_event_ids = {
        row["ocel:eid"]
        for row in event_rows
        if row["ocel:activity"] == SERVE_AS_RESOURCE_ACTIVITY
    }
    if not serve_event_ids:
        return event_rows, relation_rows

    event_by_id = {row["ocel:eid"]: row for row in event_rows}
    relation_df = pd.DataFrame(relation_rows, columns=RELATION_COLUMNS)
    if relation_df.empty:
        return event_rows, relation_rows

    object_types = (
        relation_df[["ocel:oid", "ocel:type"]]
        .drop_duplicates()
        .set_index("ocel:oid")["ocel:type"]
        .to_dict()
    )
    relations_by_event = (
        relation_df.groupby("ocel:eid")["ocel:oid"].agg(list).to_dict()
    )

    removed_event_ids = set()
    synthetic_event_rows = []
    synthetic_relation_rows = []
    synthetic_id_counter = count(1)

    for component_index, component in enumerate(
            nx.connected_components(_build_event_object_graph(relation_rows)),
            start=1,
    ):
        component_event_ids = [
            node_id for node_type, node_id in component if node_type == "event"
        ]
        component_serve_ids = [
            event_id for event_id in component_event_ids if event_id in serve_event_ids
        ]
        if not component_serve_ids:
            continue

        ordered_serve_ids = sorted(
            component_serve_ids,
            key=lambda event_id: (
                event_by_id[event_id]["ocel:timestamp"],
                event_by_id[event_id]["__order"],
                str(event_id),
            ),
        )
        serving_objects = list(
            dict.fromkeys(
                obj
                for event_id in ordered_serve_ids
                for obj in relations_by_event.get(event_id, [])
            )
        )
        if not serving_objects:
            continue

        removed_event_ids.update(component_serve_ids)

        first_event = event_by_id[ordered_serve_ids[0]]
        last_event = event_by_id[ordered_serve_ids[-1]]

        start_event_id = (
            f"serving_start__cc_{component_index}__{next(synthetic_id_counter)}"
        )
        end_event_id = (
            f"serving_end__cc_{component_index}__{next(synthetic_id_counter)}"
        )

        synthetic_event_rows.extend([
            {
                "ocel:eid": start_event_id,
                "ocel:activity": START_SERVING_ACTIVITY,
                "ocel:timestamp": first_event["ocel:timestamp"],
                "__order": first_event["__order"] - 0.25,
            },
            {
                "ocel:eid": end_event_id,
                "ocel:activity": END_SERVING_ACTIVITY,
                "ocel:timestamp": last_event["ocel:timestamp"],
                "__order": last_event["__order"] + 0.25,
            },
        ])

        for event_id, activity, timestamp in [
            (
                start_event_id,
                START_SERVING_ACTIVITY,
                first_event["ocel:timestamp"],
            ),
            (
                end_event_id,
                END_SERVING_ACTIVITY,
                last_event["ocel:timestamp"],
            ),
        ]:
            for obj in serving_objects:
                synthetic_relation_rows.append({
                    "ocel:eid": event_id,
                    "ocel:activity": activity,
                    "ocel:timestamp": timestamp,
                    "ocel:oid": obj,
                    "ocel:type": object_types[obj],
                    "ocel:qualifier": None,
                })

    retained_event_rows = [
        row for row in event_rows if row["ocel:eid"] not in removed_event_ids
    ]
    retained_relation_rows = [
        row for row in relation_rows if row["ocel:eid"] not in removed_event_ids
    ]

    return (
        retained_event_rows + synthetic_event_rows,
        retained_relation_rows + synthetic_relation_rows,
    )


def _create_layer_ocel(event_rows, relation_rows, o2o_graph_edges):
    if event_rows:
        events_df = pd.DataFrame(event_rows).sort_values(
            ["ocel:timestamp", "__order", "ocel:eid"],
            kind="stable",
        )
        event_order = events_df[["ocel:eid", "__order"]].copy()
        events_df = events_df[EVENT_COLUMNS].reset_index(drop=True)
    else:
        event_order = pd.DataFrame(columns=["ocel:eid", "__order"])
        events_df = pd.DataFrame(columns=EVENT_COLUMNS)

    if relation_rows:
        relations_df = pd.DataFrame(relation_rows, columns=RELATION_COLUMNS)
        relations_df = relations_df.merge(
            event_order,
            on="ocel:eid",
            how="left",
        ).sort_values(
            ["ocel:timestamp", "__order", "ocel:eid", "ocel:oid"],
            kind="stable",
        )
        relations_df = relations_df[RELATION_COLUMNS].reset_index(drop=True)
        used_objects = set(relations_df["ocel:oid"].unique())
    else:
        relations_df = pd.DataFrame(columns=RELATION_COLUMNS)
        used_objects = set()

    object_rows = [
        {
            "ocel:oid": obj,
            "ocel:type": relations_df.loc[
                relations_df["ocel:oid"] == obj,
                "ocel:type",
            ].iloc[0],
        }
        for obj in sorted(used_objects)
    ]
    objects_df = pd.DataFrame(object_rows, columns=OBJECT_COLUMNS)

    o2o_rows = [
        {
            "ocel:oid": source_obj,
            "ocel:oid_2": target_obj,
            "ocel:qualifier": None,
        }
        for source_obj, target_obj in o2o_graph_edges
        if source_obj in used_objects and target_obj in used_objects
    ]
    o2o_df = pd.DataFrame(o2o_rows, columns=O2O_COLUMNS)

    return OCEL(
        events=events_df,
        objects=objects_df,
        relations=relations_df,
        o2o=o2o_df,
    )


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
        lifted_activities = layer_to_activities.get(layer - 1, set())

        event_rows = []
        relation_rows = []

        for event_id, activity, timestamp, event_objects in event_records:
            if activity in selected_activities:
                target_activity = activity
            elif activity in lifted_activities:
                target_activity = SERVE_AS_RESOURCE_ACTIVITY
            else:
                continue

            selected_event_objects = [
                obj for obj in event_objects
                if object_to_type[obj] in selected_object_types
            ]
            if not selected_event_objects:
                continue

            event_rows.append({
                "ocel:eid": event_id,
                "ocel:activity": target_activity,
                "ocel:timestamp": timestamp,
                "__order": len(event_rows),
            })

            for obj in dict.fromkeys(selected_event_objects):
                relation_rows.append({
                    "ocel:eid": event_id,
                    "ocel:activity": target_activity,
                    "ocel:timestamp": timestamp,
                    "ocel:oid": obj,
                    "ocel:type": object_to_type[obj],
                    "ocel:qualifier": None,
                })

        event_rows, relation_rows = _collapse_serving_events_in_connected_components(
            event_rows,
            relation_rows,
        )
        layer_ocel = _create_layer_ocel(
            event_rows,
            relation_rows,
            ocel.o2o_graph_edges,
        )

        ocpn = None
        if not layer_ocel.events.empty and not layer_ocel.relations.empty:
            ocpn = pm4py.discover_oc_petri_net(layer_ocel)

        actual_activities = []
        if not layer_ocel.events.empty:
            actual_activities = sorted(layer_ocel.events["ocel:activity"].unique())

        discovered_models[layer] = {
            "object_types": sorted(selected_object_types),
            "activities": actual_activities,
            "lifted_activities": sorted(lifted_activities),
            "ocel": layer_ocel,
            "ocpn": ocpn,
        }

    return dict(activity_to_layer), discovered_models
