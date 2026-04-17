from collections import defaultdict
from numbers import Integral

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL

from .totem import _prepare_totem_data, get_all_event_objects
from .subprocess_detection import detect_subprocess_components


def _ocel_event_id_column(ocel):
    if "_eventId" in ocel.events.columns:
        return "_eventId"
    if "ocel:eid" in ocel.events.columns:
        return "ocel:eid"
    raise KeyError("Unsupported OCEL event id column.")


def _iter_ocel_o2o_edges(ocel):
    if hasattr(ocel, "o2o_graph_edges"):
        return tuple(ocel.o2o_graph_edges)

    o2o_df = getattr(ocel, "o2o", None)
    if isinstance(o2o_df, pd.DataFrame) and {"ocel:oid", "ocel:oid_2"} <= set(o2o_df.columns):
        return tuple(zip(o2o_df["ocel:oid"], o2o_df["ocel:oid_2"]))

    return ()


def _extract_ocel_filtering_context(ocel):
    object_to_type = {}

    objects_df = getattr(ocel, "objects", None)
    if isinstance(objects_df, pd.DataFrame) and {"ocel:oid", "ocel:type"} <= set(objects_df.columns):
        for object_id, object_type in zip(objects_df["ocel:oid"], objects_df["ocel:type"]):
            object_to_type[object_id] = object_type
    else:
        _, _, _, type_to_object = _prepare_totem_data(ocel)
        for object_type, objects in type_to_object.items():
            for object_id in objects:
                object_to_type[object_id] = object_type

    event_records = []
    event_id_column = _ocel_event_id_column(ocel)
    relations_df = getattr(ocel, "relations", None)
    relations_by_event = defaultdict(list)
    if isinstance(relations_df, pd.DataFrame) and {"ocel:eid", "ocel:oid"} <= set(relations_df.columns):
        for event_id, object_id in zip(relations_df["ocel:eid"], relations_df["ocel:oid"]):
            if object_id in object_to_type:
                relations_by_event[event_id].append(object_id)

    event_ids = list(ocel.events[event_id_column])
    if "ocel:activity" in ocel.events.columns:
        activity_by_event = dict(zip(event_ids, ocel.events["ocel:activity"]))
        timestamp_by_event = (
            dict(zip(event_ids, ocel.events["ocel:timestamp"]))
            if "ocel:timestamp" in ocel.events.columns
            else {}
        )

        for event_id in event_ids:
            event_objects = tuple(dict.fromkeys(relations_by_event.get(event_id, ())))
            if not event_objects and hasattr(ocel, "get_value"):
                event_objects = tuple(
                    obj
                    for obj in get_all_event_objects(ocel, event_id)
                    if obj in object_to_type
                )
            if not event_objects:
                continue

            event_records.append((
                event_id,
                activity_by_event[event_id],
                timestamp_by_event.get(event_id),
                event_objects,
            ))
    else:
        for event_id in event_ids:
            event_objects = tuple(
                obj
                for obj in get_all_event_objects(ocel, event_id)
                if obj in object_to_type
            )
            if not event_objects:
                continue

            event_records.append((
                event_id,
                ocel.get_event_activity(event_id),
                ocel.get_event_timestamp(event_id),
                event_objects,
            ))

    return object_to_type, event_records


def _build_ocel_filtering_context(ocel):
    object_to_type, event_records = _extract_ocel_filtering_context(ocel)
    return {
        "object_to_type": dict(object_to_type),
        "event_records": tuple(event_records),
        "o2o_edges": tuple(_iter_ocel_o2o_edges(ocel)),
    }


def _filter_ocel_filtering_context(filtering_context, selected_activities):
    selected_activities = set(selected_activities)
    filtered_event_records = tuple(
        event_record
        for event_record in filtering_context["event_records"]
        if event_record[1] in selected_activities
    )
    used_objects = {
        obj
        for _, _, _, event_objects in filtered_event_records
        for obj in event_objects
    }
    object_to_type = {
        obj: filtering_context["object_to_type"][obj]
        for obj in used_objects
        if obj in filtering_context["object_to_type"]
    }
    o2o_edges = tuple(
        (source_obj, target_obj)
        for source_obj, target_obj in filtering_context["o2o_edges"]
        if source_obj in used_objects and target_obj in used_objects
    )
    return {
        "object_to_type": object_to_type,
        "event_records": filtered_event_records,
        "o2o_edges": o2o_edges,
    }


def _merge_ocel_filtering_contexts(*filtering_contexts):
    merged_object_to_type = {}
    merged_event_records = {}
    merged_event_order = []
    merged_event_object_sets = {}
    merged_o2o_edges = []
    seen_o2o_edges = set()

    for filtering_context in filtering_contexts:
        if not filtering_context:
            continue

        merged_object_to_type.update(filtering_context.get("object_to_type", {}))

        for edge in filtering_context.get("o2o_edges", ()):
            if edge in seen_o2o_edges:
                continue
            seen_o2o_edges.add(edge)
            merged_o2o_edges.append(edge)

        for event_id, activity, timestamp, event_objects in filtering_context.get("event_records", ()):
            if event_id not in merged_event_records:
                merged_event_records[event_id] = [activity, timestamp, []]
                merged_event_object_sets[event_id] = set()
                merged_event_order.append(event_id)
            else:
                current_activity, current_timestamp, _ = merged_event_records[event_id]
                if current_activity is None:
                    merged_event_records[event_id][0] = activity
                if current_timestamp is None:
                    merged_event_records[event_id][1] = timestamp

            current_objects = merged_event_records[event_id][2]
            current_object_set = merged_event_object_sets[event_id]
            for obj in event_objects:
                if obj in current_object_set:
                    continue
                current_object_set.add(obj)
                current_objects.append(obj)

    return {
        "object_to_type": merged_object_to_type,
        "event_records": tuple(
            (
                event_id,
                merged_event_records[event_id][0],
                merged_event_records[event_id][1],
                tuple(merged_event_records[event_id][2]),
            )
            for event_id in merged_event_order
            if merged_event_records[event_id][2]
        ),
        "o2o_edges": tuple(merged_o2o_edges),
    }


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


def _build_layer_ocel(ocel, event_records, object_to_type, selected_object_types, selected_activities):
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
        for source_obj, target_obj in _iter_ocel_o2o_edges(ocel)
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

    return layer_ocel, included_event_records, sorted(included_activities)


def _build_ocel_from_filtering_context(filtering_context):
    class _FilteringContextSource:
        def __init__(self, o2o_edges):
            self.o2o_graph_edges = tuple(o2o_edges)

    source = _FilteringContextSource(filtering_context.get("o2o_edges", ()))
    selected_object_types = set(filtering_context.get("object_to_type", {}).values())
    selected_activities = {
        activity
        for _, activity, _, _ in filtering_context.get("event_records", ())
    }
    layer_ocel, _, _ = _build_layer_ocel(
        source,
        filtering_context.get("event_records", ()),
        filtering_context.get("object_to_type", {}),
        selected_object_types,
        selected_activities,
    )
    return layer_ocel


def _discover_ocpn_with_subprocess_components(layer_ocel, activity_to_layer, reference_layer):
    ocpn = _discover_ocpn(layer_ocel)
    if ocpn is None:
        return None, []

    subprocess_components = detect_subprocess_components(
        ocpn,
        activity_to_layer,
        reference_layer,
    )
    return ocpn, subprocess_components


def _discover_ocpn(layer_ocel):
    if layer_ocel.events.empty or layer_ocel.relations.empty:
        return None
    return pm4py.discover_oc_petri_net(layer_ocel)


def _component_visible_activities(component):
    return tuple(sorted({
        transition["label"]
        for transition in component.get("transitions", [])
        if transition["kind"] == "activity" and transition["label"] is not None
    }))


def _build_pruning_candidates(subprocess_components, included_activities, activity_to_layer, reference_layer):
    candidates = []
    covered_lower_layer_activities = set()

    for component in subprocess_components:
        component_activities = tuple(
            activity
            for activity in _component_visible_activities(component)
            if activity_to_layer.get(activity, float("inf")) < reference_layer
        )
        if not component_activities:
            continue

        candidates.append({
            "kind": "subprocess",
            "id": component["id"],
            "activities": component_activities,
            "component": component,
        })
        covered_lower_layer_activities.update(component_activities)

    for activity in sorted(included_activities):
        if activity_to_layer.get(activity, float("inf")) >= reference_layer:
            continue
        if activity in covered_lower_layer_activities:
            continue

        candidates.append({
            "kind": "activity",
            "id": activity,
            "activities": (activity,),
            "component": None,
        })

    return candidates


def _compute_information_loss(current_model, current_ocel, lower_layer_ocel, component):
    return 1


def _compute_simplicity_gain(current_model, current_ocel, lower_layer_ocel, component):
    return 0


def _select_best_pruning_candidate(current_model, current_ocel, lower_layer_ocel, candidates):
    best_candidate = None
    best_score = float("-inf")

    for candidate in candidates:
        score = _compute_simplicity_gain(
            current_model,
            current_ocel,
            lower_layer_ocel,
            candidate,
        ) - _compute_information_loss(
            current_model,
            current_ocel,
            lower_layer_ocel,
            candidate,
        )
        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate, best_score


def discover_models_for_hierarchy(ocel, solution, layer_context=None):
    object_to_type, event_records = _extract_ocel_filtering_context(ocel)
    activity_to_layer = {}
    layer_to_object_types = defaultdict(set)

    for obj_type, layer in solution.items():
        layer_to_object_types[layer].add(obj_type)

    for event_id, activity, timestamp, event_objects in event_records:
        event_layer = min(solution[object_to_type[obj]] for obj in event_objects)
        current_layer = activity_to_layer.get(activity)
        if current_layer is None or event_layer < current_layer:
            activity_to_layer[activity] = event_layer

    discovered_layers = sorted(layer_to_object_types)
    layer_context_by_layer = _normalize_layer_context(discovered_layers, layer_context)

    discovered_models = {}
    for layer_index, layer in enumerate(discovered_layers):
        selected_object_types = layer_to_object_types[layer]
        lower_layer_ocel = None
        if layer_index > 0:
            lower_layer = discovered_layers[layer_index - 1]
            lower_layer_ocel = discovered_models[lower_layer]["ocel"]
        active_activities = {
            activity
            for activity, activity_layer in activity_to_layer.items()
            if activity_layer <= layer and layer - activity_layer <= layer_context_by_layer[layer]
        }

        while True:
            layer_ocel, included_event_records, included_activities = _build_layer_ocel(
                ocel,
                event_records,
                object_to_type,
                selected_object_types,
                active_activities,
            )
            ocpn, iteration_components = _discover_ocpn_with_subprocess_components(
                layer_ocel,
                activity_to_layer,
                layer,
            )

            candidates = _build_pruning_candidates(
                iteration_components,
                included_activities,
                activity_to_layer,
                layer,
            )
            best_candidate, best_score = _select_best_pruning_candidate(
                ocpn,
                layer_ocel,
                lower_layer_ocel,
                candidates,
            )

            if best_candidate is None or best_score <= 0:
                break

            remaining_activities = active_activities - set(best_candidate["activities"])
            if remaining_activities == active_activities:
                break
            active_activities = remaining_activities

        activity_resources = _discover_activity_resources(
            included_event_records,
            object_to_type,
            solution,
            layer,
        )
        highlighted_activities = sorted(
            activity
            for activity in included_activities
            if activity_to_layer[activity] == layer - 1
        )
        subprocess_components = []
        if ocpn is not None:
            subprocess_components = detect_subprocess_components(
                ocpn,
                activity_to_layer,
                layer,
            )

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
            "subprocess_components": subprocess_components,
        }

    return dict(activity_to_layer), discovered_models
