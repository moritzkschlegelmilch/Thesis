from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import is_dataclass, replace
from itertools import product
from numbers import Integral
import random
import sys

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils
from tqdm import tqdm

from .totem import _prepare_totem_data, get_all_event_objects
from .subprocess_detection import (
    _clone_ocpn,
    _component_boundary_places,
    _sort_petri_net_node,
    collapse_sub_processes,
    detect_subprocess_components,
)

# Cache filtering contexts by OCEL object identity.
_OCEL_FILTERING_CONTEXT_CACHE: dict[int, dict] = {}


@contextmanager
def _progress_step(description, *, enabled=False, progress_bar=None):
    if progress_bar is not None:
        progress_bar.set_postfix_str(description, refresh=False)

    if not enabled:
        yield
        return

    step_bar = tqdm(
        total=1,
        desc=description,
        leave=False,
        file=sys.stdout,
        bar_format="{desc}: {bar} [{elapsed}]",
    )
    try:
        yield
        step_bar.update(1)
    finally:
        step_bar.close()


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
    cache_key = id(ocel)

    if cache_key not in _OCEL_FILTERING_CONTEXT_CACHE:
        object_to_type, event_records = _extract_ocel_filtering_context(ocel)
        _OCEL_FILTERING_CONTEXT_CACHE[cache_key] = {
            "object_to_type": dict(object_to_type),
            "event_records": tuple(event_records),
            "o2o_edges": tuple(_iter_ocel_o2o_edges(ocel)),
        }

    return _OCEL_FILTERING_CONTEXT_CACHE[cache_key]


def clear_ocel_filtering_context_cache(ocel=None):
    if ocel is None:
        _OCEL_FILTERING_CONTEXT_CACHE.clear()
    else:
        _OCEL_FILTERING_CONTEXT_CACHE.pop(id(ocel), None)


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


def _construct_ocel(event_rows, object_rows, relation_rows, o2o_rows):
    event_rows = [
        {
            **row,
            "ocel:timestamp": _normalize_pm4py_timestamp(row.get("ocel:timestamp")),
        }
        for row in event_rows
    ]
    relation_rows = [
        {
            **row,
            "ocel:timestamp": _normalize_pm4py_timestamp(row.get("ocel:timestamp")),
        }
        for row in relation_rows
    ]
    o2o_df = pd.DataFrame(
        o2o_rows,
        columns=["ocel:oid", "ocel:oid_2", "ocel:qualifier"],
    )
    kwargs = {
        "events": pd.DataFrame(
            event_rows,
            columns=["ocel:eid", "ocel:activity", "ocel:timestamp"],
        ),
        "objects": pd.DataFrame(
            object_rows,
            columns=["ocel:oid", "ocel:type"],
        ),
        "relations": pd.DataFrame(
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
    }

    try:
        built_ocel = OCEL(
            o2o=o2o_df,
            **kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'o2o'" not in str(exc):
            raise
        built_ocel = OCEL(**kwargs)

    if not hasattr(built_ocel, "o2o_graph_edges"):
        built_ocel.o2o_graph_edges = tuple(
            (row["ocel:oid"], row["ocel:oid_2"])
            for row in o2o_rows
        )

    return built_ocel


def _normalize_pm4py_timestamp(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return pd.NaT
    if isinstance(timestamp, pd.Timestamp):
        return timestamp
    if isinstance(timestamp, (int, float)):
        return pd.to_datetime(timestamp, unit="s")
    return pd.to_datetime(timestamp)


def _build_projected_ocel(
    ocel,
    event_records,
    object_to_type,
    selected_activities,
    *,
    selected_object_types=None,
):
    event_rows = []
    relation_rows = []
    used_objects = set()
    included_event_records = []
    included_activities = set()
    selected_object_types = (
        set(selected_object_types)
        if selected_object_types is not None
        else None
    )

    for event_id, activity, timestamp, event_objects in event_records:
        if activity not in selected_activities:
            continue

        if selected_object_types is None:
            selected_event_objects = list(dict.fromkeys(event_objects))
        else:
            selected_event_objects = [
                obj
                for obj in event_objects
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

    layer_ocel = _construct_ocel(
        event_rows,
        object_rows,
        relation_rows,
        o2o_rows,
    )

    return layer_ocel, included_event_records, sorted(included_activities)


def _build_layer_ocel(ocel, event_records, object_to_type, selected_object_types, selected_activities):
    return _build_projected_ocel(
        ocel,
        event_records,
        object_to_type,
        selected_activities,
        selected_object_types=selected_object_types,
    )


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


def _discover_ocpn_with_subprocess_components(
    layer_ocel,
    activity_to_layer,
    reference_layer,
    show_progress=False,
    progress_prefix=None,
    progress_bar=None,
):
    prefix = progress_prefix or f"Layer {reference_layer}"

    with _progress_step(
        f"{prefix}: PM4Py OCPN discovery",
        enabled=show_progress,
        progress_bar=progress_bar,
    ):
        ocpn = _discover_ocpn(layer_ocel)
    if ocpn is None:
        return None, []

    with _progress_step(
        f"{prefix}: subprocess detection",
        enabled=show_progress,
        progress_bar=progress_bar,
    ):
        subprocess_components = detect_subprocess_components(
            ocpn,
            activity_to_layer,
            reference_layer,
        )
    return ocpn, subprocess_components


def _annotate_subprocess_component_index(component, index):
    global_id = f"subprocess_{index}"

    if isinstance(component, dict):
        annotated_component = dict(component)
        annotated_component["index"] = index
        annotated_component["global_id"] = global_id
        return annotated_component

    if is_dataclass(component):
        dataclass_fields = getattr(component, "__dataclass_fields__", {})
        updates = {}
        if "index" in dataclass_fields:
            updates["index"] = index
        if "global_id" in dataclass_fields:
            updates["global_id"] = global_id
        if updates:
            return replace(component, **updates)

    return component


def _assign_subprocess_indices_across_layers(discovered_models):
    indexed_models = {}
    next_index = 1

    for layer in sorted(discovered_models):
        model_data = dict(discovered_models[layer])
        original_components = model_data.get("subprocess_components")
        component_sequence = original_components or ()
        indexed_components = [
            _annotate_subprocess_component_index(component, index)
            for index, component in enumerate(component_sequence, start=next_index)
        ]
        next_index += len(indexed_components)

        if isinstance(original_components, tuple):
            model_data["subprocess_components"] = tuple(indexed_components)
        else:
            model_data["subprocess_components"] = indexed_components

        indexed_models[layer] = model_data

    return indexed_models


def _discover_ocpn(layer_ocel):
    if _is_empty_table(layer_ocel.events) or _is_empty_table(layer_ocel.relations):
        return None
    return pm4py.discover_oc_petri_net(layer_ocel)


def _is_empty_table(table):
    if table is None:
        return True
    empty = getattr(table, "empty", None)
    if empty is not None:
        return bool(empty)
    try:
        return len(table) == 0
    except TypeError:
        return False


def _component_visible_activities(component):
    if component is None:
        return ()
    if hasattr(component, "activities"):
        return tuple(sorted(activity for activity in component.activities if activity is not None))
    return tuple(sorted({
        transition["label"]
        for transition in component.get("transitions", [])
        if transition["kind"] == "activity" and transition["label"] is not None
    }))


def _component_transition_count(component):
    if component is None:
        return 0
    if hasattr(component, "get"):
        return len(component.get("transition_keys", ()))
    return len(getattr(component, "transition_keys", ()))


def _component_place_count(component):
    if component is None:
        return 0
    if hasattr(component, "get"):
        return len(component.get("place_keys", ()))
    return len(getattr(component, "place_keys", ()))


def _component_boundary_adjacent_activities(component, activity_to_layer, reference_layer):
    adjacent_activities = set()

    for boundary_place_data in _component_boundary_places(component).values():
        for place in boundary_place_data["input"]:
            for arc in place.in_arcs:
                if not isinstance(arc.source, PetriNet.Transition):
                    continue
                if arc.source.label is None:
                    continue
                if activity_to_layer.get(arc.source.label, float("inf")) < reference_layer:
                    adjacent_activities.add(arc.source.label)

        for place in boundary_place_data["output"]:
            for arc in place.out_arcs:
                if not isinstance(arc.target, PetriNet.Transition):
                    continue
                if arc.target.label is None:
                    continue
                if activity_to_layer.get(arc.target.label, float("inf")) < reference_layer:
                    adjacent_activities.add(arc.target.label)

    return adjacent_activities


def _candidate_subprocess_activity_groups(candidate):
    groups = candidate.get("subprocess_activity_groups")
    if groups is not None:
        return tuple(
            tuple(group)
            for group in groups
            if group
        )

    if candidate.get("kind") == "subprocess":
        component_activities = tuple(candidate.get("activities", ()))
        if component_activities:
            return (component_activities,)

    return ()


def _match_subprocess_components_by_activity_groups(component_and_edge_ocpn, subprocess_activity_groups):
    if component_and_edge_ocpn is None or not subprocess_activity_groups:
        return ()

    subprocess_activities = {
        activity
        for group in subprocess_activity_groups
        for activity in group
    }
    activity_to_layer = {
        activity: 1 if activity in subprocess_activities else 2
        for activity in component_and_edge_ocpn.get("activities", ())
    }
    collapsed_components = list(detect_subprocess_components(
        component_and_edge_ocpn,
        activity_to_layer,
        reference_layer=2,
    ))

    matching_components = []
    for activity_group in subprocess_activity_groups:
        matches = [
            candidate_component
            for candidate_component in collapsed_components
            if set(_component_visible_activities(candidate_component)) == set(activity_group)
        ]
        if not matches:
            continue

        matches.sort(
            key=lambda candidate_component: (
                -_component_transition_count(candidate_component),
                -_component_place_count(candidate_component),
            ),
        )
        chosen_component = matches[0]
        matching_components.append(chosen_component)
        collapsed_components.remove(chosen_component)

    return tuple(matching_components)


def _build_pruning_candidates(subprocess_components, included_activities, activity_to_layer, reference_layer):
    candidates = []
    subprocess_candidates = []
    covered_subprocess_activities = set()
    standalone_lower_layer_activities = {
        activity
        for activity in included_activities
        if activity_to_layer.get(activity, float("inf")) < reference_layer
    }

    for component in subprocess_components:
        component_activities = tuple(
            activity
            for activity in _component_visible_activities(component)
            if activity_to_layer.get(activity, float("inf")) < reference_layer
        )
        if not component_activities:
            continue

        subprocess_candidates.append({
            "kind": "subprocess",
            "id": getattr(component, "id", None),
            "activities": component_activities,
            "component": component,
            "subprocess_components": (component,),
            "subprocess_activity_groups": (component_activities,),
            "boundary_adjacent_activities": _component_boundary_adjacent_activities(
                component,
                activity_to_layer,
                reference_layer,
            ),
        })
        covered_subprocess_activities.update(component_activities)

    standalone_activities = standalone_lower_layer_activities - covered_subprocess_activities
    activity_to_subprocess_indices = defaultdict(list)
    for index, candidate in enumerate(subprocess_candidates):
        attached_activities = tuple(sorted(
            activity
            for activity in candidate.pop("boundary_adjacent_activities", ())
            if activity in standalone_activities
        ))
        candidate["attached_activities"] = attached_activities
        for activity in attached_activities:
            activity_to_subprocess_indices[activity].append(index)

    parent = list(range(len(subprocess_candidates)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for indices in activity_to_subprocess_indices.values():
        for index in indices[1:]:
            union(indices[0], index)

    grouped_indices = defaultdict(list)
    for index in range(len(subprocess_candidates)):
        grouped_indices[find(index)].append(index)

    covered_lower_layer_activities = set()
    for root_index in sorted(grouped_indices):
        group = grouped_indices[root_index]
        group_subprocess_components = []
        group_subprocess_activity_groups = []
        group_activities = set()
        group_ids = []

        for index in group:
            candidate = subprocess_candidates[index]
            group_subprocess_components.extend(candidate.get("subprocess_components", ()))
            group_subprocess_activity_groups.extend(candidate.get("subprocess_activity_groups", ()))
            group_activities.update(candidate.get("activities", ()))
            group_activities.update(candidate.get("attached_activities", ()))
            if candidate.get("id") is not None:
                group_ids.append(candidate["id"])

        grouped_candidate_activities = tuple(sorted(group_activities))
        candidates.append({
            "kind": "subprocess",
            "id": group_ids[0] if len(group_ids) == 1 else None,
            "activities": grouped_candidate_activities,
            "component": group_subprocess_components[0] if len(group_subprocess_components) == 1 else None,
            "subprocess_components": tuple(group_subprocess_components),
            "subprocess_activity_groups": tuple(group_subprocess_activity_groups),
        })
        covered_lower_layer_activities.update(grouped_candidate_activities)

    for activity in sorted(standalone_activities):
        if activity in covered_lower_layer_activities:
            continue

        candidates.append({
            "kind": "activity",
            "id": activity,
            "activities": (activity,),
            "component": None,
        })

    return candidates


def _merge_pruning_candidates(candidates):
    candidates = tuple(candidates)
    if not candidates:
        return None

    activities = set()
    subprocess_components = []
    subprocess_activity_groups = []
    candidate_ids = []

    for candidate in candidates:
        activities.update(candidate.get("activities", ()))
        subprocess_components.extend(candidate.get("subprocess_components", ()))
        subprocess_activity_groups.extend(candidate.get("subprocess_activity_groups", ()))
        candidate_id = candidate.get("id")
        if candidate_id is not None:
            candidate_ids.append(str(candidate_id))

    merged_candidate = {
        "kind": "candidate_set" if len(candidates) > 1 else candidates[0].get("kind"),
        "id": "+".join(candidate_ids) if candidate_ids else None,
        "activities": tuple(sorted(activities)),
        "component": None,
        "candidates": candidates,
    }
    if subprocess_components:
        merged_candidate["subprocess_components"] = tuple(subprocess_components)
    if subprocess_activity_groups:
        merged_candidate["subprocess_activity_groups"] = tuple(subprocess_activity_groups)

    return merged_candidate


def _prepare_pruning_candidate_context(
    current_model,
    current_ocel,
    lower_layer_ocel,
    component_activities,
    *,
    build_component_and_edge_ocel,
):
    from .component_deletion_impact import (
        discover_component_and_edge_models,
    )

    component_and_edge_ocel, _, component_and_edge_ocpn, edge_only_ocpn = discover_component_and_edge_models(
        current_ocel,
        current_model,
        component_activities,
        lower_layer_ocel=lower_layer_ocel,
    )

    if not build_component_and_edge_ocel:
        component_and_edge_ocel = None

    return {
        "component_activities": component_activities,
        "component_and_edge_ocpn": component_and_edge_ocpn,
        "edge_only_ocpn": edge_only_ocpn,
        "component_and_edge_ocel": component_and_edge_ocel,
    }


def _merge_ocpn_models(*ocpns):
    merged_activities = set()
    merged_petri_nets = {}
    merged_double_arcs = {}

    for ocpn in ocpns:
        if ocpn is None:
            continue

        merged_activities.update(
            str(activity)
            for activity in ocpn.get("activities", ())
            if activity is not None
        )

        for object_type, petri_net_data in ocpn.get("petri_nets", {}).items():
            if object_type in merged_petri_nets:
                raise ValueError(
                    f"Cannot merge OCPNs with duplicate object type {object_type!r}."
                )
            merged_petri_nets[object_type] = petri_net_data

        for object_type, metadata in ocpn.get("double_arcs_on_activity", {}).items():
            if object_type in merged_double_arcs:
                raise ValueError(
                    f"Cannot merge OCPNs with duplicate double-arc metadata for {object_type!r}."
                )
            merged_double_arcs[object_type] = dict(metadata)

    if not merged_petri_nets:
        return None

    return {
        "activities": sorted(merged_activities),
        "object_types": set(merged_petri_nets),
        "petri_nets": merged_petri_nets,
        "tbr_results": {},
        "double_arcs_on_activity": {
            object_type: merged_double_arcs.get(object_type, {})
            for object_type in merged_petri_nets
        },
    }


def _build_model_enabled_by_context(
    quality,
    prepared,
    selected_contexts=None,
    *,
    show_progress=False,
    progress_desc="Replaying contexts",
):
    model_enabled = {}

    selected_context_set = None if selected_contexts is None else set(selected_contexts)
    if selected_context_set is None:
        replay_items = tuple(prepared["replay"].items())
    else:
        replay_items = tuple(
            (ctx, events)
            for ctx, events in prepared["replay"].items()
            if ctx in selected_context_set
        )

    if show_progress:
        replay_items = tqdm(
            replay_items,
            total=len(replay_items),
            desc=progress_desc,
            leave=False,
            file=sys.stdout,
        )

    for ctx, events in replay_items:
        enabled = set()
        for event in events:
            enabled.update(quality._replay(event))
        model_enabled[ctx] = frozenset(enabled)

    return model_enabled


def _context_weights(prepared):
    weights = defaultdict(int)

    for event_id in prepared["events"]:
        weights[prepared["ctx"][event_id]] += 1

    return {
        ctx: weight
        for ctx, weight in weights.items()
    }


def _sample_context_draw_counts(context_weights, sample_size, random_seed=None):
    if sample_size is None:
        return None
    if sample_size <= 0:
        raise ValueError("precision_context_sample_size must be positive.")
    if not context_weights:
        return {}

    contexts = tuple(context_weights)
    weights = tuple(context_weights[ctx] for ctx in contexts)
    rng = random.Random(random_seed)
    return Counter(rng.choices(contexts, weights=weights, k=sample_size))


def _precision_from_prepared(prepared, model_enabled_by_context, context_weights=None):
    context_weights = context_weights or _context_weights(prepared)
    precision_sum = 0.0
    precision_weight = 0

    for ctx, weight in context_weights.items():
        log_enabled = prepared["log"].get(ctx, frozenset())
        model_enabled = model_enabled_by_context.get(ctx, frozenset())
        overlap = log_enabled & model_enabled

        if not model_enabled or not overlap:
            continue

        precision_sum += weight * (len(overlap) / len(model_enabled))
        precision_weight += weight

    return precision_sum / precision_weight if precision_weight else 0.0


def _enabled_mass_by_context(context_weights, model_enabled_by_context):
    return float(
        sum(
            weight * len(model_enabled_by_context.get(ctx, ()))
            for ctx, weight in context_weights.items()
        )
    )


def _reduced_log_enabled_by_original_context(
    original_prepared,
    reduced_prepared,
    selected_contexts=None,
):
    reduced_enabled_by_context = defaultdict(set)
    selected_contexts = None if selected_contexts is None else set(selected_contexts)

    for event_id in original_prepared["events"]:
        original_ctx = original_prepared["ctx"][event_id]
        if selected_contexts is not None and original_ctx not in selected_contexts:
            continue
        reduced_ctx = reduced_prepared.get("ctx", {}).get(event_id)
        if reduced_ctx is None:
            continue

        reduced_enabled_by_context[original_ctx].update(
            reduced_prepared.get("log", {}).get(reduced_ctx, ())
        )

    return {
        ctx: frozenset(labels)
        for ctx, labels in reduced_enabled_by_context.items()
    }


def _build_terminal_states_by_context(
    quality,
    prepared,
    selected_contexts=None,
    *,
    show_progress=False,
    progress_desc="Replaying contexts",
):
    terminal_states_by_context = {}

    selected_context_set = None if selected_contexts is None else set(selected_contexts)
    if selected_context_set is None:
        replay_items = tuple(prepared["replay"].items())
    else:
        replay_items = tuple(
            (ctx, events)
            for ctx, events in prepared["replay"].items()
            if ctx in selected_context_set
        )

    if show_progress:
        replay_items = tqdm(
            replay_items,
            total=len(replay_items),
            desc=progress_desc,
            leave=False,
            file=sys.stdout,
        )

    for ctx, events in replay_items:
        terminal_states = {}
        for event in events:
            for state in quality._replay_terminal_states(event):
                terminal_states.setdefault(quality._state_key(state), state)
        terminal_states_by_context[ctx] = tuple(terminal_states.values())

    return terminal_states_by_context


def _enabled_labels_by_context_from_terminal_states(
    quality,
    terminal_states_by_context,
):
    model_enabled = {}

    for ctx, terminal_states in terminal_states_by_context.items():
        enabled = set()
        for state in terminal_states:
            enabled.update(quality._enabled_labels(state))
        model_enabled[ctx] = frozenset(enabled)

    return model_enabled


def _enabled_labels_with_virtual_reduction(
    quality,
    state,
    affected_labels,
    removed_object_types,
    enabled_label_cache,
):
    cache_key = (
        quality._state_key(state),
        affected_labels,
        removed_object_types,
    )
    if cache_key in enabled_label_cache:
        return enabled_label_cache[cache_key]

    mc = quality._ensure_model_cache()
    enabled = set()

    for label in sorted(affected_labels):
        by_type = mc.visible_transitions_by_label_type.get(label)
        if not by_type:
            continue

        effective_types = tuple(
            sorted(
                object_type
                for object_type in by_type
                if object_type not in removed_object_types
            )
        )
        if not effective_types:
            enabled.add(label)
            continue

        transition_choices = [tuple(by_type[object_type]) for object_type in effective_types]
        for combo in product(*transition_choices):
            if all(quality._has_some_binding(state, transition) for transition in combo):
                enabled.add(label)
                break

    enabled_label_cache[cache_key] = frozenset(enabled)
    return enabled_label_cache[cache_key]


def _virtual_reduction_enabled_labels_by_context(precision_bundle, affected_labels):
    affected_labels = frozenset(str(label) for label in affected_labels if label is not None)
    if not affected_labels:
        return {}

    candidate_enabled_cache = precision_bundle.setdefault("candidate_enabled_cache", {})
    if affected_labels in candidate_enabled_cache:
        return candidate_enabled_cache[affected_labels]

    quality = precision_bundle["quality"]
    removed_object_types = precision_bundle["removed_object_types"]
    enabled_label_cache = {}
    reduced_enabled_by_context = {}
    show_progress = precision_bundle.get("show_progress", False)
    terminal_states_by_context = precision_bundle.get("original_terminal_states_by_context", {})
    terminal_state_items = tuple(terminal_states_by_context.items())

    if show_progress:
        terminal_state_items = tqdm(
            terminal_state_items,
            total=len(terminal_state_items),
            desc="Checking reduced precision contexts",
            leave=False,
            file=sys.stdout,
        )

    for ctx, terminal_states in terminal_state_items:
        enabled = set()
        for state in terminal_states:
            enabled.update(
                _enabled_labels_with_virtual_reduction(
                    quality,
                    state,
                    affected_labels,
                    removed_object_types,
                    enabled_label_cache,
                )
            )
        reduced_enabled_by_context[ctx] = frozenset(enabled)

    candidate_enabled_cache[affected_labels] = reduced_enabled_by_context
    return reduced_enabled_by_context


def _build_precision_reference_bundle(
    source_ocel,
    event_records,
    object_to_type,
    activity_to_layer,
    reference_layer,
    removed_object_types,
    current_layer_ocpn,
    previous_layer_ocpn,
    *,
    precision_context_sample_size=None,
    precision_context_depth=5,
    precision_context_sample_seed=None,
    show_progress=False,
):
    from .net_quality import NetQuality

    if source_ocel is None or current_layer_ocpn is None or previous_layer_ocpn is None:
        return None

    projected_activities = {
        activity
        for activity, layer in activity_to_layer.items()
        if layer in {reference_layer - 1, reference_layer}
    }
    if not projected_activities:
        return None

    projected_previous_layer_ocpn = _project_ocpn_to_activities(
        previous_layer_ocpn,
        projected_activities,
    )
    projected_current_layer_ocpn = _project_ocpn_to_activities(
        current_layer_ocpn,
        projected_activities,
    )
    merged_ocpn = _merge_ocpn_models(
        projected_previous_layer_ocpn,
        projected_current_layer_ocpn,
    )
    if merged_ocpn is None:
        return None

    merged_ocel, _, _ = _build_projected_ocel(
        source_ocel,
        event_records,
        object_to_type,
        projected_activities,
        selected_object_types=None,
    )
    if merged_ocel.events.empty or merged_ocel.relations.empty:
        return None

    quality = NetQuality(
        merged_ocpn,
        merged_ocel,
        precision_context_sample_size=precision_context_sample_size,
        precision_context_depth=precision_context_depth,
        random_seed=precision_context_sample_seed,
    )
    prepared_original = quality._prepare_log(merged_ocel, show_progress=show_progress)
    exact_context_weights = _context_weights(prepared_original)
    sampled_context_weights = _sample_context_draw_counts(
        exact_context_weights,
        precision_context_sample_size,
        random_seed=precision_context_sample_seed,
    )
    context_weights = sampled_context_weights or exact_context_weights
    sampled_contexts = tuple(
        ctx
        for ctx in prepared_original["replay"]
        if ctx in context_weights
    )
    if show_progress:
        sampled_events = sum(context_weights.values())
        tqdm.write(
            "Precision reference sampling: "
            f"contexts={len(sampled_contexts)}/{len(exact_context_weights)}, "
            f"draws={sampled_events}, "
            f"mode={'sampled' if sampled_context_weights is not None else 'exact'}, "
            f"depth={'full' if precision_context_depth is None else precision_context_depth}",
            file=sys.stdout,
        )
    original_terminal_states_by_context = _build_terminal_states_by_context(
        quality,
        prepared_original,
        selected_contexts=sampled_contexts if sampled_context_weights is not None else None,
        show_progress=show_progress,
        progress_desc="Replaying precision reference contexts",
    )
    original_enabled_by_context = _enabled_labels_by_context_from_terminal_states(
        quality,
        original_terminal_states_by_context,
    )
    baseline_enabled_mass = _enabled_mass_by_context(
        context_weights,
        original_enabled_by_context,
    )
    original_precision = _precision_from_prepared(
        prepared_original,
        original_enabled_by_context,
        context_weights=context_weights,
    )

    remaining_object_types = set(object_to_type.values()) - set(removed_object_types)
    reduced_log_ocel, _, _ = _build_projected_ocel(
        source_ocel,
        event_records,
        object_to_type,
        projected_activities,
        selected_object_types=remaining_object_types,
    )
    if reduced_log_ocel.events.empty or reduced_log_ocel.relations.empty:
        reduced_prepared = {
            "events": tuple(),
            "ctx": {},
            "log": {},
            "replay": {},
        }
    else:
        reduced_prepared = quality._prepare_log(reduced_log_ocel, show_progress=show_progress)

    return {
        "quality": quality,
        "original_ocpn": merged_ocpn,
        "original_ocel": merged_ocel,
        "reduced_log_ocel": reduced_log_ocel,
        "prepared_original": prepared_original,
        "original_terminal_states_by_context": original_terminal_states_by_context,
        "original_enabled_by_context": original_enabled_by_context,
        "context_weights": context_weights,
        "baseline_enabled_mass": baseline_enabled_mass,
        "reduced_log_enabled_by_context": _reduced_log_enabled_by_original_context(
            prepared_original,
            reduced_prepared,
            selected_contexts=sampled_contexts if sampled_context_weights is not None else None,
        ),
        "removed_object_types": frozenset(str(object_type) for object_type in removed_object_types),
        "projected_activities": frozenset(str(activity) for activity in projected_activities),
        "precision": original_precision,
        "candidate_enabled_cache": {},
        "sampled_contexts": sampled_contexts if sampled_context_weights is not None else None,
        "show_progress": show_progress,
    }


def _compute_information_loss_from_prepared_context(prepared_context):
    if prepared_context is None:
        return 1

    from .net_quality import NetQuality

    component_and_edge_ocpn = prepared_context["component_and_edge_ocpn"]
    edge_only_ocpn = prepared_context["edge_only_ocpn"]
    component_and_edge_ocel = prepared_context["component_and_edge_ocel"]

    precision_with_component = 0.0
    if component_and_edge_ocpn is not None and component_and_edge_ocel is not None:
        precision_with_component = float(
            NetQuality(
                component_and_edge_ocpn,
                component_and_edge_ocel,
            ).precision()
        )

    precision_without_component = 0.0
    if edge_only_ocpn is not None:
        precision_without_component = float(
            NetQuality(
                edge_only_ocpn,
                component_and_edge_ocel,
            ).precision()
        )

    information_loss = 1 - (precision_without_component / precision_with_component)

    return information_loss


def _compute_information_loss(
    current_model,
    current_ocel,
    lower_layer_ocel,
    component,
    precision_bundle=None,
):
    component_activities = tuple(component.get("activities", ()))
    if not component_activities:
        return 0

    if precision_bundle is not None:
        baseline_enabled_mass = precision_bundle.get("baseline_enabled_mass", 0.0)
        if baseline_enabled_mass <= 0:
            return 0

        affected_labels = frozenset(
            str(activity)
            for activity in component_activities
            if activity is not None
        )
        reduced_enabled_by_context = _virtual_reduction_enabled_labels_by_context(
            precision_bundle,
            affected_labels,
        )
        unsupported_mass = 0.0

        for ctx, weight in precision_bundle.get("context_weights", {}).items():
            original_enabled = precision_bundle["original_enabled_by_context"].get(
                ctx,
                frozenset(),
            )
            reduced_enabled = reduced_enabled_by_context.get(ctx, frozenset())
            new_labels = reduced_enabled - original_enabled
            if not new_labels:
                continue

            unsupported_labels = new_labels - precision_bundle["reduced_log_enabled_by_context"].get(
                ctx,
                frozenset(),
            )
            unsupported_mass += weight * len(unsupported_labels)

        return unsupported_mass / baseline_enabled_mass

    if current_model is None or current_ocel is None:
        return 1

    prepared_context = _prepare_pruning_candidate_context(
        current_model,
        current_ocel,
        lower_layer_ocel,
        component_activities,
        build_component_and_edge_ocel=True,
    )
    return _compute_information_loss_from_prepared_context(prepared_context)


def _compute_simplicity_gain_from_prepared_context(prepared_context, component):
    if prepared_context is None:
        return 0

    from .net_quality import NetQuality

    component_activities = prepared_context["component_activities"]
    component_and_edge_ocpn = prepared_context["component_and_edge_ocpn"]
    edge_only_ocpn = prepared_context["edge_only_ocpn"]

    if component_and_edge_ocpn is None:
        return 0

    complexity_with_component_ocpn = component_and_edge_ocpn
    subprocess_activity_groups = _candidate_subprocess_activity_groups(component)
    if subprocess_activity_groups:
        matching_components = _match_subprocess_components_by_activity_groups(
            component_and_edge_ocpn,
            subprocess_activity_groups,
        )
        if matching_components:
            complexity_with_component_ocpn, _ = collapse_sub_processes(
                component_and_edge_ocpn,
                list(matching_components),
            )

    complexity_with_component = float(NetQuality(complexity_with_component_ocpn).complexity())
    complexity_without_component = 0.0
    if edge_only_ocpn is not None:
        complexity_without_component = float(NetQuality(edge_only_ocpn).complexity())

    simplicity_gain = 1 - (complexity_without_component / complexity_with_component)
    return simplicity_gain


def _is_variable_arc(ocpn, object_type, arc):
    if bool(getattr(arc, "variable", False)) or bool(getattr(arc, "properties", {}).get("variable", False)):
        return True

    transition = None
    if isinstance(arc.source, PetriNet.Transition):
        transition = arc.source
    elif isinstance(arc.target, PetriNet.Transition):
        transition = arc.target

    if transition is None or transition.label is None:
        return False

    return bool(
        ocpn.get("double_arcs_on_activity", {})
        .get(object_type, {})
        .get(str(transition.label), False)
    )


def _simplicity_estimator_score(ocpn):
    if ocpn is None:
        return 0.0

    visible_activities = set()
    place_count = 0
    silent_transition_count = 0
    arc_count = 0
    variable_arc_count = 0

    for object_type, (net, _, _) in ocpn["petri_nets"].items():
        place_count += len(net.places)

        for transition in net.transitions:
            if transition.label is None:
                silent_transition_count += 1
            else:
                visible_activities.add(str(transition.label))

        for arc in net.arcs:
            arc_count += 1
            if _is_variable_arc(ocpn, object_type, arc):
                variable_arc_count += 1

    return float(
        len(visible_activities)
        + place_count
        + 2 * silent_transition_count
        + arc_count
        + variable_arc_count
    )


def _copy_double_arc_metadata(source_ocpn, target_ocpn):
    target_ocpn["double_arcs_on_activity"] = {
        object_type: dict(
            source_ocpn.get("double_arcs_on_activity", {}).get(object_type, {})
        )
        for object_type in target_ocpn.get("petri_nets", {})
    }


def _filter_marking_to_net(marking, net):
    filtered_marking = Marking()

    for place, count in marking.items():
        if place in net.places and count:
            filtered_marking[place] = count

    return filtered_marking


def _refresh_ocpn_activity_metadata(ocpn):
    if ocpn is None:
        return

    ocpn["activities"] = sorted({
        str(transition.label)
        for net, _, _ in ocpn.get("petri_nets", {}).values()
        for transition in net.transitions
        if transition.label is not None
    })

    double_arcs_on_activity = ocpn.setdefault("double_arcs_on_activity", {})
    for object_type, (net, _, _) in ocpn.get("petri_nets", {}).items():
        remaining_labels = {
            str(transition.label)
            for transition in net.transitions
            if transition.label is not None
        }
        double_arcs_on_activity[object_type] = {
            str(label): value
            for label, value in double_arcs_on_activity.get(object_type, {}).items()
            if str(label) in remaining_labels
        }


def _project_ocpn_to_activities(ocpn, allowed_activities):
    if ocpn is None:
        return None

    allowed_labels = {
        str(activity)
        for activity in allowed_activities
        if activity is not None
    }
    projected_ocpn, _ = _clone_ocpn(ocpn)
    if projected_ocpn is None:
        return None

    _copy_double_arc_metadata(ocpn, projected_ocpn)

    for object_type, (net, _, _) in projected_ocpn["petri_nets"].items():
        transitions_to_remove = [
            transition
            for transition in sorted(net.transitions, key=_sort_petri_net_node)
            if transition.label is not None
            and str(transition.label) not in allowed_labels
        ]
        for transition in transitions_to_remove:
            if transition in net.transitions:
                petri_utils.remove_transition(net, transition)

    boundary_places = _boundary_places_by_type(projected_ocpn)
    while True:
        removed_orphan_places, _ = _remove_orphan_places_from_working_ocpn(
            projected_ocpn,
            boundary_places,
        )
        removed_silent_transitions, _ = _remove_useless_silent_transitions_from_working_ocpn(
            projected_ocpn
        )
        if not removed_orphan_places and not removed_silent_transitions:
            break

    for object_type, (net, initial_marking, final_marking) in projected_ocpn["petri_nets"].items():
        projected_ocpn["petri_nets"][object_type] = (
            net,
            _filter_marking_to_net(initial_marking, net),
            _filter_marking_to_net(final_marking, net),
        )

    _refresh_ocpn_activity_metadata(projected_ocpn)
    return projected_ocpn


def _build_simplicity_gain_working_ocpn(current_model, component):
    if current_model is None:
        return None, frozenset()

    subprocess_components = tuple(component.get("subprocess_components", ()))
    if not subprocess_components:
        subprocess_activity_groups = _candidate_subprocess_activity_groups(component)
        if subprocess_activity_groups:
            subprocess_components = _match_subprocess_components_by_activity_groups(
                current_model,
                subprocess_activity_groups,
            )

    if subprocess_components:
        working_ocpn, inserted_transitions = collapse_sub_processes(
            current_model,
            list(subprocess_components),
        )
    else:
        working_ocpn, _ = _clone_ocpn(current_model)
        inserted_transitions = frozenset()

    if working_ocpn is None:
        return None, frozenset()

    _copy_double_arc_metadata(current_model, working_ocpn)
    return working_ocpn, inserted_transitions


def _remove_arc_and_track_score(net, arc, object_type, ocpn):
    removed_score = 1
    if _is_variable_arc(ocpn, object_type, arc):
        removed_score += 1
    petri_utils.remove_arc(net, arc)
    return removed_score


def _remove_place_and_track_score(net, place, object_type, ocpn):
    removed_score = 1
    for arc in tuple(place.in_arcs) + tuple(place.out_arcs):
        if arc in net.arcs:
            removed_score += _remove_arc_and_track_score(net, arc, object_type, ocpn)
    petri_utils.remove_place(net, place)
    return removed_score


def _remove_transition_and_track_score(net, transition, object_type, ocpn, removed_visible_labels):
    removed_score = 0
    for arc in tuple(transition.in_arcs) + tuple(transition.out_arcs):
        if arc in net.arcs:
            removed_score += _remove_arc_and_track_score(net, arc, object_type, ocpn)

    if transition.label is None:
        removed_score += 2
    else:
        visible_label = str(transition.label)
        if visible_label not in removed_visible_labels:
            removed_visible_labels.add(visible_label)
            removed_score += 1

    petri_utils.remove_transition(net, transition)
    return removed_score


def _boundary_places_by_type(ocpn):
    boundary_places = defaultdict(set)

    for object_type, (net, initial_marking, final_marking) in ocpn["petri_nets"].items():
        for place, count in initial_marking.items():
            if count > 0 and place in net.places:
                boundary_places[object_type].add(place)
        for place, count in final_marking.items():
            if count > 0 and place in net.places:
                boundary_places[object_type].add(place)

    return {
        object_type: frozenset(places)
        for object_type, places in boundary_places.items()
    }


def _prune_candidate_activities_from_working_ocpn(working_ocpn, component, inserted_transitions):
    collapsed_subprocess_activities = {
        activity
        for group in _candidate_subprocess_activity_groups(component)
        for activity in group
    }
    visible_labels_to_remove = {
        str(activity)
        for activity in component.get("activities", ())
        if activity not in collapsed_subprocess_activities
    }
    inserted_transition_keys = {
        (object_type, transition)
        for object_type, transition in inserted_transitions
    }

    removed_score = 0
    removed_visible_labels = set()

    for object_type, (net, _, _) in working_ocpn["petri_nets"].items():
        transitions_to_remove = [
            transition
            for transition in sorted(net.transitions, key=_sort_petri_net_node)
            if (transition.label is not None and str(transition.label) in visible_labels_to_remove)
            or (object_type, transition) in inserted_transition_keys
        ]
        for transition in transitions_to_remove:
            if transition not in net.transitions:
                continue
            removed_score += _remove_transition_and_track_score(
                net,
                transition,
                object_type,
                working_ocpn,
                removed_visible_labels,
            )

    return removed_score


def _remove_orphan_places_from_working_ocpn(working_ocpn, boundary_places):
    removed_score = 0
    removed_any = False

    for object_type, (net, _, _) in working_ocpn["petri_nets"].items():
        protected_places = boundary_places.get(object_type, frozenset())
        orphan_places = [
            place
            for place in sorted(net.places, key=_sort_petri_net_node)
            if place not in protected_places
            and (not place.in_arcs or not place.out_arcs)
        ]
        for place in orphan_places:
            if place not in net.places:
                continue
            removed_score += _remove_place_and_track_score(
                net,
                place,
                object_type,
                working_ocpn,
            )
            removed_any = True

    return removed_any, removed_score


def _remove_useless_silent_transitions_from_working_ocpn(working_ocpn):
    removed_score = 0
    removed_any = False
    removed_visible_labels = set()

    for object_type, (net, _, _) in working_ocpn["petri_nets"].items():
        silent_transitions = [
            transition
            for transition in sorted(net.transitions, key=_sort_petri_net_node)
            if transition.label is None
            and (not transition.in_arcs or not transition.out_arcs)
        ]
        for transition in silent_transitions:
            if transition not in net.transitions:
                continue
            removed_score += _remove_transition_and_track_score(
                net,
                transition,
                object_type,
                working_ocpn,
                removed_visible_labels,
            )
            removed_any = True

    return removed_any, removed_score


def _compute_simplicity_gain(current_model, current_ocel, lower_layer_ocel, component, baseline_score=None):
    del current_ocel, lower_layer_ocel

    component_activities = tuple(component.get("activities", ()))
    if current_model is None or not component_activities:
        return 0

    if baseline_score is None:
        baseline_score = _simplicity_estimator_score(current_model)
    if baseline_score <= 0:
        return 0

    working_ocpn, inserted_transitions = _build_simplicity_gain_working_ocpn(
        current_model,
        component,
    )
    if working_ocpn is None:
        return 0

    removed_score = _prune_candidate_activities_from_working_ocpn(
        working_ocpn,
        component,
        inserted_transitions,
    )

    boundary_places = _boundary_places_by_type(working_ocpn)
    while True:
        removed_orphan_places, orphan_place_score = _remove_orphan_places_from_working_ocpn(
            working_ocpn,
            boundary_places,
        )
        removed_score += orphan_place_score

        removed_silent_transitions, silent_transition_score = (
            _remove_useless_silent_transitions_from_working_ocpn(working_ocpn)
        )
        removed_score += silent_transition_score

        if not removed_orphan_places and not removed_silent_transitions:
            break

    return removed_score / baseline_score


def _build_collapsed_subprocess_model(component_and_edge_ocpn, subprocess_activity_groups):
    if component_and_edge_ocpn is None or not subprocess_activity_groups:
        return None

    matching_components = _match_subprocess_components_by_activity_groups(
        component_and_edge_ocpn,
        subprocess_activity_groups,
    )
    if not matching_components:
        return component_and_edge_ocpn

    collapsed_ocpn, _ = collapse_sub_processes(
        component_and_edge_ocpn,
        list(matching_components),
    )
    return collapsed_ocpn


def _debug_pruning_candidate(current_model, current_ocel, lower_layer_ocel, candidate, simplicity_gain, information_loss):
    component_activities = tuple(candidate.get("activities", ()))
    if current_model is None or current_ocel is None or not component_activities:
        return

    from .net_quality import NetQuality
    from ..helpers.vorbose import render_pruning_candidate_debug

    prepared_context = _prepare_pruning_candidate_context(
        current_model,
        current_ocel,
        lower_layer_ocel,
        component_activities,
        build_component_and_edge_ocel=True,
    )
    component_and_edge_ocpn = prepared_context["component_and_edge_ocpn"]
    edge_only_ocpn = prepared_context["edge_only_ocpn"]
    component_and_edge_ocel = prepared_context["component_and_edge_ocel"]

    if component_and_edge_ocpn is None and edge_only_ocpn is None:
        return

    debug_with_ocpn = component_and_edge_ocpn
    subprocess_activity_groups = _candidate_subprocess_activity_groups(candidate)
    if subprocess_activity_groups:
        debug_with_ocpn = _build_collapsed_subprocess_model(
            component_and_edge_ocpn,
            subprocess_activity_groups,
        )

    complexity_with_component = 0.0
    if debug_with_ocpn is not None:
        complexity_with_component = float(NetQuality(debug_with_ocpn).complexity())

    complexity_without_component = 0.0
    if edge_only_ocpn is not None:
        complexity_without_component = float(NetQuality(edge_only_ocpn).complexity())

    precision_with_component = 0.0
    if component_and_edge_ocpn is not None and component_and_edge_ocel is not None:
        precision_with_component = float(
            NetQuality(
                component_and_edge_ocpn,
                component_and_edge_ocel,
                max_nodes_per_replay=100,
            ).precision()
        )

    precision_without_component = 0.0
    if edge_only_ocpn is not None and component_and_edge_ocel is not None:
        precision_without_component = float(
            NetQuality(
                edge_only_ocpn,
                component_and_edge_ocel,
                max_nodes_per_replay=10,
            ).precision()
        )

    candidate_label = candidate.get("id") or ",".join(component_activities)
    safe_candidate_label = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(candidate_label)
    ).strip("_") or "candidate"
    #output_path = Path(tempfile.gettempdir()) / f"pruning_candidate_debug_{safe_candidate_label}.png"

    # render_pruning_candidate_debug(
    #     debug_with_ocpn,
    #     edge_only_ocpn,
    #     title=f"Pruning Candidate: {candidate_label}",
    #     with_component_title="With component",
    #     without_component_title="Without component",
    #     with_component_metrics={
    #         "Complexity": complexity_with_component,
    #         "Precision": precision_with_component,
    #     },
    #     without_component_metrics={
    #         "Complexity": complexity_without_component,
    #         "Precision": precision_without_component,
    #     },
    #     summary_metrics={
    #         "Simplicity gain": simplicity_gain,
    #         "Precision loss": information_loss,
    #         "Score": information_loss - simplicity_gain,
    #     },
    #     output_path=output_path,
    #     show=True,
    # )

    #print(f"Pruning candidate debug image: {output_path}")
def _select_best_pruning_candidate(
    current_model,
    current_ocel,
    lower_layer_ocel,
    candidates,
    *,
    precision_bundle=None,
    show_progress=False,
    progress_desc="Evaluating pruning candidates",
):
    candidates = list(candidates)
    if not candidates:
        return None, 0.0

    best_candidate = None
    best_score = 0.0
    frontier = [tuple([index]) for index in range(len(candidates))]
    simplicity_baseline_score = _simplicity_estimator_score(current_model)

    round_number = 1
    while frontier:
        frontier_iter = frontier
        if show_progress:
            frontier_iter = tqdm(
                frontier,
                total=len(frontier),
                desc=f"{progress_desc} (round {round_number})",
                leave=False,
                file=sys.stdout,
            )

        positive_results = []
        extension_indices = set()

        for candidate_indices in frontier_iter:
            merged_candidate = _merge_pruning_candidates(
                candidates[index]
                for index in candidate_indices
            )
            simplicity_gain = _compute_simplicity_gain(
                current_model,
                current_ocel,
                lower_layer_ocel,
                merged_candidate,
                simplicity_baseline_score,
            )
            information_loss = _compute_information_loss(
                current_model,
                current_ocel,
                lower_layer_ocel,
                merged_candidate,
                precision_bundle,
            )
            # _debug_pruning_candidate(
            #     current_model,
            #     current_ocel,
            #     lower_layer_ocel,
            #     merged_candidate,
            #     simplicity_gain,
            #     information_loss,
            # )
            score = simplicity_gain - information_loss

            if show_progress:
                candidate_label = merged_candidate.get("id") or ",".join(
                    merged_candidate.get("activities", ())
                )
                frontier_iter.set_postfix_str(
                    f"best={best_score:.3f} current={score:.3f} candidate={candidate_label}",
                    refresh=False,
                )

            if score < 0:
                continue

            positive_results.append((candidate_indices, merged_candidate, score))
            extension_indices.update(candidate_indices)

        if not positive_results:
            break

        best_candidate_indices, best_candidate, best_score = max(
            positive_results,
            key=lambda result: result[2],
        )
        extension_indices.difference_update(best_candidate_indices)
        if not extension_indices:
            break

        frontier = [
            tuple(sorted(best_candidate_indices + (candidate_index,)))
            for candidate_index in sorted(extension_indices)
            if candidate_index not in best_candidate_indices
        ]
        round_number += 1

    return best_candidate, best_score


def discover_models_for_hierarchy(
    ocel,
    solution,
    layer_context=None,
    show_progress=False,
    *,
    precision_context_sample_size=512,
    precision_context_depth=5,
    precision_context_sample_seed=None,
):
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
    layer_iter = discovered_layers
    layer_progress_bar = None
    if show_progress:
        layer_progress_bar = tqdm(
            discovered_layers,
            total=len(discovered_layers),
            desc="Discovering models",
            file=sys.stdout,
        )
        layer_iter = layer_progress_bar

    for layer_index, layer in enumerate(layer_iter):
        selected_object_types = layer_to_object_types[layer]
        subprocess_components = []
        native_layer_activities = {
            activity
            for activity, activity_layer in activity_to_layer.items()
            if activity_layer == layer
        }
        active_activities = {
            activity
            for activity, activity_layer in activity_to_layer.items()
            if activity_layer <= layer and layer - activity_layer <= layer_context_by_layer[layer]
        }
        progress_prefix = f"Layer {layer}"

        with _progress_step(
            f"{progress_prefix}: building temporary layer log",
            enabled=show_progress,
            progress_bar=layer_progress_bar,
        ):
            layer_ocel, included_event_records, included_activities = _build_layer_ocel(
                ocel,
                event_records,
                object_to_type,
                selected_object_types,
                active_activities,
            )

        ocpn, subprocess_components = _discover_ocpn_with_subprocess_components(
            layer_ocel,
            activity_to_layer,
            layer,
            show_progress,
            progress_prefix,
            layer_progress_bar,
        )

        with _progress_step(
            f"{progress_prefix}: building pruning candidates",
            enabled=show_progress,
            progress_bar=layer_progress_bar,
        ):
            candidates = _build_pruning_candidates(
                subprocess_components,
                included_activities,
                activity_to_layer,
                layer,
            )

        precision_reference = None
        if layer_index > 0 and ocpn is not None:
            previous_layer = discovered_layers[layer_index - 1]
            previous_layer_model = discovered_models.get(previous_layer)
            previous_layer_ocpn = None
            if previous_layer_model is not None:
                previous_layer_ocpn = previous_layer_model.get("ocpn")

            if previous_layer_ocpn is not None:
                with _progress_step(
                    f"{progress_prefix}: precomputing precision reference",
                    enabled=show_progress,
                    progress_bar=layer_progress_bar,
                ):
                    precision_reference = _build_precision_reference_bundle(
                        ocel,
                        event_records,
                        object_to_type,
                        activity_to_layer,
                        layer,
                        selected_object_types,
                        ocpn,
                        previous_layer_ocpn,
                        precision_context_sample_size=precision_context_sample_size,
                        precision_context_depth=precision_context_depth,
                        precision_context_sample_seed=precision_context_sample_seed,
                        show_progress=show_progress,
                    )

        selected_candidate = None
        if candidates:
            with _progress_step(
                f"{progress_prefix}: greedy pruning search",
                enabled=show_progress,
                progress_bar=layer_progress_bar,
            ):
                selected_candidate, _ = _select_best_pruning_candidate(
                    current_model=ocpn,
                    current_ocel=layer_ocel,
                    lower_layer_ocel=None,
                    candidates=candidates,
                    precision_bundle=precision_reference,
                    show_progress=show_progress,
                    progress_desc=f"{progress_prefix}: evaluating candidate sets",
                )

        retained_lower_layer_activities = set()
        if selected_candidate is not None:
            retained_lower_layer_activities.update(selected_candidate.get("activities", ()))

        final_active_activities = native_layer_activities | retained_lower_layer_activities
        if final_active_activities != active_activities:
            with _progress_step(
                f"{progress_prefix}: rebuilding final layer log",
                enabled=show_progress,
                progress_bar=layer_progress_bar,
            ):
                layer_ocel, included_event_records, included_activities = _build_layer_ocel(
                    ocel,
                    event_records,
                    object_to_type,
                    selected_object_types,
                    final_active_activities,
                )

            ocpn, subprocess_components = _discover_ocpn_with_subprocess_components(
                layer_ocel,
                activity_to_layer,
                layer,
                show_progress,
                f"{progress_prefix} final",
                layer_progress_bar,
            )

        with _progress_step(
            f"Layer {layer}: deriving activity resources",
            enabled=show_progress,
            progress_bar=layer_progress_bar,
        ):
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
            "precision_reference": precision_reference,
        }

        if layer_progress_bar is not None:
            layer_progress_bar.set_postfix_str(f"Layer {layer}: completed", refresh=False)

    discovered_models = _assign_subprocess_indices_across_layers(discovered_models)
    return dict(activity_to_layer), discovered_models
