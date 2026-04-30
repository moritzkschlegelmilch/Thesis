from collections import defaultdict
from contextlib import contextmanager
from numbers import Integral
import sys

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import PetriNet
from tqdm import tqdm

from .totem import _prepare_totem_data, get_all_event_objects
from .subprocess_detection import _component_boundary_places, collapse_sub_processes, detect_subprocess_components

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


def _discover_ocpn(layer_ocel):
    if layer_ocel.events.empty or layer_ocel.relations.empty:
        return None
    return pm4py.discover_oc_petri_net(layer_ocel)


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
    return len(component.get("transition_keys", ()))


def _component_place_count(component):
    if component is None:
        return 0
    if hasattr(component, "get"):
        return len(component.get("place_keys", ()))
    return len(component.get("place_keys", ()))


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


def _compute_information_loss(current_model, current_ocel, lower_layer_ocel, component):
    component_activities = tuple(component.get("activities", ()))
    if current_model is None or current_ocel is None or not component_activities:
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


def _compute_simplicity_gain(current_model, current_ocel, lower_layer_ocel, component):
    component_activities = tuple(component.get("activities", ()))
    if current_model is None or current_ocel is None or not component_activities:
        return 0

    prepared_context = _prepare_pruning_candidate_context(
        current_model,
        current_ocel,
        lower_layer_ocel,
        component_activities,
        build_component_and_edge_ocel=False,
    )

    return _compute_simplicity_gain_from_prepared_context(prepared_context, component)


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
    #         "Score": simplicity_gain - information_loss,
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
    show_progress=False,
    progress_desc="Evaluating pruning candidates",
):
    best_candidate = None
    best_score = float("-inf")

    candidate_iter = candidates
    if show_progress:
        candidate_iter = tqdm(
            candidates,
            total=len(candidates),
            desc=progress_desc,
            leave=False,
            file=sys.stdout,
        )

    for candidate in candidate_iter:
        component_activities = tuple(candidate.get("activities", ()))
        prepared_context = None
        if current_model is not None and current_ocel is not None and component_activities:
            prepared_context = _prepare_pruning_candidate_context(
                current_model,
                current_ocel,
                lower_layer_ocel,
                component_activities,
                build_component_and_edge_ocel=True,
            )
        simplicity_gain = _compute_simplicity_gain_from_prepared_context(prepared_context, candidate)
        information_loss = _compute_information_loss_from_prepared_context(prepared_context)
        score = simplicity_gain - information_loss

        if show_progress:
            candidate_label = candidate.get("id") or ",".join(component_activities)
            candidate_iter.set_postfix_str(
                f"best={best_score:.3f} current={score:.3f} candidate={candidate_label}",
                refresh=False,
            )

        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate, best_score


def discover_models_for_hierarchy(ocel, solution, layer_context=None, show_progress=False):
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

        i = 0
        while True:
            i += 1
            progress_prefix = f"Layer {layer} iteration {i}"

            with _progress_step(
                f"{progress_prefix}: building layer log",
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

            ocpn, iteration_components = _discover_ocpn_with_subprocess_components(
                layer_ocel,
                activity_to_layer,
                layer,
                show_progress,
                progress_prefix,
                layer_progress_bar,
            )
            subprocess_components = iteration_components

            with _progress_step(
                f"{progress_prefix}: building pruning candidates",
                enabled=show_progress,
                progress_bar=layer_progress_bar,
            ):
                candidates = _build_pruning_candidates(
                    iteration_components,
                    included_activities,
                    activity_to_layer,
                    layer,
                )

            # Temporary correctness mode:
            # keep only native activities for this layer plus lower-layer
            # activities that belong to detected subprocess candidates.
            # Candidate activities already include any standalone
            # boundary-adjacent activities attached in
            # _build_pruning_candidates().
            with _progress_step(
                f"{progress_prefix}: retaining subprocess activities",
                enabled=show_progress,
                progress_bar=layer_progress_bar,
            ):
                retained_subprocess_activities = {
                    activity
                    for candidate in candidates
                    if candidate.get("kind") == "subprocess"
                    for activity in candidate.get("activities", ())
                }
            next_active_activities = native_layer_activities | retained_subprocess_activities
            if next_active_activities == active_activities:
                break
            active_activities = next_active_activities

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
        }

        if layer_progress_bar is not None:
            layer_progress_bar.set_postfix_str(f"Layer {layer}: completed", refresh=False)

    return dict(activity_to_layer), discovered_models
