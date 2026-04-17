from collections import defaultdict, deque

from .discovery_preparation import (
    _build_ocel_filtering_context,
    _build_ocel_from_filtering_context,
    _discover_ocpn,
    _filter_ocel_filtering_context,
    _merge_ocel_filtering_contexts,
)
from .subprocess_detection import _arc_key, _place_key, _transition_key


def _sort_petri_net_node(node):
    return getattr(node, "name", ""), id(node)


def _sorted_successor_places(transition):
    neighbors = [
        (arc.target, arc)
        for arc in transition.out_arcs
    ]
    neighbors.sort(key=lambda item: _sort_petri_net_node(item[0]))
    return tuple(neighbors)


def _sorted_successor_transitions(place):
    neighbors = [
        (arc.target, arc)
        for arc in place.out_arcs
    ]
    neighbors.sort(key=lambda item: _sort_petri_net_node(item[0]))
    return tuple(neighbors)


def _normalized_component_labels(component_activity_labels):
    return tuple(sorted({
        label
        for label in component_activity_labels
        if label is not None
    }))


def _component_net_contains_activity(component_net, component_labels):
    return any(
        transition.label in component_labels
        for transition in component_net.transitions
    )


def calculate_component_deletion_impact_footprint(component_net, component_activity_labels):
    component_labels = frozenset(_normalized_component_labels(component_activity_labels))
    if not component_labels:
        return {
            "impact_labels": set(),
            "transition_labels": set(),
            "transitions": set(),
            "places": set(),
            "arcs": set(),
        }

    transitions_by_label = defaultdict(list)
    for transition in component_net.transitions:
        if transition.label is not None:
            transitions_by_label[transition.label].append(transition)

    for transitions in transitions_by_label.values():
        transitions.sort(key=_sort_petri_net_node)

    explored_component_labels = set()
    impact_labels = set()
    visited_transition_labels = set()
    visited_transitions = set()
    visited_places = set()
    visited_arcs = set()
    sorted_component_labels = tuple(sorted(component_labels))

    while True:
        start_label = next(
            (
                label
                for label in sorted_component_labels
                if label not in explored_component_labels
            ),
            None,
        )
        if start_label is None:
            break

        explored_component_labels.add(start_label)
        queue = deque(
            ("transition", transition)
            for transition in transitions_by_label.get(start_label, ())
        )
        bfs_visited_transitions = set()
        bfs_visited_places = set()

        while queue:
            node_type, node = queue.popleft()
            if node_type == "transition":
                if node in bfs_visited_transitions:
                    continue

                bfs_visited_transitions.add(node)
                visited_transitions.add(node)
                transition_label = node.label
                if transition_label is not None:
                    visited_transition_labels.add(transition_label)

                if transition_label is None:
                    should_expand = True
                elif transition_label in component_labels:
                    explored_component_labels.add(transition_label)
                    should_expand = True
                else:
                    impact_labels.add(transition_label)
                    should_expand = False

                if not should_expand:
                    continue

                for place, arc in _sorted_successor_places(node):
                    visited_arcs.add(arc)
                    queue.append(("place", place))
            else:
                if node in bfs_visited_places:
                    continue

                bfs_visited_places.add(node)
                visited_places.add(node)
                for transition, arc in _sorted_successor_transitions(node):
                    visited_arcs.add(arc)
                    queue.append(("transition", transition))

    return {
        "impact_labels": impact_labels,
        "transition_labels": visited_transition_labels,
        "transitions": visited_transitions,
        "places": visited_places,
        "arcs": visited_arcs,
    }


def calculate_component_deletion_impact(component_net, component_activity_labels):
    footprint = calculate_component_deletion_impact_footprint(
        component_net,
        component_activity_labels,
    )
    return footprint["impact_labels"]


def calculate_ocpn_component_deletion_impact(ocpn, component_activity_labels):
    component_labels = frozenset(_normalized_component_labels(component_activity_labels))
    if ocpn is None or not component_labels:
        return set()

    impact_labels = set()
    for _, (net, _, _) in ocpn["petri_nets"].items():
        if not _component_net_contains_activity(net, component_labels):
            continue
        impact_labels.update(calculate_component_deletion_impact(net, component_labels))

    return impact_labels


def _build_component_and_edge_ocels(ocel, ocpn, component_activity_labels, lower_layer_ocel=None):
    component_labels = frozenset(_normalized_component_labels(component_activity_labels))
    if not component_labels:
        return None, None

    edge_labels = calculate_ocpn_component_deletion_impact(ocpn, component_labels)
    relevant_activities = component_labels | edge_labels

    upper_context = _build_ocel_filtering_context(ocel)
    upper_component_and_edge_context = _filter_ocel_filtering_context(
        upper_context,
        relevant_activities,
    )
    lower_component_and_edge_context = {}
    if lower_layer_ocel is not None:
        lower_context = _build_ocel_filtering_context(lower_layer_ocel)
        lower_component_and_edge_context = _filter_ocel_filtering_context(
            lower_context,
            relevant_activities,
        )

    component_and_edge_context = _merge_ocel_filtering_contexts(
        upper_component_and_edge_context,
        lower_component_and_edge_context,
    )
    component_and_edge_ocel = _build_ocel_from_filtering_context(component_and_edge_context)

    upper_edge_only_context = _filter_ocel_filtering_context(
        upper_context,
        edge_labels,
    )
    right_union_context = _merge_ocel_filtering_contexts(
        upper_edge_only_context,
        lower_component_and_edge_context,
    )
    edge_only_ocel = _build_ocel_from_filtering_context(right_union_context)
    return component_and_edge_ocel, edge_only_ocel


def discover_component_and_edge_ocpns(ocel, ocpn, component_activity_labels, lower_layer_ocel=None):
    component_and_edge_ocel, edge_only_ocel = _build_component_and_edge_ocels(
        ocel,
        ocpn,
        component_activity_labels,
        lower_layer_ocel=lower_layer_ocel,
    )
    component_and_edge_ocpn = _discover_ocpn(component_and_edge_ocel)
    edge_only_ocpn = _discover_ocpn(edge_only_ocel)
    return component_and_edge_ocpn, edge_only_ocpn


def build_component_deletion_highlight(ocpn, component_activity_labels, highlight_color="#f7d7a6"):
    transition_keys = set()
    place_keys = set()
    arc_keys = set()
    component_labels = frozenset(_normalized_component_labels(component_activity_labels))

    for object_type, (net, _, _) in ocpn["petri_nets"].items():
        if not _component_net_contains_activity(net, component_labels):
            continue

        footprint = calculate_component_deletion_impact_footprint(
            net,
            component_labels,
        )

        transition_keys.update(
            _transition_key(object_type, transition)
            for transition in footprint["transitions"]
        )
        place_keys.update(
            _place_key(object_type, place)
            for place in footprint["places"]
        )
        arc_keys.update(
            _arc_key(object_type, arc)
            for arc in footprint["arcs"]
        )

    return {
        "id": "component_deletion_impact",
        "color": highlight_color,
        "fillcolor": highlight_color,
        "transition_keys": frozenset(transition_keys),
        "place_keys": frozenset(place_keys),
        "arc_keys": frozenset(arc_keys),
    }


def render_component_deletion_impact(
        ocpn,
        component_activity_labels,
        highlight_color="#f7d7a6",
        max_size=(1400, 700),
):
    if ocpn is None:
        return None

    from ..helpers.vorbose import _render_ocpn_image

    highlight_component = build_component_deletion_highlight(
        ocpn,
        component_activity_labels,
        highlight_color=highlight_color,
    )
    return _render_ocpn_image(
        ocpn,
        max_size=max_size,
        subprocess_components=[highlight_component],
    )
