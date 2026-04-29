from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any

from pm4py.objects.petri_net.obj import PetriNet
from .subprocess_detection import (
    _arc_key,
    _describe_place_key,
    _describe_transition_key,
    _place_key,
    _sort_place_key,
    _sort_transition_key,
    _transition_key,
)


@dataclass(frozen=True)
class LocalRegion:
    object_type: str
    source: PetriNet.Place
    internal: frozenset[Any]
    target: PetriNet.Place
    vertices: frozenset[Any]
    activities: frozenset[str]


@dataclass(frozen=True)
class ObjectCentricRegion:
    source: frozenset[tuple[str, PetriNet.Place]]
    internal: frozenset[tuple[str, Any]]
    target: frozenset[tuple[str, PetriNet.Place]]
    local_regions: frozenset[LocalRegion]
    activities: frozenset[str]
    id: str | None = None
    color: str | None = field(default=None, compare=False, hash=False)
    fillcolor: str | None = field(default=None, compare=False, hash=False)
    marker: str | None = field(default=None, compare=False, hash=False)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        mapping = {
            "id": self.id,
            "color": self.color,
            "fillcolor": self.fillcolor,
            "marker": self.marker,
            "object_types": self.object_types,
            "transition_keys": self.transition_keys,
            "place_keys": self.place_keys,
            "arc_keys": self.arc_keys,
            "transitions": self.transitions,
            "places": self.places,
            "arcs": self.arcs,
            "activities": tuple(sorted(self.activities)),
        }
        if key not in mapping:
            raise KeyError(key)
        return mapping[key]

    def __setitem__(self, key, value):
        if key in {"id", "color", "fillcolor", "marker"}:
            object.__setattr__(self, key, value)
            return
        raise KeyError(key)

    @property
    def object_types(self):
        return sorted({
            local_region.object_type
            for local_region in self.local_regions
        })

    @property
    def transition_keys(self):
        return frozenset(_region_transition_keys(self))

    @property
    def place_keys(self):
        return frozenset(_region_place_keys(self))

    @property
    def arc_keys(self):
        return frozenset(_region_arc_keys(self))

    @property
    def transitions(self):
        return [
            _describe_transition_key(transition_key)
            for transition_key in sorted(self.transition_keys, key=_sort_transition_key)
        ]

    @property
    def places(self):
        return [
            _describe_place_key(place_key)
            for place_key in sorted(self.place_keys, key=_sort_place_key)
        ]

    @property
    def arcs(self):
        return [
            _describe_region_arc(arc_key)
            for arc_key in sorted(
                self.arc_keys,
                key=lambda arc_key: (
                    arc_key[1],
                    _sort_petri_net_node(arc_key[2].source),
                    _sort_petri_net_node(arc_key[2].target),
                ),
            )
        ]


@dataclass(frozen=True)
class _ObjectTypeContext:
    object_type: str
    net: PetriNet
    places: tuple[PetriNet.Place, ...]
    transitions: tuple[PetriNet.Transition, ...]
    vertices: frozenset[Any]
    out_neighbors: dict[Any, frozenset[Any]]
    in_neighbors: dict[Any, frozenset[Any]]
    undirected_neighbors: dict[Any, frozenset[Any]]
    visible_labels: frozenset[str]


@dataclass(frozen=True)
class _ObligationGraph:
    adjacency: dict[Any, frozenset[Any]]
    obligation_nodes: frozenset[tuple[str, str]]


def detect_object_centric_regions(ocpn, allowed_activities=None):
    if ocpn is None:
        return []

    petri_nets = getattr(ocpn, "get", lambda *_: None)("petri_nets")
    if not isinstance(petri_nets, dict):
        raise TypeError("ocpn must expose a 'petri_nets' dictionary like pm4py OCPNs.")

    contexts = {
        object_type: _build_object_type_context(object_type, net)
        for object_type, (net, _, _) in petri_nets.items()
    }
    allowed_activities = _normalize_allowed_activities(contexts, allowed_activities)
    activity_to_object_types = _build_activity_to_object_types(contexts)

    local_regions = set()
    for object_type, context in contexts.items():
        blocked_activities = context.visible_labels - allowed_activities
        local_regions.update(
            _maximal_regions_avoiding_labels(context, blocked_activities)
        )

    output_regions = _resolve_region_set(
        frozenset(local_regions),
        frozenset(),
        contexts,
        activity_to_object_types,
    )
    output_regions = _remove_duplicate_and_contained_outputs(output_regions)
    output_regions = _remove_trivial_regions(output_regions)
    output_regions = sorted(output_regions, key=_sort_output_region_key)
    return [
        replace(region, id=f"subprocess_{index}")
        for index, region in enumerate(output_regions, start=1)
    ]


def serialize_object_centric_region(region: ObjectCentricRegion):
    return {
        "source": [
            _describe_typed_place(object_type, place)
            for object_type, place in sorted(
                region.source,
                key=lambda item: (item[0], _sort_petri_net_node(item[1])),
            )
        ],
        "internal": [
            _describe_typed_vertex(object_type, vertex)
            for object_type, vertex in sorted(
                region.internal,
                key=lambda item: (item[0], _sort_vertex(item[1])),
            )
        ],
        "target": [
            _describe_typed_place(object_type, place)
            for object_type, place in sorted(
                region.target,
                key=lambda item: (item[0], _sort_petri_net_node(item[1])),
            )
        ],
        "activities": sorted(region.activities),
        "local_regions": [
            serialize_local_region(local_region)
            for local_region in sorted(region.local_regions, key=_sort_local_region_key)
        ],
    }


def serialize_local_region(region: LocalRegion):
    return {
        "object_type": region.object_type,
        "source": _describe_typed_place(region.object_type, region.source),
        "internal": [
            _describe_typed_vertex(region.object_type, vertex)
            for vertex in sorted(region.internal, key=_sort_vertex)
        ],
        "target": _describe_typed_place(region.object_type, region.target),
        "activities": sorted(region.activities),
    }


def build_object_centric_region_highlight(region, border_color="#f7d7a6", fillcolor="#f7d7a6", marker=None):
    transition_keys = set()
    place_keys = set()
    arc_keys = set()

    transition_keys.update(_region_transition_keys(region))
    place_keys.update(_region_place_keys(region))
    arc_keys.update(_region_arc_keys(region))

    return {
        "id": "object_centric_region",
        "color": border_color,
        "fillcolor": fillcolor,
        "marker": marker,
        "transition_keys": frozenset(transition_keys),
        "place_keys": frozenset(place_keys),
        "arc_keys": frozenset(arc_keys),
    }


def _region_transition_keys(region):
    return {
        _transition_key(local_region.object_type, vertex)
        for local_region in region.local_regions
        for vertex in local_region.internal
        if isinstance(vertex, PetriNet.Transition)
    }


def _region_place_keys(region):
    place_keys = set(region.source) | set(region.target)
    place_keys.update(
        (local_region.object_type, vertex)
        for local_region in region.local_regions
        for vertex in local_region.internal
        if isinstance(vertex, PetriNet.Place)
    )
    return {
        _place_key(object_type, place)
        for object_type, place in place_keys
    }


def _region_arc_keys(region):
    arc_keys = set()
    for local_region in region.local_regions:
        object_type = local_region.object_type
        region_vertices = local_region.vertices
        for vertex in region_vertices:
            for arc in vertex.in_arcs | vertex.out_arcs:
                if arc.source in region_vertices and arc.target in region_vertices:
                    arc_keys.add(_arc_key(object_type, arc))
    return arc_keys


def _describe_region_arc(arc_key):
    _, object_type, arc = arc_key
    return {
        "object_type": object_type,
        "source": _describe_place_key(_place_key(object_type, arc.source))
        if isinstance(arc.source, PetriNet.Place)
        else _describe_transition_key(_transition_key(object_type, arc.source)),
        "target": _describe_place_key(_place_key(object_type, arc.target))
        if isinstance(arc.target, PetriNet.Place)
        else _describe_transition_key(_transition_key(object_type, arc.target)),
    }


def _normalize_allowed_activities(contexts, allowed_activities):
    if allowed_activities is None:
        return frozenset(
            label
            for context in contexts.values()
            for label in context.visible_labels
        )
    return frozenset(
        activity
        for activity in allowed_activities
        if activity is not None
    )


def _build_object_type_context(object_type, net):
    places = tuple(sorted(net.places, key=_sort_petri_net_node))
    transitions = tuple(sorted(net.transitions, key=_sort_petri_net_node))
    vertices = frozenset(places) | frozenset(transitions)

    out_neighbors = {vertex: set() for vertex in vertices}
    in_neighbors = {vertex: set() for vertex in vertices}
    undirected_neighbors = {vertex: set() for vertex in vertices}
    visible_labels = set()

    for transition in transitions:
        if transition.label is not None:
            visible_labels.add(transition.label)

    for arc in net.arcs:
        out_neighbors[arc.source].add(arc.target)
        in_neighbors[arc.target].add(arc.source)
        undirected_neighbors[arc.source].add(arc.target)
        undirected_neighbors[arc.target].add(arc.source)

    return _ObjectTypeContext(
        object_type=object_type,
        net=net,
        places=places,
        transitions=transitions,
        vertices=vertices,
        out_neighbors={
            vertex: frozenset(neighbors)
            for vertex, neighbors in out_neighbors.items()
        },
        in_neighbors={
            vertex: frozenset(neighbors)
            for vertex, neighbors in in_neighbors.items()
        },
        undirected_neighbors={
            vertex: frozenset(neighbors)
            for vertex, neighbors in undirected_neighbors.items()
        },
        visible_labels=frozenset(visible_labels),
    )


def _build_activity_to_object_types(contexts):
    activity_to_object_types = defaultdict(set)
    for object_type, context in contexts.items():
        for label in context.visible_labels:
            activity_to_object_types[label].add(object_type)
    return {
        activity: frozenset(object_types)
        for activity, object_types in activity_to_object_types.items()
    }


def _maximal_regions_avoiding_labels(context, blocked_activities):
    forbidden_transitions = {
        transition
        for transition in context.transitions
        if transition.label is not None and transition.label in blocked_activities
    }
    return _maximal_regions_inside_universe(
        context,
        context.vertices,
        forbidden_transitions,
    )


def _maximal_regions_inside_universe(context, universe, forbidden_transitions):
    allowed_vertices = frozenset(universe) - frozenset(forbidden_transitions)
    regions = set()
    allowed_components = _weakly_connected_components(
        context.undirected_neighbors,
        allowed_vertices,
        _sort_vertex,
    )
    for component in allowed_components:
        component_places = [
            place
            for place in context.places
            if place in component
        ]
        for source in component_places:
            for target in component_places:
                region = _largest_region_for_pair_inside_universe(
                    context,
                    allowed_vertices,
                    source,
                    target,
                )
                if region is not None and region.internal:
                    regions.add(region)

    return _keep_only_inclusion_maximal_regions_of_same_type(regions)


def _largest_region_for_pair_inside_universe(context, allowed_vertices, source, target):
    outside_vertices = context.vertices - frozenset(allowed_vertices)
    reachable_from_source = _reachable_vertices(
        context.out_neighbors,
        frozenset(allowed_vertices),
        source,
    )
    can_reach_target = _reachable_vertices(
        context.in_neighbors,
        frozenset(allowed_vertices),
        target,
    )
    internal_candidates = frozenset(allowed_vertices) - {source, target}
    components = _weakly_connected_components(
        context.undirected_neighbors,
        internal_candidates,
        _sort_vertex,
    )

    internal_vertices = set()
    for component in components:
        if not component <= reachable_from_source:
            continue
        if not component <= can_reach_target:
            continue
        if any(
            context.undirected_neighbors[vertex] & outside_vertices
            for vertex in component
        ):
            continue
        if source != target:
            if any(source in context.out_neighbors[vertex] for vertex in component):
                continue
            if any(vertex in context.out_neighbors[target] for vertex in component):
                continue
        internal_vertices.update(component)

    return _make_local_region(
        context.object_type,
        source,
        internal_vertices,
        target,
    )


def _reachable_vertices(neighbors_by_vertex, allowed_vertices, start_vertex):
    if start_vertex not in allowed_vertices:
        return frozenset()

    visited = {start_vertex}
    queue = deque([start_vertex])

    while queue:
        current_vertex = queue.popleft()
        for next_vertex in neighbors_by_vertex.get(current_vertex, ()):
            if next_vertex not in allowed_vertices or next_vertex in visited:
                continue
            visited.add(next_vertex)
            queue.append(next_vertex)

    return frozenset(visited)


def _weakly_connected_components(neighbors_by_vertex, vertices, sort_key):
    remaining_vertices = set(vertices)
    components = []

    while remaining_vertices:
        start_vertex = min(remaining_vertices, key=sort_key)
        queue = deque([start_vertex])
        component = set()

        while queue:
            current_vertex = queue.popleft()
            if current_vertex not in remaining_vertices:
                continue

            remaining_vertices.remove(current_vertex)
            component.add(current_vertex)
            for next_vertex in neighbors_by_vertex.get(current_vertex, ()):
                if next_vertex in remaining_vertices:
                    queue.append(next_vertex)

        components.append(frozenset(component))

    components.sort(key=lambda component: (sort_key(min(component, key=sort_key)), len(component)))
    return tuple(components)


def _keep_only_inclusion_maximal_regions_of_same_type(regions):
    regions_by_type = defaultdict(list)
    for region in regions:
        regions_by_type[region.object_type].append(region)

    maximal_regions = set()
    for same_type_regions in regions_by_type.values():
        for region in same_type_regions:
            if any(
                region.vertices < other_region.vertices
                for other_region in same_type_regions
            ):
                continue
            maximal_regions.add(region)

    return frozenset(maximal_regions)


def _build_obligation_graph(regions, activity_to_object_types):
    adjacency = defaultdict(set)
    obligation_nodes = set()

    for region in regions:
        adjacency.setdefault(region, set())

    visible_activities = sorted({
        activity
        for region in regions
        for activity in region.activities
    })
    for activity in visible_activities:
        adjacency.setdefault(activity, set())
        for object_type in sorted(activity_to_object_types.get(activity, ())):
            obligation_node = (activity, object_type)
            obligation_nodes.add(obligation_node)
            adjacency.setdefault(obligation_node, set())
            adjacency[activity].add(obligation_node)
            adjacency[obligation_node].add(activity)

    for region in regions:
        for activity in region.activities:
            obligation_node = (activity, region.object_type)
            adjacency.setdefault(obligation_node, set())
            adjacency[region].add(obligation_node)
            adjacency[obligation_node].add(region)

    return _ObligationGraph(
        adjacency={
            node: frozenset(neighbors)
            for node, neighbors in adjacency.items()
        },
        obligation_nodes=frozenset(obligation_nodes),
    )


def _open_obligation_nodes(graph):
    return [
        obligation_node
        for obligation_node in sorted(graph.obligation_nodes)
        if all(
            not isinstance(neighbor, LocalRegion)
            for neighbor in graph.adjacency.get(obligation_node, ())
        )
    ]


def _prune_unsupported_activities(regions, blocked_activities, contexts, activity_to_object_types):
    current_regions = frozenset(regions)
    current_blocked_activities = frozenset(blocked_activities)

    while True:
        graph = _build_obligation_graph(current_regions, activity_to_object_types)
        open_obligation_nodes = _open_obligation_nodes(graph)
        if not open_obligation_nodes:
            return current_regions, current_blocked_activities, graph

        activity, _ = open_obligation_nodes[0]
        current_blocked_activities = current_blocked_activities | {activity}
        current_regions = _keep_only_inclusion_maximal_regions_of_same_type(
            _replace_regions_avoiding_labels(
                current_regions,
                current_blocked_activities,
                contexts,
            )
        )


def _replace_regions_avoiding_labels(regions, blocked_activities, contexts):
    replacement_regions = set()

    for region in regions:
        if not region.activities & blocked_activities:
            replacement_regions.add(region)
            continue

        context = contexts[region.object_type]
        universe = region.vertices
        forbidden_transitions = {
            vertex
            for vertex in universe
            if isinstance(vertex, PetriNet.Transition)
            and vertex.label is not None
            and vertex.label in blocked_activities
        }
        replacement_regions.update(
            _maximal_regions_inside_universe(
                context,
                universe,
                forbidden_transitions,
            )
        )

    return frozenset(replacement_regions)


def _resolve_region_set(regions, blocked_activities, contexts, activity_to_object_types):
    pruned_regions, blocked_activities, graph = _prune_unsupported_activities(
        regions,
        blocked_activities,
        contexts,
        activity_to_object_types,
    )
    if not pruned_regions:
        return frozenset()

    output_regions = set()
    for component in _weakly_connected_components(
        graph.adjacency,
        frozenset(graph.adjacency),
        _sort_graph_node,
    ):
        component_regions = frozenset(
            node
            for node in component
            if isinstance(node, LocalRegion)
        )
        if not component_regions:
            continue

        if _has_at_most_one_region_per_object_type(component_regions):
            output_regions.add(_make_output_region(component_regions))
            continue

        object_type = _choose_conflicting_object_type(component_regions)
        conflicting_regions = frozenset(
            region
            for region in component_regions
            if region.object_type == object_type
        )
        for selected_region in sorted(conflicting_regions, key=_sort_local_region_key):
            output_regions.update(
                _resolve_region_set(
                    (component_regions - conflicting_regions) | {selected_region},
                    blocked_activities,
                    contexts,
                    activity_to_object_types,
                )
            )

    return _remove_duplicate_and_contained_outputs(output_regions)


def _has_at_most_one_region_per_object_type(regions):
    counts = defaultdict(int)
    for region in regions:
        counts[region.object_type] += 1
        if counts[region.object_type] > 1:
            return False
    return True


def _choose_conflicting_object_type(regions):
    counts = defaultdict(int)
    for region in regions:
        counts[region.object_type] += 1

    conflicting_counts = [
        (count, object_type)
        for object_type, count in counts.items()
        if count > 1
    ]
    _, object_type = min(conflicting_counts, key=lambda item: (item[0], item[1]))
    return object_type


def _make_output_region(regions):
    source = set()
    internal = set()
    target = set()
    activities = set()

    for region in regions:
        source.add((region.object_type, region.source))
        target.add((region.object_type, region.target))
        internal.update((region.object_type, vertex) for vertex in region.internal)
        activities.update(region.activities)

    return ObjectCentricRegion(
        source=frozenset(source),
        internal=frozenset(internal),
        target=frozenset(target),
        local_regions=frozenset(regions),
        activities=frozenset(activities),
    )


def _remove_duplicate_and_contained_outputs(output_regions):
    unique_regions = set(output_regions)
    maximal_regions = set()

    for region in unique_regions:
        if any(
            region != other_region and _output_region_is_contained_in(region, other_region)
            for other_region in unique_regions
        ):
            continue
        maximal_regions.add(region)

    return frozenset(maximal_regions)


def _remove_trivial_regions(output_regions):
    return frozenset(
        region
        for region in output_regions
        if not _is_trivial_region(region)
    )


def _is_trivial_region(region):
    return _count_region_transitions(region) == 1


def _count_region_transitions(region):
    return sum(
        1
        for _, vertex in region.internal
        if isinstance(vertex, PetriNet.Transition)
    )


def _output_region_is_contained_in(region, other_region):
    region_vertices = region.source | region.internal | region.target
    other_region_vertices = other_region.source | other_region.internal | other_region.target
    return region_vertices < other_region_vertices


def _make_local_region(object_type, source, internal_vertices, target):
    internal_vertices = frozenset(internal_vertices)
    vertices = internal_vertices | {source, target}
    activities = frozenset(
        vertex.label
        for vertex in internal_vertices
        if isinstance(vertex, PetriNet.Transition) and vertex.label is not None
    )
    return LocalRegion(
        object_type=object_type,
        source=source,
        internal=internal_vertices,
        target=target,
        vertices=frozenset(vertices),
        activities=activities,
    )


def _describe_typed_place(object_type, place):
    return {
        "object_type": object_type,
        "kind": "place",
        "name": getattr(place, "name", str(place)),
    }


def _describe_typed_vertex(object_type, vertex):
    if isinstance(vertex, PetriNet.Place):
        return _describe_typed_place(object_type, vertex)

    return {
        "object_type": object_type,
        "kind": "silent" if vertex.label is None else "activity",
        "name": getattr(vertex, "name", str(vertex)),
        "label": vertex.label,
    }


def _sort_petri_net_node(node):
    return getattr(node, "name", ""), id(node)


def _sort_vertex(vertex):
    if isinstance(vertex, PetriNet.Place):
        return 0, getattr(vertex, "name", ""), id(vertex)
    return (
        1,
        vertex.label is None,
        vertex.label or "",
        getattr(vertex, "name", ""),
        id(vertex),
    )


def _sort_graph_node(node):
    if isinstance(node, LocalRegion):
        return 0, _sort_local_region_key(node)
    if isinstance(node, tuple) and len(node) == 2:
        return 1, node
    if isinstance(node, str):
        return 2, node
    return 3, repr(node)


def _sort_local_region_key(region):
    return (
        region.object_type,
        _sort_petri_net_node(region.source),
        _sort_petri_net_node(region.target),
        len(region.internal),
        tuple(_sort_vertex(vertex) for vertex in sorted(region.internal, key=_sort_vertex)),
    )


def _sort_output_region_key(region):
    return (
        tuple(
            (object_type, _sort_petri_net_node(place))
            for object_type, place in sorted(
                region.source,
                key=lambda item: (item[0], _sort_petri_net_node(item[1])),
            )
        ),
        tuple(
            (object_type, _sort_vertex(vertex))
            for object_type, vertex in sorted(
                region.internal,
                key=lambda item: (item[0], _sort_vertex(item[1])),
            )
        ),
        tuple(
            (object_type, _sort_petri_net_node(place))
            for object_type, place in sorted(
                region.target,
                key=lambda item: (item[0], _sort_petri_net_node(item[1])),
            )
        ),
    )


__all__ = [
    "LocalRegion",
    "ObjectCentricRegion",
    "build_object_centric_region_highlight",
    "detect_object_centric_regions",
    "serialize_local_region",
    "serialize_object_centric_region",
]
