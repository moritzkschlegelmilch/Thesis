from collections import defaultdict, deque
import colorsys

from pm4py.objects.petri_net.obj import PetriNet


def _transition_key(object_type, transition):
    if transition.label is None:
        return ("silent", object_type, transition)
    return ("activity", transition.label)


def _place_key(object_type, place):
    return ("place", object_type, place)


def _arc_key(object_type, arc):
    return ("arc", object_type, arc)


def _sort_transition_key(transition_key):
    if transition_key[0] == "activity":
        return 0, transition_key[1], "", 0
    return (
        1,
        transition_key[1],
        getattr(transition_key[2], "name", ""),
        id(transition_key[2]),
    )


def _sort_place_key(place_key):
    return (
        place_key[1],
        getattr(place_key[2], "name", ""),
        id(place_key[2]),
    )


def _iter_bits(mask):
    while mask:
        lowest_bit = mask & -mask
        yield lowest_bit.bit_length() - 1
        mask ^= lowest_bit


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def _build_component_colors(component_count):
    colors = []
    golden_ratio = 0.618033988749895
    used_colors = set()

    for index in range(component_count):
        color = None
        for variant in range(18):
            hue = (index * golden_ratio + variant * (1 / 54)) % 1.0
            saturation = 0.56 + 0.08 * (variant % 3)
            lightness = 0.58 + 0.06 * ((variant // 3) % 3)
            candidate = _rgb_to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))
            if candidate not in used_colors:
                color = candidate
                break

        if color is None:
            color = f"#{(index * 0x9E3779B1 + 0xABCDE) & 0xFFFFFF:06x}"
            while color in used_colors:
                color = f"#{(int(color[1:], 16) + 1) & 0xFFFFFF:06x}"

        used_colors.add(color)
        colors.append(color)

    return colors


def _describe_transition_key(transition_key):
    if transition_key[0] == "activity":
        return {
            "kind": "activity",
            "label": transition_key[1],
        }

    return {
        "kind": "silent",
        "object_type": transition_key[1],
        "name": getattr(transition_key[2], "name", str(transition_key[2])),
    }


def _describe_place_key(place_key):
    return {
        "object_type": place_key[1],
        "name": getattr(place_key[2], "name", str(place_key[2])),
    }


def _describe_arc_info(arc_info):
    source = arc_info["source"]
    target = arc_info["target"]
    return {
        "object_type": arc_info["object_type"],
        "source": _describe_place_key(source) if source[0] == "place" else _describe_transition_key(source),
        "target": _describe_place_key(target) if target[0] == "place" else _describe_transition_key(target),
    }


def _build_merged_ocpn(ocpn, activity_to_layer, reference_layer):
    places = {}
    transitions = {}
    arcs = {}
    candidate_transitions = set()

    for object_type, (net, _, _) in ocpn["petri_nets"].items():
        for place in net.places:
            place_key = _place_key(object_type, place)
            places.setdefault(place_key, {
                "key": place_key,
                "object_type": object_type,
                "place": place,
                "pred_transitions": set(),
                "succ_transitions": set(),
                "pred_arcs": set(),
                "succ_arcs": set(),
            })

        for transition in net.transitions:
            transition_key = _transition_key(object_type, transition)
            transition_info = transitions.setdefault(transition_key, {
                "key": transition_key,
                "is_silent": transition.label is None,
                "label": transition.label,
                "object_types": set(),
                "original_transitions": set(),
                "pred_places": set(),
                "succ_places": set(),
                "in_arcs": set(),
                "out_arcs": set(),
            })
            transition_info["object_types"].add(object_type)
            transition_info["original_transitions"].add((object_type, transition))

            if transition.label is None or activity_to_layer.get(transition.label, float("inf")) < reference_layer:
                candidate_transitions.add(transition_key)

        for arc in net.arcs:
            arc_key = _arc_key(object_type, arc)

            if isinstance(arc.source, PetriNet.Place):
                place_key = _place_key(object_type, arc.source)
                transition_key = _transition_key(object_type, arc.target)
                source_key = place_key
                target_key = transition_key
                places[place_key]["succ_transitions"].add(transition_key)
                places[place_key]["succ_arcs"].add(arc_key)
                transitions[transition_key]["pred_places"].add(place_key)
                transitions[transition_key]["in_arcs"].add(arc_key)
            else:
                transition_key = _transition_key(object_type, arc.source)
                place_key = _place_key(object_type, arc.target)
                source_key = transition_key
                target_key = place_key
                places[place_key]["pred_transitions"].add(transition_key)
                places[place_key]["pred_arcs"].add(arc_key)
                transitions[transition_key]["succ_places"].add(place_key)
                transitions[transition_key]["out_arcs"].add(arc_key)

            arcs[arc_key] = {
                "key": arc_key,
                "object_type": object_type,
                "arc": arc,
                "source": source_key,
                "target": target_key,
                "transition_key": transition_key,
                "place_key": place_key,
            }

    place_graph_by_type = defaultdict(lambda: defaultdict(set))
    for place_key in places:
        place_graph_by_type[place_key[1]][place_key]

    for transition_info in transitions.values():
        pred_places_by_type = defaultdict(list)
        succ_places_by_type = defaultdict(list)

        for place_key in transition_info["pred_places"]:
            pred_places_by_type[place_key[1]].append(place_key)
        for place_key in transition_info["succ_places"]:
            succ_places_by_type[place_key[1]].append(place_key)

        for object_type in pred_places_by_type.keys() | succ_places_by_type.keys():
            for pred_place_key in pred_places_by_type[object_type]:
                for succ_place_key in succ_places_by_type[object_type]:
                    place_graph_by_type[object_type][pred_place_key].add(succ_place_key)

    return {
        "places": places,
        "transitions": transitions,
        "arcs": arcs,
        "candidate_transitions": candidate_transitions,
        "place_graph_by_type": {
            object_type: {
                place_key: tuple(sorted(successors, key=_sort_place_key))
                for place_key, successors in adjacency.items()
            }
            for object_type, adjacency in place_graph_by_type.items()
        },
    }


def _candidate_transition_components(merged_ocpn):
    candidate_transitions = merged_ocpn["candidate_transitions"]
    transitions = merged_ocpn["transitions"]
    places = merged_ocpn["places"]
    visited_transitions = set()
    components = []

    for start_transition in sorted(candidate_transitions, key=_sort_transition_key):
        if start_transition in visited_transitions:
            continue

        component_transitions = set()
        component_places = set()
        queue = deque([("transition", start_transition)])

        while queue:
            node_type, node_key = queue.popleft()
            if node_type == "transition":
                if node_key in component_transitions:
                    continue

                component_transitions.add(node_key)
                visited_transitions.add(node_key)
                transition_info = transitions[node_key]
                adjacent_places = transition_info["pred_places"] | transition_info["succ_places"]
                for place_key in adjacent_places:
                    queue.append(("place", place_key))
            else:
                if node_key in component_places:
                    continue

                component_places.add(node_key)
                place_info = places[node_key]
                adjacent_transitions = place_info["pred_transitions"] | place_info["succ_transitions"]
                for transition_key in adjacent_transitions:
                    if transition_key in candidate_transitions:
                        queue.append(("transition", transition_key))

        components.append({
            "transitions": frozenset(component_transitions),
            "places": frozenset(component_places),
        })

    return components


def _prepare_component_search_data(merged_ocpn, component):
    transition_keys = sorted(component["transitions"], key=_sort_transition_key)
    place_keys = sorted(component["places"], key=_sort_place_key)

    transition_index = {
        transition_key: index
        for index, transition_key in enumerate(transition_keys)
    }
    place_index = {
        place_key: index
        for index, place_key in enumerate(place_keys)
    }

    transition_degrees = []
    transition_edges_by_type = []
    transition_arc_keys = []
    transition_place_indices = []
    for transition_key in transition_keys:
        transition_info = merged_ocpn["transitions"][transition_key]
        transition_degrees.append(len(transition_info["pred_places"]) + len(transition_info["succ_places"]))
        transition_arc_keys.append(frozenset(transition_info["in_arcs"] | transition_info["out_arcs"]))
        transition_place_indices.append(tuple(sorted({
            place_index[place_key]
            for place_key in transition_info["pred_places"] | transition_info["succ_places"]
        })))

        object_type_to_places = defaultdict(lambda: {"pred": [], "succ": []})
        for place_key in transition_info["pred_places"]:
            object_type_to_places[place_key[1]]["pred"].append(place_index[place_key])
        for place_key in transition_info["succ_places"]:
            object_type_to_places[place_key[1]]["succ"].append(place_index[place_key])

        transition_edges = {}
        for object_type, type_places in object_type_to_places.items():
            transition_edges[object_type] = (
                tuple(sorted(type_places["pred"])),
                tuple(sorted(type_places["succ"])),
            )
        transition_edges_by_type.append(transition_edges)

    place_data = []
    for place_key in place_keys:
        place_info = merged_ocpn["places"][place_key]
        pred_mask = 0
        succ_mask = 0

        for transition_key in place_info["pred_transitions"]:
            if transition_key in transition_index:
                pred_mask |= 1 << transition_index[transition_key]
        for transition_key in place_info["succ_transitions"]:
            if transition_key in transition_index:
                succ_mask |= 1 << transition_index[transition_key]

        incident_mask = pred_mask | succ_mask
        place_data.append({
            "key": place_key,
            "object_type": place_key[1],
            "pred_mask": pred_mask,
            "succ_mask": succ_mask,
            "incident_mask": incident_mask,
            "pred_total": len(place_info["pred_transitions"]),
            "succ_total": len(place_info["succ_transitions"]),
        })

    return {
        "transition_keys": tuple(transition_keys),
        "place_keys": tuple(place_keys),
        "transition_degrees": tuple(transition_degrees),
        "transition_arc_keys": tuple(transition_arc_keys),
        "transition_place_indices": tuple(transition_place_indices),
        "transition_edges_by_type": tuple(transition_edges_by_type),
        "place_data": tuple(place_data),
        "place_graph_by_type": merged_ocpn["place_graph_by_type"],
    }


def _selected_transition_components(component_search_data, mask):
    place_data = component_search_data["place_data"]
    transition_place_indices = component_search_data["transition_place_indices"]
    selected_transition_indices = tuple(_iter_bits(mask))
    unvisited_transitions = set(selected_transition_indices)
    connected_components = []

    while unvisited_transitions:
        start_transition = next(iter(unvisited_transitions))
        transition_component = set()
        place_component = set()
        queue = deque([("transition", start_transition)])

        while queue:
            node_type, node_index = queue.popleft()
            if node_type == "transition":
                if node_index in transition_component:
                    continue

                transition_component.add(node_index)
                unvisited_transitions.discard(node_index)
                for place_index in transition_place_indices[node_index]:
                    if place_data[place_index]["incident_mask"] & mask:
                        queue.append(("place", place_index))
            else:
                if node_index in place_component:
                    continue

                place_component.add(node_index)
                incident_transition_mask = place_data[node_index]["incident_mask"] & mask
                for transition_index in _iter_bits(incident_transition_mask):
                    queue.append(("transition", transition_index))

        connected_components.append({
            "transitions": tuple(sorted(transition_component)),
            "places": tuple(sorted(place_component)),
        })

    connected_components.sort(
        key=lambda component: (
            -len(component["transitions"]),
            -len(component["places"]),
            component["transitions"],
        )
    )
    return connected_components


def _has_mixed_boundary_path(place_graph, start_place_key, end_place_key, inside_place_keys):
    visited_states = {(start_place_key, False, False)}
    queue = deque([(start_place_key, False, False)])

    while queue:
        current_place_key, seen_inside_places, seen_outside_places = queue.popleft()
        for next_place_key in place_graph.get(current_place_key, ()):
            next_seen_inside_places = seen_inside_places
            next_seen_outside_places = seen_outside_places

            if next_place_key != end_place_key:
                if next_place_key in inside_place_keys:
                    next_seen_inside_places = True
                else:
                    next_seen_outside_places = True

            if (
                next_place_key == end_place_key
                and next_seen_inside_places
                and next_seen_outside_places
            ):
                return True

            next_state = (
                next_place_key,
                next_seen_inside_places,
                next_seen_outside_places,
            )
            if next_state in visited_states:
                continue

            visited_states.add(next_state)
            queue.append(next_state)

    return False


def _analyze_component_mask(component_search_data, mask):
    if mask == 0:
        return False, ()

    place_data = component_search_data["place_data"]
    transition_edges_by_type = component_search_data["transition_edges_by_type"]
    place_graph_by_type = component_search_data["place_graph_by_type"]

    included_places_by_type = defaultdict(list)
    selected_pred_counts = {}
    selected_succ_counts = {}
    input_place_candidates = {}
    output_place_candidates = {}
    connected_components = _selected_transition_components(component_search_data, mask)
    if len(connected_components) > 1:
        return False, tuple(component["transitions"][0] for component in connected_components)

    selected_transition_indices = connected_components[0]["transitions"]

    for place_index, place_info in enumerate(place_data):
        selected_pred_count = (mask & place_info["pred_mask"]).bit_count()
        selected_succ_count = (mask & place_info["succ_mask"]).bit_count()

        if selected_pred_count == 0 and selected_succ_count == 0:
            continue

        is_input_place = selected_pred_count == 0
        is_output_place = selected_succ_count == 0
        is_internal_place = (
            selected_pred_count == place_info["pred_total"]
            and selected_succ_count == place_info["succ_total"]
        )

        if not (is_input_place or is_output_place or is_internal_place):
            conflict_mask = mask & place_info["incident_mask"]
            return False, tuple(sorted(_iter_bits(conflict_mask)))

        included_places_by_type[place_info["object_type"]].append(place_index)
        selected_pred_counts[place_index] = selected_pred_count
        selected_succ_counts[place_index] = selected_succ_count
        input_place_candidates[place_index] = is_input_place
        output_place_candidates[place_index] = is_output_place

    for object_type, place_indices in included_places_by_type.items():
        input_places = [
            place_index
            for place_index in place_indices
            if input_place_candidates[place_index]
        ]
        output_places = [
            place_index
            for place_index in place_indices
            if output_place_candidates[place_index]
        ]

        if len(input_places) != 1 or len(output_places) != 1:
            conflict_mask = 0
            for place_index in place_indices:
                conflict_mask |= place_data[place_index]["incident_mask"]
            conflict_mask &= mask
            return False, tuple(sorted(_iter_bits(conflict_mask)))

        unique_input_place = input_places[0]
        unique_output_place = output_places[0]

        forward_adjacency = defaultdict(set)
        backward_adjacency = defaultdict(set)
        for transition_index in selected_transition_indices:
            transition_edges = transition_edges_by_type[transition_index].get(object_type)
            if transition_edges is None:
                continue

            pred_places, succ_places = transition_edges
            if not pred_places or not succ_places:
                continue

            for pred_place_index in pred_places:
                for succ_place_index in succ_places:
                    forward_adjacency[pred_place_index].add(succ_place_index)
                    backward_adjacency[succ_place_index].add(pred_place_index)

        reachable_from_input = {unique_input_place}
        queue = deque([unique_input_place])
        while queue:
            current_place = queue.popleft()
            for next_place in forward_adjacency.get(current_place, ()):
                if next_place not in reachable_from_input:
                    reachable_from_input.add(next_place)
                    queue.append(next_place)

        can_reach_output = {unique_output_place}
        queue = deque([unique_output_place])
        while queue:
            current_place = queue.popleft()
            for previous_place in backward_adjacency.get(current_place, ()):
                if previous_place not in can_reach_output:
                    can_reach_output.add(previous_place)
                    queue.append(previous_place)

        for place_index in place_indices:
            if place_index in reachable_from_input and place_index in can_reach_output:
                continue

            conflict_mask = 0
            for candidate_place_index in place_indices:
                conflict_mask |= place_data[candidate_place_index]["incident_mask"]
            conflict_mask &= mask
            return False, tuple(sorted(_iter_bits(conflict_mask)))

        inside_place_keys = {
            place_data[place_index]["key"]
            for place_index in place_indices
        }
        full_place_graph = place_graph_by_type.get(object_type, {})
        if _has_mixed_boundary_path(
            full_place_graph,
            place_data[unique_input_place]["key"],
            place_data[unique_output_place]["key"],
            inside_place_keys,
        ):
            conflict_mask = 0
            for candidate_place_index in place_indices:
                conflict_mask |= place_data[candidate_place_index]["incident_mask"]
            conflict_mask &= mask
            return False, tuple(sorted(_iter_bits(conflict_mask)))

    return True, ()


def _enumerate_maximal_component_masks(component_search_data):
    transition_count = len(component_search_data["transition_keys"])
    if transition_count == 0:
        return []

    full_mask = (1 << transition_count) - 1
    visited_masks = set()
    feasible_masks = set()

    def is_subsumed_by_feasible(mask):
        for feasible_mask in feasible_masks:
            if mask & ~feasible_mask == 0:
                return True
        return False

    def search(mask):
        if mask == 0 or mask in visited_masks or is_subsumed_by_feasible(mask):
            return

        visited_masks.add(mask)
        is_feasible, conflict_indices = _analyze_component_mask(component_search_data, mask)
        if is_feasible:
            feasible_masks.add(mask)
            return

        if not conflict_indices:
            return

        branching_order = sorted(
            conflict_indices,
            key=lambda index: component_search_data["transition_degrees"][index],
            reverse=True,
        )
        for transition_index in branching_order:
            search(mask & ~(1 << transition_index))

    search(full_mask)

    maximal_masks = []
    for mask in feasible_masks:
        if any(mask != other_mask and mask & ~other_mask == 0 for other_mask in feasible_masks):
            continue
        maximal_masks.append(mask)

    maximal_masks.sort(key=lambda mask: (-mask.bit_count(), mask))
    return maximal_masks


def _build_component_fragment(component_search_data, mask):
    transition_keys = component_search_data["transition_keys"]
    place_data = component_search_data["place_data"]
    transition_arc_keys = component_search_data["transition_arc_keys"]

    selected_transition_indices = tuple(_iter_bits(mask))
    selected_transition_keys = frozenset(
        transition_keys[transition_index]
        for transition_index in selected_transition_indices
    )

    selected_place_keys = set()
    selected_object_types = set()
    for place_index, place_info in enumerate(place_data):
        selected_pred_count = (mask & place_info["pred_mask"]).bit_count()
        selected_succ_count = (mask & place_info["succ_mask"]).bit_count()
        if selected_pred_count == 0 and selected_succ_count == 0:
            continue

        selected_place_keys.add(place_info["key"])
        selected_object_types.add(place_info["object_type"])

    selected_arc_keys = set()
    for transition_index in selected_transition_indices:
        selected_arc_keys.update(transition_arc_keys[transition_index])

    return {
        "transition_keys": selected_transition_keys,
        "place_keys": frozenset(selected_place_keys),
        "arc_keys": frozenset(selected_arc_keys),
        "object_types": frozenset(selected_object_types),
    }


def detect_subprocess_components(ocpn, activity_to_layer, reference_layer):
    if ocpn is None:
        return []

    merged_ocpn = _build_merged_ocpn(ocpn, activity_to_layer, reference_layer)
    candidate_components = _candidate_transition_components(merged_ocpn)

    connected_components = []
    for component in candidate_components:
        component_search_data = _prepare_component_search_data(merged_ocpn, component)
        maximal_masks = _enumerate_maximal_component_masks(component_search_data)
        component_fragments = [
            _build_component_fragment(component_search_data, mask)
            for mask in maximal_masks
        ]
        connected_components.extend(
            fragment
            for fragment in component_fragments
            if len(fragment["transition_keys"]) >= 2
        )

    component_colors = _build_component_colors(len(connected_components))

    serialized_components = []
    for index, (component, component_color) in enumerate(
        zip(
            sorted(
                connected_components,
                key=lambda component: (
                    -len(component["transition_keys"]),
                    -len(component["place_keys"]),
                    tuple(sorted(component["object_types"])),
                ),
            ),
            component_colors,
        ),
        start=1,
    ):
        transition_keys = component["transition_keys"]
        place_keys = component["place_keys"]
        arc_keys = component["arc_keys"]

        serialized_components.append({
            "id": f"subprocess_{index}",
            "color": component_color,
            "object_types": sorted(component["object_types"]),
            "transition_keys": transition_keys,
            "place_keys": place_keys,
            "arc_keys": arc_keys,
            "transitions": [
                _describe_transition_key(transition_key)
                for transition_key in sorted(transition_keys, key=_sort_transition_key)
            ],
            "places": [
                _describe_place_key(place_key)
                for place_key in sorted(place_keys, key=_sort_place_key)
            ],
            "arcs": [
                _describe_arc_info(merged_ocpn["arcs"][arc_key])
                for arc_key in sorted(
                    arc_keys,
                    key=lambda arc_key: (
                        merged_ocpn["arcs"][arc_key]["object_type"],
                        _sort_place_key(merged_ocpn["arcs"][arc_key]["place_key"]),
                        _sort_transition_key(merged_ocpn["arcs"][arc_key]["transition_key"]),
                    ),
                )
            ],
        })

    return serialized_components
