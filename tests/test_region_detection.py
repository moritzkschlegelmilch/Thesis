import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from repo.discovery.region_detection import (
    _make_local_region,
    _make_output_region,
    _remove_duplicate_and_contained_outputs,
    detect_object_centric_regions,
)


def _build_net(name, places, transitions, arcs):
    net = PetriNet(name)
    place_nodes = {}
    transition_nodes = {}

    for place_name in places:
        place = PetriNet.Place(place_name)
        net.places.add(place)
        place_nodes[place_name] = place

    for transition_name, transition_label in transitions.items():
        transition = PetriNet.Transition(transition_name, transition_label)
        net.transitions.add(transition)
        transition_nodes[transition_name] = transition

    nodes = {}
    nodes.update(place_nodes)
    nodes.update(transition_nodes)

    for source_name, target_name in arcs:
        petri_utils.add_arc_from_to(nodes[source_name], nodes[target_name], net)

    return net, nodes


def _build_ocpn(nets_by_object_type):
    activities = sorted({
        transition.label
        for net, *_ in nets_by_object_type.values()
        for transition in net.transitions
        if transition.label is not None
    })

    return {
        "activities": activities,
        "petri_nets": {
            object_type: (
                entry[0],
                Marking(),
                Marking({place: 1 for place in entry[2]}) if len(entry) > 2 else Marking(),
            )
            for object_type, entry in nets_by_object_type.items()
        },
        "tbr_results": {},
        "double_arcs_on_activity": {
            object_type: {}
            for object_type in nets_by_object_type
        },
    }


def _typed_place_names(typed_places):
    return frozenset({
        (object_type, place.name)
        for object_type, place in typed_places
    })


def _typed_transition_labels(typed_vertices):
    return {
        (object_type, vertex.label)
        for object_type, vertex in typed_vertices
        if isinstance(vertex, PetriNet.Transition)
    }


class RegionDetectionTests(unittest.TestCase):
    def test_detect_object_centric_regions_filters_disallowed_visible_activities(self):
        item_net, _ = _build_net(
            "item",
            places=["p0", "p1", "p2", "p3"],
            transitions={
                "a_t": "a",
                "b_t": "b",
                "x_t": "x",
            },
            arcs=[
                ("p0", "a_t"),
                ("a_t", "p1"),
                ("p1", "b_t"),
                ("b_t", "p2"),
                ("p2", "x_t"),
                ("x_t", "p3"),
            ],
        )
        ocpn = _build_ocpn({"item": (item_net, {})})

        regions = detect_object_centric_regions(ocpn, {"a", "b"})

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].activities, frozenset({"a", "b"}))
        self.assertEqual(_typed_place_names(regions[0].source), {("item", "p0")})
        self.assertEqual(_typed_place_names(regions[0].target), {("item", "p2")})
        self.assertEqual(
            _typed_transition_labels(regions[0].internal),
            {("item", "a"), ("item", "b")},
        )

    def test_detect_object_centric_regions_prunes_unsupported_activities(self):
        item_net, _ = _build_net(
            "item",
            places=["p0", "p1", "p2"],
            transitions={
                "a_item": "a",
                "b_item": "b",
            },
            arcs=[
                ("p0", "a_item"),
                ("a_item", "p1"),
                ("p1", "b_item"),
                ("b_item", "p2"),
            ],
        )
        order_net, _ = _build_net(
            "order",
            places=["q0", "q1", "q2", "r0", "r1"],
            transitions={
                "a_order": "a",
                "b_order": "b",
            },
            arcs=[
                ("q0", "a_order"),
                ("a_order", "q1"),
                ("q1", "b_order"),
                ("r0", "b_order"),
                ("b_order", "q2"),
                ("b_order", "r1"),
            ],
        )
        ocpn = _build_ocpn({
            "item": (item_net, {}),
            "order": (order_net, {}),
        })

        regions = detect_object_centric_regions(ocpn, {"a", "b"})

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].activities, frozenset({"a"}))
        self.assertEqual(
            _typed_place_names(regions[0].source),
            frozenset({("item", "p0"), ("order", "q0")}),
        )
        self.assertEqual(
            _typed_place_names(regions[0].target),
            frozenset({("item", "p1"), ("order", "q1")}),
        )
        self.assertEqual(
            _typed_transition_labels(regions[0].internal),
            {("item", "a"), ("order", "a")},
        )

    def test_detect_object_centric_regions_branches_on_conflicting_same_type_regions(self):
        item_net, _ = _build_net(
            "item",
            places=["p0", "p1", "p2", "p3"],
            transitions={
                "a_item": "a",
                "x_item": "x",
                "b_item": "b",
            },
            arcs=[
                ("p0", "a_item"),
                ("a_item", "p1"),
                ("p1", "x_item"),
                ("x_item", "p2"),
                ("p2", "b_item"),
                ("b_item", "p3"),
            ],
        )
        order_net, _ = _build_net(
            "order",
            places=["q0", "q1", "q2"],
            transitions={
                "a_order": "a",
                "b_order": "b",
            },
            arcs=[
                ("q0", "a_order"),
                ("a_order", "q1"),
                ("q1", "b_order"),
                ("b_order", "q2"),
            ],
        )
        ocpn = _build_ocpn({
            "item": (item_net, {}),
            "order": (order_net, {}),
        })

        regions = detect_object_centric_regions(ocpn, {"a", "b"})

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            {region.activities for region in regions},
            {frozenset({"a"}), frozenset({"b"})},
        )
        self.assertTrue(all(len(region.local_regions) == 2 for region in regions))
        self.assertEqual(
            {_typed_place_names(region.source) for region in regions},
            {
                frozenset({("item", "p0"), ("order", "q0")}),
                frozenset({("item", "p2"), ("order", "q1")}),
            },
        )
        self.assertEqual(
            {_typed_place_names(region.target) for region in regions},
            {
                frozenset({("item", "p1"), ("order", "q1")}),
                frozenset({("item", "p3"), ("order", "q2")}),
            },
        )

    def test_detect_object_centric_regions_keeps_region_with_multiple_transitions(self):
        item_net, item_nodes = _build_net(
            "item",
            places=["p0", "p1", "p2", "p3"],
            transitions={
                "a_item": "a",
                "x_item": "x",
                "b_item": "b",
            },
            arcs=[
                ("p0", "a_item"),
                ("a_item", "p1"),
                ("p1", "x_item"),
                ("x_item", "p2"),
                ("p2", "b_item"),
                ("b_item", "p3"),
            ],
        )
        order_net, order_nodes = _build_net(
            "order",
            places=["q0", "q1", "q2"],
            transitions={
                "a_order": "a",
                "b_order": "b",
            },
            arcs=[
                ("q0", "a_order"),
                ("a_order", "q1"),
                ("q1", "b_order"),
                ("b_order", "q2"),
            ],
        )
        ocpn = _build_ocpn({
            "item": (item_net, {}, [item_nodes["p3"]]),
            "order": (order_net, {}, [order_nodes["q2"]]),
        })

        regions = detect_object_centric_regions(ocpn, {"a", "b"})

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            {region.activities for region in regions},
            {frozenset({"a"}), frozenset({"b"})},
        )

    def test_detect_object_centric_regions_removes_trivial_single_transition_region(self):
        item_net, item_nodes = _build_net(
            "item",
            places=["p0", "p1"],
            transitions={
                "a_item": "a",
            },
            arcs=[
                ("p0", "a_item"),
                ("a_item", "p1"),
            ],
        )
        ocpn = _build_ocpn({
            "item": (item_net, {}, [item_nodes["p1"]]),
        })

        regions = detect_object_centric_regions(ocpn, {"a"})

        self.assertEqual(regions, [])

    def test_detect_object_centric_regions_removes_trivial_single_silent_transition_region(self):
        item_net, _ = _build_net(
            "item",
            places=["p0", "p1", "p2"],
            transitions={
                "tau_item": None,
                "a_item": "a",
            },
            arcs=[
                ("p0", "tau_item"),
                ("tau_item", "p1"),
                ("p1", "a_item"),
                ("a_item", "p2"),
            ],
        )
        ocpn = _build_ocpn({
            "item": (item_net, {}),
        })

        regions = detect_object_centric_regions(ocpn, {"a"})

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].activities, frozenset({"a"}))
        self.assertGreater(
            sum(
                1
                for _, vertex in regions[0].internal
                if isinstance(vertex, PetriNet.Transition)
            ),
            1,
        )

    def test_remove_duplicate_and_contained_outputs_keeps_only_maximal_output(self):
        item_net, nodes = _build_net(
            "item",
            places=["p0", "p1", "p2"],
            transitions={
                "a_t": "a",
                "b_t": "b",
            },
            arcs=[
                ("p0", "a_t"),
                ("a_t", "p1"),
                ("p1", "b_t"),
                ("b_t", "p2"),
            ],
        )

        small_region = _make_local_region(
            "item",
            nodes["p0"],
            {nodes["a_t"]},
            nodes["p1"],
        )
        large_region = _make_local_region(
            "item",
            nodes["p0"],
            {nodes["a_t"], nodes["p1"], nodes["b_t"]},
            nodes["p2"],
        )
        small_output = _make_output_region({small_region})
        large_output = _make_output_region({large_region})

        output_regions = _remove_duplicate_and_contained_outputs(
            {small_output, large_output, large_output}
        )

        self.assertEqual(output_regions, frozenset({large_output}))


if __name__ == "__main__":
    unittest.main()
