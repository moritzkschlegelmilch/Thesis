import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from repo.discovery.subprocess_detection import _build_component_colors, detect_subprocess_components
from repo.helpers.vorbose import _build_ocpn_graphviz


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

    return net


def _build_ocpn(nets_by_object_type):
    activities = sorted({
        transition.label
        for net in nets_by_object_type.values()
        for transition in net.transitions
        if transition.label is not None
    })

    return {
        "activities": activities,
        "petri_nets": {
            object_type: (net, Marking(), Marking())
            for object_type, net in nets_by_object_type.items()
        },
        "tbr_results": {},
        "double_arcs_on_activity": {
            object_type: {}
            for object_type in nets_by_object_type
        },
    }


def _activity_labels(component):
    return {
        transition["label"]
        for transition in component["transitions"]
        if transition["kind"] == "activity"
    }


def _component_activity_sets(components):
    return {
        frozenset(_activity_labels(component))
        for component in components
    }


class SubprocessDetectionTests(unittest.TestCase):
    def test_component_palette_is_unique(self):
        colors = _build_component_colors(24)
        self.assertEqual(len(colors), len(set(colors)))

    def test_detects_merged_multi_type_subprocess(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_mid", "item_out"],
            transitions={
                "start_item": "start_item",
                "pack_item": "pack",
                "ship_item": "ship",
                "end_item": "end_item",
            },
            arcs=[
                ("start_item", "item_in"),
                ("item_in", "pack_item"),
                ("pack_item", "item_mid"),
                ("item_mid", "ship_item"),
                ("ship_item", "item_out"),
                ("item_out", "end_item"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["order_in", "order_out"],
            transitions={
                "start_order": "start_order",
                "pack_order": "pack",
                "end_order": "end_order",
            },
            arcs=[
                ("start_order", "order_in"),
                ("order_in", "pack_order"),
                ("pack_order", "order_out"),
                ("order_out", "end_order"),
            ],
        )

        ocpn = _build_ocpn({
            "item": item_net,
            "order": order_net,
        })
        activity_to_layer = {
            "start_item": 2,
            "pack": 1,
            "ship": 1,
            "end_item": 2,
            "start_order": 2,
            "end_order": 2,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["object_types"], ["item", "order"])
        self.assertEqual(_activity_labels(components[0]), {"pack", "ship"})
        self.assertEqual(len(components[0]["place_keys"]), 5)
        self.assertEqual(len(components[0]["arc_keys"]), 6)

    def test_global_input_places_can_start_a_subprocess(self):
        item_net = _build_net(
            "item",
            places=["in", "mid", "out"],
            transitions={
                "a": "a",
                "b": "b",
                "end": "end",
            },
            arcs=[
                ("in", "a"),
                ("a", "mid"),
                ("mid", "b"),
                ("b", "out"),
                ("out", "end"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        activity_to_layer = {
            "a": 1,
            "b": 1,
            "end": 2,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        self.assertEqual(len(components), 1)
        self.assertEqual(_activity_labels(components[0]), {"a", "b"})

    def test_global_final_places_can_end_a_subprocess(self):
        item_net = _build_net(
            "item",
            places=["in", "mid", "out"],
            transitions={
                "start": "start",
                "a": "a",
                "b": "b",
            },
            arcs=[
                ("start", "in"),
                ("in", "a"),
                ("a", "mid"),
                ("mid", "b"),
                ("b", "out"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        activity_to_layer = {
            "start": 2,
            "a": 1,
            "b": 1,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        self.assertEqual(len(components), 1)
        self.assertEqual(_activity_labels(components[0]), {"a", "b"})

    def test_outside_only_alternative_path_keeps_component_valid(self):
        item_net = _build_net(
            "item",
            places=["in", "mid", "out", "ext"],
            transitions={
                "a": "a",
                "b": "b",
                "x": "x",
                "y": "y",
            },
            arcs=[
                ("in", "a"),
                ("a", "mid"),
                ("mid", "b"),
                ("b", "out"),
                ("in", "x"),
                ("x", "ext"),
                ("ext", "y"),
                ("y", "out"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        activity_to_layer = {
            "a": 1,
            "b": 1,
            "x": 2,
            "y": 2,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        self.assertIn(frozenset({"a", "b"}), _component_activity_sets(components))

    def test_mixed_path_between_input_and_output_invalidates_component(self):
        item_net = _build_net(
            "item",
            places=["in_1", "in_2", "mid", "out", "ext"],
            transitions={
                "a": "a",
                "b": "b",
                "d": "d",
                "x": "x",
                "y": "y",
            },
            arcs=[
                ("in_1", "a"),
                ("a", "mid"),
                ("mid", "b"),
                ("b", "out"),
                ("in_2", "d"),
                ("d", "out"),
                ("in_1", "x"),
                ("x", "ext"),
                ("ext", "y"),
                ("y", "in_2"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        activity_to_layer = {
            "a": 1,
            "b": 1,
            "d": 1,
            "x": 2,
            "y": 2,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        component_activity_sets = _component_activity_sets(components)
        self.assertNotIn(frozenset({"a", "b", "d"}), component_activity_sets)
        self.assertIn(frozenset({"a", "b"}), component_activity_sets)

    def test_disconnected_regions_are_not_combined(self):
        item_net = _build_net(
            "item",
            places=["left_in", "left_out", "right_in", "right_out"],
            transitions={
                "start_left": "start_left",
                "left": "left",
                "end_left": "end_left",
                "start_right": "start_right",
                "right": "right",
                "end_right": "end_right",
            },
            arcs=[
                ("start_left", "left_in"),
                ("left_in", "left"),
                ("left", "left_out"),
                ("left_out", "end_left"),
                ("start_right", "right_in"),
                ("right_in", "right"),
                ("right", "right_out"),
                ("right_out", "end_right"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        activity_to_layer = {
            "start_left": 2,
            "left": 1,
            "end_left": 2,
            "start_right": 2,
            "right": 1,
            "end_right": 2,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        self.assertEqual(components, [])

    def test_multiple_output_places_invalidate_component(self):
        item_net = _build_net(
            "item",
            places=["in", "mid_left", "mid_right", "out", "dead"],
            transitions={
                "start": "start",
                "a": "a",
                "bridge": "bridge",
                "c": "c",
                "end": "end",
            },
            arcs=[
                ("start", "in"),
                ("in", "a"),
                ("a", "mid_left"),
                ("mid_left", "bridge"),
                ("bridge", "mid_right"),
                ("mid_right", "c"),
                ("c", "out"),
                ("out", "end"),
                ("bridge", "dead"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        activity_to_layer = {
            "start": 2,
            "a": 1,
            "bridge": 1,
            "c": 1,
            "end": 2,
        }

        components = detect_subprocess_components(ocpn, activity_to_layer, reference_layer=2)

        self.assertEqual(components, [])

    def test_renderer_uses_subprocess_component_colors_for_debug_highlighting(self):
        item_net = _build_net(
            "item",
            places=["in", "mid", "out"],
            transitions={
                "start": "start",
                "a": "a",
                "b": "b",
                "end": "end",
            },
            arcs=[
                ("start", "in"),
                ("in", "a"),
                ("a", "mid"),
                ("mid", "b"),
                ("b", "out"),
                ("out", "end"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        components = detect_subprocess_components(
            ocpn,
            {
                "start": 2,
                "a": 1,
                "b": 1,
                "end": 2,
            },
            reference_layer=2,
        )
        self.assertEqual(len(components), 1)

        components[0]["color"] = "#123abc"
        graphviz = _build_ocpn_graphviz(
            ocpn,
            subprocess_components=components,
        )

        self.assertGreaterEqual(graphviz.source.count("#123abc"), 6)
        self.assertNotIn('fillcolor="#123abc"', graphviz.source)

    def test_renderer_preserves_original_colors_for_overlaps(self):
        item_net = _build_net(
            "item",
            places=["in", "out"],
            transitions={
                "start": "start",
                "a": "a",
                "end": "end",
            },
            arcs=[
                ("start", "in"),
                ("in", "a"),
                ("a", "out"),
                ("out", "end"),
            ],
        )

        ocpn = _build_ocpn({"item": item_net})
        subprocess_components = [
            {
                "color": "#123abc",
                "transition_keys": (("activity", "a"),),
                "place_keys": (),
                "arc_keys": (("arc", "item", next(arc for arc in item_net.arcs if getattr(arc.source, "name", "") == "in")),),
            },
            {
                "color": "#456def",
                "transition_keys": (("activity", "a"),),
                "place_keys": (),
                "arc_keys": (("arc", "item", next(arc for arc in item_net.arcs if getattr(arc.source, "name", "") == "in")),),
            },
        ]

        graphviz = _build_ocpn_graphviz(
            ocpn,
            subprocess_components=subprocess_components,
        )

        self.assertIn('color="#123abc:#456def"', graphviz.source)


if __name__ == "__main__":
    unittest.main()
