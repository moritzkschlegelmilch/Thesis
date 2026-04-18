import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import pandas as pd
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from repo.discovery.component_deletion_impact import (
    build_component_deletion_highlight,
    calculate_component_deletion_impact,
    calculate_ocpn_component_deletion_impact,
    calculate_component_deletion_impact_footprint,
    discover_component_and_edge_ocpns,
)
from repo.discovery.discovery_preparation import (
    _compute_information_loss,
    _compute_simplicity_gain,
    discover_models_for_hierarchy,
)
from repo.discovery.subprocess_detection import (
    _build_component_colors,
    _arc_key,
    _place_key,
    detect_subprocess_components,
)
from repo.discovery.totem import clear_totem_cache
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


def _ocpn_activities(ocpn):
    if ocpn is None:
        return None
    return set(ocpn["activities"])


def _ocpn_object_types(ocpn):
    if ocpn is None:
        return None
    return set(ocpn["petri_nets"])


class _FakeInputOCEL:
    def __init__(self, object_to_type, events):
        self.object_types = sorted(set(object_to_type.values()))
        self.o2o_graph_edges = []
        self._object_to_type = dict(object_to_type)
        self._events = {
            event["event_id"]: {
                "activity": event["activity"],
                "timestamp": event["timestamp"],
                "event_objects": tuple(event["event_objects"]),
            }
            for event in events
        }
        self.events = pd.DataFrame({
            "_eventId": [event["event_id"] for event in events],
        })

    def get_event_activity(self, event_id):
        return self._events[event_id]["activity"]

    def get_event_timestamp(self, event_id):
        return self._events[event_id]["timestamp"]

    def get_value(self, event_id, key):
        if key != "event_objects":
            raise KeyError(key)
        return list(self._events[event_id]["event_objects"])

    def get_event_objects_by_type(self, event_id, object_type):
        return [
            obj
            for obj in self._events[event_id]["event_objects"]
            if self._object_to_type.get(obj) == object_type
        ]


def _build_hierarchy_test_ocel():
    object_to_type = {
        "item_1": "item",
        "item_2": "item",
        "order_1": "order",
        "order_2": "order",
    }
    events = [
        {"event_id": "e1", "activity": "start", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
        {"event_id": "e2", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1", "order_1"]},
        {"event_id": "e3", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1", "order_1"]},
        {"event_id": "e4", "activity": "end", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["item_1"]},
        {"event_id": "e5", "activity": "start", "timestamp": pd.Timestamp("2024-01-02T00:00:00"), "event_objects": ["item_2"]},
        {"event_id": "e6", "activity": "a", "timestamp": pd.Timestamp("2024-01-02T00:01:00"), "event_objects": ["item_2", "order_2"]},
        {"event_id": "e7", "activity": "b", "timestamp": pd.Timestamp("2024-01-02T00:02:00"), "event_objects": ["item_2", "order_2"]},
        {"event_id": "e8", "activity": "end", "timestamp": pd.Timestamp("2024-01-02T00:03:00"), "event_objects": ["item_2"]},
    ]
    return _FakeInputOCEL(object_to_type, events)


class SubprocessDetectionTests(unittest.TestCase):
    def tearDown(self):
        clear_totem_cache()

    def test_component_palette_is_unique(self):
        colors = _build_component_colors(24)
        self.assertEqual(len(colors), len(set(colors)))

    def test_calculate_component_deletion_impact_stops_at_outside_activity_boundaries(self):
        item_net = _build_net(
            "item",
            places=["in", "mid", "after", "sink"],
            transitions={
                "a": "a",
                "x": "x",
                "y": "y",
            },
            arcs=[
                ("in", "a"),
                ("a", "mid"),
                ("mid", "x"),
                ("x", "after"),
                ("after", "y"),
                ("y", "sink"),
            ],
        )

        impact = calculate_component_deletion_impact(item_net, {"a"})

        self.assertEqual(impact, {"x"})

    def test_calculate_component_deletion_impact_traverses_silent_transitions(self):
        item_net = _build_net(
            "item",
            places=["in", "mid_left", "mid_right", "out", "sink"],
            transitions={
                "a": "a",
                "tau": None,
                "b": "b",
                "x": "x",
            },
            arcs=[
                ("in", "a"),
                ("a", "mid_left"),
                ("mid_left", "tau"),
                ("tau", "mid_right"),
                ("mid_right", "b"),
                ("b", "out"),
                ("out", "x"),
                ("x", "sink"),
            ],
        )

        impact = calculate_component_deletion_impact(item_net, {"a", "b"})

        self.assertEqual(impact, {"x"})

    def test_calculate_component_deletion_impact_restarts_from_unexplored_component_labels(self):
        item_net = _build_net(
            "item",
            places=[
                "left_in",
                "left_out",
                "left_sink",
                "right_in",
                "right_out",
                "right_sink",
            ],
            transitions={
                "a": "a",
                "x": "x",
                "b": "b",
                "y": "y",
            },
            arcs=[
                ("left_in", "a"),
                ("a", "left_out"),
                ("left_out", "x"),
                ("x", "left_sink"),
                ("right_in", "b"),
                ("b", "right_out"),
                ("right_out", "y"),
                ("y", "right_sink"),
            ],
        )

        impact = calculate_component_deletion_impact(item_net, {"a", "b"})

        self.assertEqual(impact, {"x", "y"})

    def test_calculate_component_deletion_impact_footprint_contains_full_bfs_region(self):
        item_net = _build_net(
            "item",
            places=["in", "mid_left", "mid_right", "out", "sink"],
            transitions={
                "a": "a",
                "tau": None,
                "b": "b",
                "x": "x",
            },
            arcs=[
                ("in", "a"),
                ("a", "mid_left"),
                ("mid_left", "tau"),
                ("tau", "mid_right"),
                ("mid_right", "b"),
                ("b", "out"),
                ("out", "x"),
                ("x", "sink"),
            ],
        )

        footprint = calculate_component_deletion_impact_footprint(item_net, {"a", "b"})

        self.assertEqual(footprint["impact_labels"], {"x"})
        self.assertEqual(footprint["transition_labels"], {"a", "b", "x"})
        self.assertEqual(
            {transition.label for transition in footprint["transitions"] if transition.label is not None},
            {"a", "b", "x"},
        )
        self.assertEqual(
            {place.name for place in footprint["places"]},
            {"mid_left", "mid_right", "out"},
        )
        self.assertEqual(
            {
                (
                    getattr(arc.source, "name", None),
                    getattr(arc.target, "name", None),
                )
                for arc in footprint["arcs"]
            },
            {
                ("a", "mid_left"),
                ("mid_left", "tau"),
                ("tau", "mid_right"),
                ("mid_right", "b"),
                ("b", "out"),
                ("out", "x"),
            },
        )

    def test_calculate_component_deletion_impact_footprint_does_not_walk_backwards(self):
        item_net = _build_net(
            "item",
            places=["pre", "mid", "out", "sink"],
            transitions={
                "pre_x": "pre_x",
                "a": "a",
                "x": "x",
            },
            arcs=[
                ("pre", "pre_x"),
                ("pre_x", "mid"),
                ("mid", "a"),
                ("a", "out"),
                ("out", "x"),
                ("x", "sink"),
            ],
        )

        footprint = calculate_component_deletion_impact_footprint(item_net, {"a"})

        self.assertEqual(footprint["impact_labels"], {"x"})
        self.assertNotIn("pre_x", footprint["transition_labels"])
        self.assertNotIn("mid", {place.name for place in footprint["places"]})
        self.assertNotIn(
            ("pre_x", "mid"),
            {
                (
                    getattr(arc.source, "name", None),
                    getattr(arc.target, "name", None),
                )
                for arc in footprint["arcs"]
            },
        )

    def test_calculate_ocpn_component_deletion_impact_unions_relevant_object_types(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_mid", "item_out"],
            transitions={
                "a_item": "a",
                "b_item": "b",
                "x_item": "x",
            },
            arcs=[
                ("item_in", "a_item"),
                ("a_item", "item_mid"),
                ("item_mid", "b_item"),
                ("b_item", "item_out"),
                ("item_out", "x_item"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["order_in", "order_out", "order_sink"],
            transitions={
                "a_order": "a",
                "y_order": "y",
            },
            arcs=[
                ("order_in", "a_order"),
                ("a_order", "order_out"),
                ("order_out", "y_order"),
                ("y_order", "order_sink"),
            ],
        )

        ocpn = _build_ocpn({
            "item": item_net,
            "order": order_net,
        })

        self.assertEqual(
            calculate_ocpn_component_deletion_impact(ocpn, {"a", "b"}),
            {"x", "y"},
        )

    def test_discover_component_and_edge_ocpns_filters_expected_activities(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_mid", "item_out", "item_sink"],
            transitions={
                "a_item": "a",
                "b_item": "b",
                "x_item": "x",
                "end_item": "end_item",
            },
            arcs=[
                ("item_in", "a_item"),
                ("a_item", "item_mid"),
                ("item_mid", "b_item"),
                ("b_item", "item_out"),
                ("item_out", "x_item"),
                ("x_item", "item_sink"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["order_in", "order_out", "order_sink"],
            transitions={
                "a_order": "a",
                "y_order": "y",
            },
            arcs=[
                ("order_in", "a_order"),
                ("a_order", "order_out"),
                ("order_out", "y_order"),
                ("y_order", "order_sink"),
            ],
        )
        shipment_net = _build_net(
            "shipment",
            places=["shipment_in", "shipment_sink"],
            transitions={
                "z_shipment": "z",
            },
            arcs=[
                ("shipment_in", "z_shipment"),
                ("z_shipment", "shipment_sink"),
            ],
        )

        ocpn = _build_ocpn({
            "item": item_net,
            "order": order_net,
            "shipment": shipment_net,
        })
        ocel = _FakeInputOCEL(
            {
                "item_1": "item",
                "order_1": "order",
                "shipment_1": "shipment",
            },
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e2", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1"]},
                {"event_id": "e3", "activity": "x", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1"]},
                {"event_id": "e4", "activity": "y", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["order_1"]},
                {"event_id": "e5", "activity": "z", "timestamp": pd.Timestamp("2024-01-01T00:04:00"), "event_objects": ["shipment_1"]},
            ],
        )

        component_and_edge_ocpn, edge_only_ocpn = discover_component_and_edge_ocpns(
            ocel,
            ocpn,
            {"a", "b"},
        )

        self.assertEqual(_ocpn_activities(component_and_edge_ocpn), {"a", "b", "x", "y"})
        self.assertEqual(_ocpn_activities(edge_only_ocpn), {"x", "y"})

    def test_discover_component_and_edge_ocpns_returns_none_for_empty_edge_model(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_mid", "item_out"],
            transitions={
                "a_item": "a",
                "b_item": "b",
            },
            arcs=[
                ("item_in", "a_item"),
                ("a_item", "item_mid"),
                ("item_mid", "b_item"),
                ("b_item", "item_out"),
            ],
        )
        ocpn = _build_ocpn({"item": item_net})
        ocel = _FakeInputOCEL(
            {"item_1": "item"},
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
                {"event_id": "e2", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1"]},
            ],
        )

        component_and_edge_ocpn, edge_only_ocpn = discover_component_and_edge_ocpns(
            ocel,
            ocpn,
            {"a", "b"},
        )

        self.assertEqual(_ocpn_activities(component_and_edge_ocpn), {"a", "b"})
        self.assertIsNone(edge_only_ocpn)

    def test_discover_component_and_edge_ocpns_merges_lower_layer_events_and_relations(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_mid", "item_out", "item_sink"],
            transitions={
                "a_item": "a",
                "b_item": "b",
                "x_item": "x",
            },
            arcs=[
                ("item_in", "a_item"),
                ("a_item", "item_mid"),
                ("item_mid", "b_item"),
                ("b_item", "item_out"),
                ("item_out", "x_item"),
                ("x_item", "item_sink"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["order_in", "order_out", "order_sink"],
            transitions={
                "a_order": "a",
                "y_order": "y",
            },
            arcs=[
                ("order_in", "a_order"),
                ("a_order", "order_out"),
                ("order_out", "y_order"),
                ("y_order", "order_sink"),
            ],
        )

        ocpn = _build_ocpn({
            "item": item_net,
            "order": order_net,
        })
        upper_ocel = _FakeInputOCEL(
            {"item_1": "item"},
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
                {"event_id": "e2", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1"]},
                {"event_id": "e3", "activity": "x", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1"]},
            ],
        )
        lower_ocel = _FakeInputOCEL(
            {
                "order_1": "order",
                "order_2": "order",
            },
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["order_1"]},
                {"event_id": "e4", "activity": "y", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["order_1"]},
                {"event_id": "e5", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:04:00"), "event_objects": ["order_2"]},
            ],
        )

        component_and_edge_ocpn, edge_only_ocpn = discover_component_and_edge_ocpns(
            upper_ocel,
            ocpn,
            {"a", "b"},
            lower_layer_ocel=lower_ocel,
        )

        self.assertEqual(_ocpn_activities(component_and_edge_ocpn), {"a", "b", "x", "y"})
        self.assertEqual(_ocpn_object_types(component_and_edge_ocpn), {"item", "order"})
        self.assertEqual(_ocpn_activities(edge_only_ocpn), {"a", "b", "x", "y"})
        self.assertEqual(_ocpn_object_types(edge_only_ocpn), {"item", "order"})

    def test_compute_information_loss_uses_precision_ratio(self):
        fake_ocel = _FakeInputOCEL(
            {"item_1": "item"},
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
            ],
        )
        with_ocpn = {"petri_nets": {"item": (object(), object(), object())}}
        without_ocpn = {"petri_nets": {"item": (object(), object(), object())}}
        qualities = {}

        def net_quality_side_effect(ocpn, ocel):
            self.assertIs(ocel, fake_ocel)
            quality = unittest.mock.Mock()
            qualities[id(ocpn)] = quality
            if ocpn is with_ocpn:
                quality.precision.return_value = 0.8
                return quality
            if ocpn is without_ocpn:
                quality.precision.return_value = 0.4
                return quality
            raise AssertionError("Unexpected OCPN")

        with patch(
            "repo.discovery.component_deletion_impact.discover_component_and_edge_ocpns",
            return_value=(with_ocpn, without_ocpn),
        ), patch(
            "repo.discovery.component_deletion_impact._build_component_and_edge_ocels",
            return_value=(fake_ocel, fake_ocel),
        ), patch(
            "repo.discovery.net_quality.NetQuality",
            side_effect=net_quality_side_effect,
        ):
            information_loss = _compute_information_loss(
                current_model=object(),
                current_ocel=fake_ocel,
                lower_layer_ocel=None,
                component={"activities": ("a", "b")},
            )

        self.assertEqual(information_loss, 0.5)
        self.assertEqual(
            qualities[id(with_ocpn)].precision.call_args_list,
            [unittest.mock.call(show_progress=True)],
        )
        self.assertEqual(
            qualities[id(without_ocpn)].precision.call_args_list,
            [unittest.mock.call(show_progress=True)],
        )

    def test_compute_simplicity_gain_uses_complexity_ratio(self):
        with_ocpn = {"petri_nets": {"item": (object(), object(), object())}}
        without_ocpn = {"petri_nets": {"item": (object(), object(), object())}}

        def net_quality_side_effect(ocpn, ocel=None):
            quality = unittest.mock.Mock()
            if ocpn is with_ocpn:
                quality.complexity.return_value = 12.0
                return quality
            if ocpn is without_ocpn:
                quality.complexity.return_value = 3.0
                return quality
            raise AssertionError("Unexpected OCPN")

        with patch(
            "repo.discovery.component_deletion_impact.discover_component_and_edge_ocpns",
            return_value=(with_ocpn, without_ocpn),
        ), patch(
            "repo.discovery.net_quality.NetQuality",
            side_effect=net_quality_side_effect,
        ):
            simplicity_gain = _compute_simplicity_gain(
                current_model=object(),
                current_ocel=object(),
                lower_layer_ocel=None,
                component={"activities": ("a", "b")},
            )

        self.assertEqual(simplicity_gain, 0.75)

    def test_build_component_deletion_highlight_marks_union_across_object_types(self):
        item_net = _build_net(
            "item",
            places=["in", "mid", "out"],
            transitions={
                "a": "a",
                "b": "b",
                "x": "x",
            },
            arcs=[
                ("in", "a"),
                ("a", "mid"),
                ("mid", "b"),
                ("b", "out"),
                ("out", "x"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["in", "out", "sink"],
            transitions={
                "a": "a",
                "y": "y",
            },
            arcs=[
                ("in", "a"),
                ("a", "out"),
                ("out", "y"),
                ("y", "sink"),
            ],
        )

        ocpn = _build_ocpn({
            "item": item_net,
            "order": order_net,
        })
        highlight = build_component_deletion_highlight(
            ocpn,
            {"a", "b"},
            highlight_color="#abc123",
        )

        self.assertEqual(highlight["color"], "#abc123")
        self.assertEqual(highlight["fillcolor"], "#abc123")
        self.assertIn(("activity", "a"), highlight["transition_keys"])
        self.assertIn(("activity", "b"), highlight["transition_keys"])
        self.assertIn(("activity", "x"), highlight["transition_keys"])
        self.assertIn(("activity", "y"), highlight["transition_keys"])
        self.assertIn(
            _place_key("item", next(place for place in item_net.places if place.name == "mid")),
            highlight["place_keys"],
        )
        self.assertIn(
            _place_key("order", next(place for place in order_net.places if place.name == "out")),
            highlight["place_keys"],
        )
        self.assertIn(
            _arc_key("item", next(arc for arc in item_net.arcs if getattr(arc.source, "name", "") == "out")),
            highlight["arc_keys"],
        )
        self.assertIn(
            _arc_key("order", next(arc for arc in order_net.arcs if getattr(arc.source, "name", "") == "out")),
            highlight["arc_keys"],
        )

        graphviz = _build_ocpn_graphviz(
            ocpn,
            subprocess_components=[highlight],
        )

        self.assertIn('fillcolor="#abc123"', graphviz.source)
        self.assertIn('color="#abc123"', graphviz.source)

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

    def test_layer_discovery_prunes_component_with_default_scores(self):
        ocel = _build_hierarchy_test_ocel()
        _, discovered_models = discover_models_for_hierarchy(
            ocel,
            {"order": 1, "item": 2},
        )

        layer_two_model = discovered_models[2]

        self.assertEqual(layer_two_model["activities"], ["end", "start"])
        self.assertEqual(layer_two_model["subprocess_components"], [])

    def test_layer_discovery_recomputes_final_components_after_pruning(self):
        ocel = _build_hierarchy_test_ocel()

        def simplicity_gain(_, __, lower_layer_ocel, candidate):
            self.assertIsNotNone(lower_layer_ocel)
            self.assertEqual(
                sorted(set(lower_layer_ocel.events["ocel:activity"].tolist())),
                ["a", "b"],
            )
            return 2 if tuple(candidate["activities"]) == ("a", "b") else 0

        with patch(
            "repo.discovery.discovery_preparation._compute_simplicity_gain",
            side_effect=simplicity_gain,
        ):
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2},
            )

        layer_two_model = discovered_models[2]

        self.assertEqual(layer_two_model["activities"], ["end", "start"])
        self.assertEqual(layer_two_model["subprocess_components"], [])
        self.assertEqual(
            sorted(set(layer_two_model["ocel"].events["ocel:activity"].tolist())),
            ["end", "start"],
        )


if __name__ == "__main__":
    unittest.main()
