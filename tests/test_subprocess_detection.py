from contextlib import redirect_stdout
from io import StringIO
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import pandas as pd
from PIL import Image
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
    _build_precision_reference_bundle,
    _build_ocel_filtering_context,
    _build_pruning_candidates,
    _compute_information_loss,
    _compute_simplicity_gain,
    _select_best_pruning_candidate,
    _virtual_reduction_enabled_labels_by_context,
    clear_ocel_filtering_context_cache,
    discover_models_for_hierarchy,
)
from repo.discovery.region_detection import _make_local_region, _make_output_region
from repo.discovery.subprocess_detection import (
    _build_component_colors,
    _arc_key,
    _place_key,
    _transition_key,
    collapse_sub_processes,
    detect_subprocess_components,
)
from repo.discovery.totem import clear_totem_cache
from repo.helpers.vorbose import (
    _build_ocpn_graphviz,
    _render_indexed_subprocess_image_for_hierarchy_row,
    _render_model_image_for_hierarchy_row,
    render_pruning_candidate_debug,
    render_collapsed_sub_processes,
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


def _node_by_name(nodes, name):
    return next(node for node in nodes if getattr(node, "name", None) == name)


def _arc_by_endpoints(net, source_name, target_name):
    return next(
        arc
        for arc in net.arcs
        if getattr(arc.source, "name", None) == source_name
        and getattr(arc.target, "name", None) == target_name
    )


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


def _build_simple_debug_ocpn():
    net = _build_net(
        "item",
        places=["in", "out"],
        transitions={
            "a": "a",
        },
        arcs=[
            ("in", "a"),
            ("a", "out"),
        ],
    )
    return _build_ocpn({"item": net})


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
        clear_ocel_filtering_context_cache()

    def test_build_ocel_filtering_context_caches_by_ocel_identity(self):
        fake_ocel = object()
        object_to_type = {"item_1": "item"}
        event_records = [
            ("e1", "a", pd.Timestamp("2024-01-01T00:00:00"), ("item_1",)),
        ]
        o2o_edges = (("item_1", "item_1"),)

        with patch(
            "repo.discovery.discovery_preparation._extract_ocel_filtering_context",
            return_value=(object_to_type, event_records),
        ) as extract_patch, patch(
            "repo.discovery.discovery_preparation._iter_ocel_o2o_edges",
            return_value=o2o_edges,
        ) as edges_patch:
            first = _build_ocel_filtering_context(fake_ocel)
            second = _build_ocel_filtering_context(fake_ocel)
            clear_ocel_filtering_context_cache(fake_ocel)
            third = _build_ocel_filtering_context(fake_ocel)

        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(extract_patch.call_count, 2)
        self.assertEqual(edges_patch.call_count, 2)

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

    def test_discover_component_and_edge_ocpns_reuses_shared_model_builder(self):
        with_ocpn = {"petri_nets": {"item": (object(), object(), object())}}
        without_ocpn = {"petri_nets": {"item": (object(), object(), object())}}

        with patch(
            "repo.discovery.component_deletion_impact.discover_component_and_edge_models",
            return_value=(object(), object(), with_ocpn, without_ocpn),
        ) as discover_patch:
            component_and_edge_ocpn, edge_only_ocpn = discover_component_and_edge_ocpns(
                ocel=object(),
                ocpn=object(),
                component_activity_labels=("a", "b"),
                lower_layer_ocel=object(),
            )

        self.assertIs(component_and_edge_ocpn, with_ocpn)
        self.assertIs(edge_only_ocpn, without_ocpn)
        discover_patch.assert_called_once()

    def test_build_pruning_candidates_joins_boundary_activities_with_subprocess(self):
        net = _build_net(
            "item",
            places=["in", "mid", "out"],
            transitions={
                "start_t": "start",
                "a_t": "a",
                "b_t": "b",
                "end_t": "end",
            },
            arcs=[
                ("start_t", "in"),
                ("in", "a_t"),
                ("a_t", "mid"),
                ("mid", "b_t"),
                ("b_t", "out"),
                ("out", "end_t"),
            ],
        )
        place_in = _node_by_name(net.places, "in")
        place_mid = _node_by_name(net.places, "mid")
        place_out = _node_by_name(net.places, "out")
        component = {
            "id": "subprocess_1",
            "transitions": [
                {"kind": "activity", "label": "a"},
                {"kind": "activity", "label": "b"},
            ],
            "transition_keys": (("activity", "a"), ("activity", "b")),
            "place_keys": (
                ("place", "item", place_in),
                ("place", "item", place_mid),
                ("place", "item", place_out),
            ),
            "arc_keys": (
                ("arc", "item", _arc_by_endpoints(net, "in", "a_t")),
                ("arc", "item", _arc_by_endpoints(net, "a_t", "mid")),
                ("arc", "item", _arc_by_endpoints(net, "mid", "b_t")),
                ("arc", "item", _arc_by_endpoints(net, "b_t", "out")),
            ),
        }

        candidates = _build_pruning_candidates(
            [component],
            {"start", "a", "b", "end"},
            {"start": 1, "a": 1, "b": 1, "end": 1},
            reference_layer=2,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["kind"], "subprocess")
        self.assertEqual(set(candidates[0]["activities"]), {"start", "a", "b", "end"})
        self.assertEqual(candidates[0]["subprocess_activity_groups"], (("a", "b"),))

    def test_build_pruning_candidates_merges_subprocesses_sharing_boundary_activity(self):
        net = _build_net(
            "item",
            places=["in_1", "out_1", "in_2", "out_2"],
            transitions={
                "a_t": "a",
                "x_t": "x",
                "b_t": "b",
            },
            arcs=[
                ("in_1", "a_t"),
                ("a_t", "out_1"),
                ("out_1", "x_t"),
                ("x_t", "in_2"),
                ("in_2", "b_t"),
                ("b_t", "out_2"),
            ],
        )
        component_a = {
            "id": "subprocess_1",
            "transitions": [{"kind": "activity", "label": "a"}],
            "transition_keys": (("activity", "a"),),
            "place_keys": (
                ("place", "item", _node_by_name(net.places, "in_1")),
                ("place", "item", _node_by_name(net.places, "out_1")),
            ),
            "arc_keys": (
                ("arc", "item", _arc_by_endpoints(net, "in_1", "a_t")),
                ("arc", "item", _arc_by_endpoints(net, "a_t", "out_1")),
            ),
        }
        component_b = {
            "id": "subprocess_2",
            "transitions": [{"kind": "activity", "label": "b"}],
            "transition_keys": (("activity", "b"),),
            "place_keys": (
                ("place", "item", _node_by_name(net.places, "in_2")),
                ("place", "item", _node_by_name(net.places, "out_2")),
            ),
            "arc_keys": (
                ("arc", "item", _arc_by_endpoints(net, "in_2", "b_t")),
                ("arc", "item", _arc_by_endpoints(net, "b_t", "out_2")),
            ),
        }

        candidates = _build_pruning_candidates(
            [component_a, component_b],
            {"a", "x", "b"},
            {"a": 1, "x": 1, "b": 1},
            reference_layer=2,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(set(candidates[0]["activities"]), {"a", "x", "b"})
        self.assertEqual(
            candidates[0]["subprocess_activity_groups"],
            (("a",), ("b",)),
        )

    def test_compute_information_loss_delegates_to_prepared_context(self):
        prepared_context = {"prepared": True}

        with patch(
            "repo.discovery.discovery_preparation._prepare_pruning_candidate_context",
            return_value=prepared_context,
        ) as prepare_patch, patch(
            "repo.discovery.discovery_preparation._compute_information_loss_from_prepared_context",
            return_value=0.25,
        ) as compute_patch:
            information_loss = _compute_information_loss(
                current_model=object(),
                current_ocel=object(),
                lower_layer_ocel=None,
                component={"activities": ("a", "b")},
            )

        self.assertAlmostEqual(information_loss, 0.25)
        prepare_patch.assert_called_once_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            None,
            ("a", "b"),
            build_component_and_edge_ocel=True,
        )
        compute_patch.assert_called_once_with(prepared_context)

    def test_compute_information_loss_uses_fixed_precision_bundle(self):
        precision_bundle = {
            "baseline_enabled_mass": 5.0,
            "context_weights": {
                "ctx_a": 2,
                "ctx_b": 1,
            },
            "original_enabled_by_context": {
                "ctx_a": frozenset({"a", "b"}),
                "ctx_b": frozenset({"a"}),
            },
            "reduced_log_enabled_by_context": {
                "ctx_a": frozenset(),
                "ctx_b": frozenset({"y"}),
            },
        }

        with patch(
            "repo.discovery.discovery_preparation._virtual_reduction_enabled_labels_by_context",
            return_value={
                "ctx_a": frozenset({"b", "x"}),
                "ctx_b": frozenset({"y"}),
            },
        ) as reduced_patch:
            information_loss = _compute_information_loss(
                current_model=None,
                current_ocel=None,
                lower_layer_ocel=None,
                component={"activities": ("x", "y")},
                precision_bundle=precision_bundle,
            )

        self.assertAlmostEqual(information_loss, 0.4)
        reduced_patch.assert_called_once_with(
            precision_bundle,
            frozenset({"x", "y"}),
        )

    def test_virtual_reduction_enabled_labels_by_context_uses_cached_terminal_states(self):
        state_a = object()
        state_b = object()

        class _FakeQuality:
            def _state_key(self, state):
                return id(state)

        precision_bundle = {
            "quality": _FakeQuality(),
            "removed_object_types": frozenset({"item"}),
            "original_terminal_states_by_context": {
                "ctx_a": (state_a, state_b),
                "ctx_b": (state_b,),
            },
            "candidate_enabled_cache": {},
            "show_progress": False,
        }

        with patch(
            "repo.discovery.discovery_preparation._enabled_labels_with_virtual_reduction",
            side_effect=lambda quality, state, affected_labels, removed_object_types, cache: {
                state_a: frozenset({"x"}),
                state_b: frozenset({"y"}),
            }[state],
        ) as enabled_patch:
            reduced_enabled = _virtual_reduction_enabled_labels_by_context(
                precision_bundle,
                {"x", "y"},
            )
            cached_enabled = _virtual_reduction_enabled_labels_by_context(
                precision_bundle,
                {"y", "x"},
            )

        self.assertEqual(
            reduced_enabled,
            {
                "ctx_a": frozenset({"x", "y"}),
                "ctx_b": frozenset({"y"}),
            },
        )
        self.assertEqual(cached_enabled, reduced_enabled)
        self.assertEqual(enabled_patch.call_count, 3)

    def test_build_precision_reference_bundle_projects_model_to_two_layers(self):
        previous_net = _build_net(
            "order",
            places=["a_in", "a_out", "b_in", "b_out"],
            transitions={
                "a_t": "a",
                "b_t": "b",
            },
            arcs=[
                ("a_in", "a_t"),
                ("a_t", "a_out"),
                ("b_in", "b_t"),
                ("b_t", "b_out"),
            ],
        )
        current_net = _build_net(
            "item",
            places=["c_in", "c_out"],
            transitions={
                "c_t": "c",
            },
            arcs=[
                ("c_in", "c_t"),
                ("c_t", "c_out"),
            ],
        )
        previous_ocpn = _build_ocpn({"order": previous_net})
        current_ocpn = _build_ocpn({"item": current_net})
        previous_ocpn["petri_nets"]["order"] = (
            previous_net,
            Marking({
                _node_by_name(previous_net.places, "a_in"): 1,
                _node_by_name(previous_net.places, "b_in"): 1,
            }),
            Marking({
                _node_by_name(previous_net.places, "a_out"): 1,
                _node_by_name(previous_net.places, "b_out"): 1,
            }),
        )
        current_ocpn["petri_nets"]["item"] = (
            current_net,
            Marking({_node_by_name(current_net.places, "c_in"): 1}),
            Marking({_node_by_name(current_net.places, "c_out"): 1}),
        )

        merged_ocel = unittest.mock.Mock()
        merged_ocel.events = pd.DataFrame({"ocel:eid": ["e1"]})
        merged_ocel.relations = pd.DataFrame({"ocel:eid": ["e1"]})
        reduced_ocel = unittest.mock.Mock()
        reduced_ocel.events = pd.DataFrame({"ocel:eid": ["e1"]})
        reduced_ocel.relations = pd.DataFrame({"ocel:eid": ["e1"]})
        captured = {}

        class _FakeQuality:
            def __init__(self, ocpn, ocel, **kwargs):
                captured["ocpn"] = ocpn
                captured["kwargs"] = kwargs

            def _prepare_log(self, ocel, show_progress=False):
                return {
                    "events": ("e1",),
                    "ctx": {"e1": "ctx"},
                    "log": {"ctx": frozenset({"b"})},
                    "replay": {"ctx": [object()]},
                }

            def _replay_terminal_states(self, event):
                return ("terminal",)

            def _enabled_labels(self, state):
                return frozenset({"b"})

            def _state_key(self, state):
                return state

        with patch(
            "repo.discovery.discovery_preparation._build_projected_ocel",
            side_effect=[
                (merged_ocel, [], ["b", "c"]),
                (reduced_ocel, [], ["b", "c"]),
            ],
        ), patch(
            "repo.discovery.net_quality.NetQuality",
            _FakeQuality,
        ):
            precision_bundle = _build_precision_reference_bundle(
                source_ocel=object(),
                event_records=[],
                object_to_type={},
                activity_to_layer={"a": 1, "b": 2, "c": 3},
                reference_layer=3,
                removed_object_types={"item"},
                current_layer_ocpn=current_ocpn,
                previous_layer_ocpn=previous_ocpn,
            )

        self.assertIsNotNone(precision_bundle)
        self.assertIsNone(captured["kwargs"]["precision_context_depth"])
        self.assertEqual(set(captured["ocpn"]["activities"]), {"b", "c"})
        self.assertEqual(
            {
                transition.label
                for transition in captured["ocpn"]["petri_nets"]["order"][0].transitions
                if transition.label is not None
            },
            {"b"},
        )

    def test_build_precision_reference_bundle_samples_contexts_by_draw_count(self):
        previous_net = _build_net(
            "order",
            places=["a_in", "a_out"],
            transitions={"a_t": "a"},
            arcs=[
                ("a_in", "a_t"),
                ("a_t", "a_out"),
            ],
        )
        current_net = _build_net(
            "item",
            places=["b_in", "b_out"],
            transitions={"b_t": "b"},
            arcs=[
                ("b_in", "b_t"),
                ("b_t", "b_out"),
            ],
        )
        merged_ocel = unittest.mock.Mock()
        merged_ocel.events = pd.DataFrame({"ocel:eid": ["e1", "e2"]})
        merged_ocel.relations = pd.DataFrame({"ocel:eid": ["e1", "e2"]})
        reduced_ocel = unittest.mock.Mock()
        reduced_ocel.events = pd.DataFrame({"ocel:eid": ["e1", "e2"]})
        reduced_ocel.relations = pd.DataFrame({"ocel:eid": ["e1", "e2"]})
        replay_event_a = object()
        replay_event_b = object()

        class _FakeQuality:
            def __init__(self, ocpn, ocel, **kwargs):
                self.kwargs = kwargs

            def _prepare_log(self, ocel, show_progress=False):
                raise AssertionError("full precision log should not be prepared")

            def _prepare_sampled_precision_log(self, ocel, sample_size, show_progress=False):
                if ocel is merged_ocel:
                    return {
                        "events": ("e2",),
                        "ctx": {
                            "e2": "ctx_b",
                        },
                        "log": {
                            "ctx_b": frozenset({"b"}),
                        },
                        "context_weights": {
                            "ctx_b": 3,
                        },
                        "replay": {
                            "ctx_b": [replay_event_b],
                        },
                    }
                raise AssertionError("unexpected sampled OCEL")

            def _prepare_precision_log_for_event_counts(self, ocel, event_counts, show_progress=False):
                return {
                    "events": ("e2",),
                    "ctx": {
                        "e2": "reduced_b",
                    },
                    "log": {
                        "reduced_b": frozenset({"b"}),
                    },
                    "replay": {
                        "reduced_b": [object()],
                    },
                }

            def _replay_terminal_states(self, event):
                if event is replay_event_a:
                    return ("state_a",)
                return ("state_b",)

            def _enabled_labels(self, state):
                if state == "state_a":
                    return frozenset({"a"})
                return frozenset({"b", "c"})

            def _state_key(self, state):
                return state

        with patch(
            "repo.discovery.discovery_preparation._build_projected_ocel",
            side_effect=[
                (merged_ocel, [], ["a", "b"]),
                (reduced_ocel, [], ["a", "b"]),
            ],
        ), patch(
            "repo.discovery.net_quality.NetQuality",
            _FakeQuality,
        ):
            precision_bundle = _build_precision_reference_bundle(
                source_ocel=object(),
                event_records=[],
                object_to_type={},
                activity_to_layer={"a": 1, "b": 2},
                reference_layer=2,
                removed_object_types=set(),
                current_layer_ocpn=_build_ocpn({"item": current_net}),
                previous_layer_ocpn=_build_ocpn({"order": previous_net}),
                precision_context_sample_size=3,
                precision_context_depth=5,
                precision_context_sample_seed=11,
            )

        self.assertEqual(precision_bundle["context_weights"], {"ctx_b": 3})
        self.assertEqual(precision_bundle["sampled_contexts"], ("ctx_b",))
        self.assertEqual(
            precision_bundle["original_terminal_states_by_context"],
            {"ctx_b": ("state_b",)},
        )
        self.assertEqual(
            precision_bundle["original_enabled_by_context"],
            {"ctx_b": frozenset({"b", "c"})},
        )
        self.assertEqual(
            precision_bundle["reduced_log_enabled_by_context"],
            {"ctx_b": frozenset({"b"})},
        )
        self.assertAlmostEqual(precision_bundle["precision"], 0.5)

    def test_build_precision_reference_bundle_emits_progress_details(self):
        previous_net = _build_net(
            "order",
            places=["a_in", "a_out"],
            transitions={"a_t": "a"},
            arcs=[
                ("a_in", "a_t"),
                ("a_t", "a_out"),
            ],
        )
        current_net = _build_net(
            "item",
            places=["b_in", "b_out"],
            transitions={"b_t": "b"},
            arcs=[
                ("b_in", "b_t"),
                ("b_t", "b_out"),
            ],
        )
        merged_ocel = unittest.mock.Mock()
        merged_ocel.events = pd.DataFrame({"ocel:eid": ["e1"]})
        merged_ocel.relations = pd.DataFrame({"ocel:eid": ["e1"]})
        reduced_ocel = unittest.mock.Mock()
        reduced_ocel.events = pd.DataFrame({"ocel:eid": ["e1"]})
        reduced_ocel.relations = pd.DataFrame({"ocel:eid": ["e1"]})
        prepare_progress_flags = []

        class _FakeQuality:
            def __init__(self, ocpn, ocel, **kwargs):
                pass

            def _prepare_log(self, ocel, show_progress=False):
                raise AssertionError("full precision log should not be prepared")

            def _prepare_sampled_precision_log(self, ocel, sample_size, show_progress=False):
                prepare_progress_flags.append(show_progress)
                return {
                    "events": ("e1",),
                    "ctx": {"e1": "ctx"},
                    "log": {"ctx": frozenset({"b"})},
                    "context_weights": {"ctx": 1},
                    "replay": {"ctx": [object()]},
                }

            def _prepare_precision_log_for_event_counts(self, ocel, event_counts, show_progress=False):
                prepare_progress_flags.append(show_progress)
                return {
                    "events": ("e1",),
                    "ctx": {"e1": "ctx"},
                    "log": {"ctx": frozenset({"b"})},
                    "replay": {"ctx": [object()]},
                }

            def _replay_terminal_states(self, event):
                return ("terminal",)

            def _enabled_labels(self, state):
                return frozenset({"b"})

            def _state_key(self, state):
                return state

        buffer = StringIO()
        with patch(
            "repo.discovery.discovery_preparation._build_projected_ocel",
            side_effect=[
                (merged_ocel, [], ["a", "b"]),
                (reduced_ocel, [], ["a", "b"]),
            ],
        ), patch(
            "repo.discovery.net_quality.NetQuality",
            _FakeQuality,
        ), redirect_stdout(buffer):
            _build_precision_reference_bundle(
                source_ocel=object(),
                event_records=[],
                object_to_type={},
                activity_to_layer={"a": 1, "b": 2},
                reference_layer=2,
                removed_object_types=set(),
                current_layer_ocpn=_build_ocpn({"item": current_net}),
                previous_layer_ocpn=_build_ocpn({"order": previous_net}),
                precision_context_sample_size=1,
                precision_context_depth=5,
                precision_context_sample_seed=11,
                show_progress=True,
            )

        output = buffer.getvalue()
        self.assertIn("Precision reference sampling:", output)
        self.assertIn("Replaying precision reference contexts", output)
        self.assertEqual(prepare_progress_flags, [True, True])

    def test_compute_simplicity_gain_prunes_activity_and_cleanup_chain(self):
        net = _build_net(
            "item",
            places=["source", "mid", "sink"],
            transitions={
                "a_t": "a",
                "tau": None,
            },
            arcs=[
                ("source", "a_t"),
                ("a_t", "mid"),
                ("mid", "tau"),
                ("tau", "sink"),
            ],
        )
        ocpn = _build_ocpn({"item": net})
        ocpn["petri_nets"]["item"] = (
            net,
            Marking({_node_by_name(net.places, "source"): 1}),
            Marking({_node_by_name(net.places, "sink"): 1}),
        )

        simplicity_gain = _compute_simplicity_gain(
            current_model=ocpn,
            current_ocel=None,
            lower_layer_ocel=None,
            component={"activities": ("a",)},
        )

        self.assertAlmostEqual(simplicity_gain, 0.8)

    def test_compute_simplicity_gain_collapses_subprocess_candidates_before_pruning(self):
        net = _build_net(
            "item",
            places=["in", "mid", "out"],
            transitions={
                "a_t": "a",
                "b_t": "b",
            },
            arcs=[
                ("in", "a_t"),
                ("a_t", "mid"),
                ("mid", "b_t"),
                ("b_t", "out"),
            ],
        )
        ocpn = _build_ocpn({"item": net})
        ocpn["petri_nets"]["item"] = (
            net,
            Marking({_node_by_name(net.places, "in"): 1}),
            Marking({_node_by_name(net.places, "out"): 1}),
        )
        subprocess_components = detect_subprocess_components(
            ocpn,
            {"a": 1, "b": 1},
            reference_layer=2,
        )

        self.assertEqual(len(subprocess_components), 1)

        simplicity_gain = _compute_simplicity_gain(
            current_model=ocpn,
            current_ocel=None,
            lower_layer_ocel=None,
            component={
                "kind": "subprocess",
                "activities": ("a", "b"),
                "subprocess_components": tuple(subprocess_components),
                "subprocess_activity_groups": (("a", "b"),),
            },
        )

        self.assertAlmostEqual(simplicity_gain, 1 / 3)

    def test_select_best_pruning_candidate_greedily_expands_single_best_positive_candidate(self):
        candidates = [
            {"id": "a", "activities": ("a",)},
            {"id": "b", "activities": ("b",)},
            {"id": "c", "activities": ("c",)},
        ]
        scores = {
            ("a",): 0.6,
            ("b",): 0.4,
            ("c",): 0.1,
            ("a", "b"): 0.2,
            ("a", "c"): 0.5,
            ("a", "b", "c"): -0.1,
        }

        with patch(
            "repo.discovery.discovery_preparation._compute_simplicity_gain",
            return_value=0.0,
        ) as simplicity_patch, patch(
            "repo.discovery.discovery_preparation._compute_information_loss",
            side_effect=lambda *args: scores[tuple(args[3]["activities"])],
        ) as information_patch, patch(
            "repo.discovery.discovery_preparation._debug_pruning_candidate",
        ):
            best_candidate, best_score = _select_best_pruning_candidate(
                current_model=_build_simple_debug_ocpn(),
                current_ocel=object(),
                lower_layer_ocel=None,
                candidates=candidates,
            )

        self.assertEqual(best_candidate["activities"], ("a", "c"))
        self.assertAlmostEqual(best_score, 0.5)
        self.assertEqual(simplicity_patch.call_count, 6)
        self.assertEqual(information_patch.call_count, 6)

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

    def test_collapse_sub_processes_replaces_single_type_component_with_boundary_transition(self):
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

        collapsed_ocpn, inserted_transitions = collapse_sub_processes(ocpn, components)

        self.assertEqual(len(inserted_transitions), 1)
        object_type, collapsed_transition = next(iter(inserted_transitions))
        self.assertEqual(object_type, "item")
        self.assertEqual(collapsed_transition.label, "subprocess_1")

        collapsed_net, _, _ = collapsed_ocpn["petri_nets"]["item"]
        self.assertEqual(
            {place.name for place in collapsed_net.places},
            {"in", "out"},
        )
        self.assertEqual(
            {transition.label for transition in collapsed_net.transitions},
            {"start", "subprocess_1", "end"},
        )
        self.assertEqual(
            {arc.source.name for arc in collapsed_transition.in_arcs},
            {"in"},
        )
        self.assertEqual(
            {arc.target.name for arc in collapsed_transition.out_arcs},
            {"out"},
        )
        self.assertEqual(
            {transition.label for transition in item_net.transitions},
            {"start", "a", "b", "end"},
        )

    def test_collapse_sub_processes_replaces_multi_type_component_in_each_net(self):
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
        components = detect_subprocess_components(
            ocpn,
            {
                "start_item": 2,
                "pack": 1,
                "ship": 1,
                "end_item": 2,
                "start_order": 2,
                "end_order": 2,
            },
            reference_layer=2,
        )

        collapsed_ocpn, inserted_transitions = collapse_sub_processes(ocpn, components)

        inserted_by_type = {
            object_type: transition
            for object_type, transition in inserted_transitions
        }
        self.assertEqual(set(inserted_by_type), {"item", "order"})
        self.assertEqual(
            {transition.label for transition in inserted_by_type.values()},
            {"subprocess_1"},
        )
        self.assertEqual(
            set(collapsed_ocpn["activities"]),
            {"start_item", "start_order", "subprocess_1", "end_item", "end_order"},
        )

        collapsed_item_net, _, _ = collapsed_ocpn["petri_nets"]["item"]
        self.assertEqual(
            {place.name for place in collapsed_item_net.places},
            {"item_in", "item_out"},
        )
        self.assertEqual(
            {transition.label for transition in collapsed_item_net.transitions},
            {"start_item", "subprocess_1", "end_item"},
        )
        self.assertEqual(
            {arc.source.name for arc in inserted_by_type["item"].in_arcs},
            {"item_in"},
        )
        self.assertEqual(
            {arc.target.name for arc in inserted_by_type["item"].out_arcs},
            {"item_out"},
        )

        collapsed_order_net, _, _ = collapsed_ocpn["petri_nets"]["order"]
        self.assertEqual(
            {place.name for place in collapsed_order_net.places},
            {"order_in", "order_out"},
        )
        self.assertEqual(
            {transition.label for transition in collapsed_order_net.transitions},
            {"start_order", "subprocess_1", "end_order"},
        )
        self.assertEqual(
            {arc.source.name for arc in inserted_by_type["order"].in_arcs},
            {"order_in"},
        )
        self.assertEqual(
            {arc.target.name for arc in inserted_by_type["order"].out_arcs},
            {"order_out"},
        )

    def test_collapse_sub_processes_keeps_shared_input_place_for_multiple_regions(self):
        item_net = _build_net(
            "item",
            places=["p0", "p1", "p2"],
            transitions={
                "a": "a",
                "b": "b",
            },
            arcs=[
                ("p0", "a"),
                ("a", "p1"),
                ("p0", "b"),
                ("b", "p2"),
            ],
        )
        ocpn = _build_ocpn({"item": item_net})

        place_by_name = {place.name: place for place in item_net.places}
        transition_by_name = {transition.name: transition for transition in item_net.transitions}
        region_a = _make_output_region({
            _make_local_region(
                "item",
                place_by_name["p0"],
                {transition_by_name["a"]},
                place_by_name["p1"],
            )
        })
        region_b = _make_output_region({
            _make_local_region(
                "item",
                place_by_name["p0"],
                {transition_by_name["b"]},
                place_by_name["p2"],
            )
        })
        region_a["id"] = "subprocess_1"
        region_b["id"] = "subprocess_2"

        collapsed_ocpn, inserted_transitions = collapse_sub_processes(
            ocpn,
            [region_a, region_b],
        )

        collapsed_item_net, _, _ = collapsed_ocpn["petri_nets"]["item"]
        collapsed_places = {place.name: place for place in collapsed_item_net.places}
        self.assertIn("p0", collapsed_places)
        self.assertEqual(
            {arc.target.label for arc in collapsed_places["p0"].out_arcs},
            {"subprocess_1", "subprocess_2"},
        )
        self.assertEqual(
            {transition.label for _, transition in inserted_transitions},
            {"subprocess_1", "subprocess_2"},
        )

    def test_render_collapsed_sub_processes_highlights_inserted_transition(self):
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

        collapsed_ocpn, inserted_transitions = collapse_sub_processes(ocpn, components)
        highlight_component = {
            "id": "collapsed_sub_processes",
            "color": "#12ab34",
            "transition_keys": frozenset(
                _transition_key(object_type, transition)
                for object_type, transition in inserted_transitions
            ),
            "place_keys": frozenset(),
            "arc_keys": frozenset(),
        }

        graphviz = _build_ocpn_graphviz(
            collapsed_ocpn,
            subprocess_components=[highlight_component],
        )

        self.assertIn("#12ab34", graphviz.source)
        self.assertIn("subprocess_1", graphviz.source)

        image = render_collapsed_sub_processes(
            ocpn,
            components,
            highlight_color="#12ab34",
            max_size=(400, 200),
        )

        self.assertIsNotNone(image)
        self.assertLessEqual(image.width, 400)
        self.assertLessEqual(image.height, 200)

    def test_render_collapsed_sub_processes_marks_inserted_transition_with_plus_icon(self):
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

        collapsed_ocpn, inserted_transitions = collapse_sub_processes(ocpn, components)
        graphviz = _build_ocpn_graphviz(
            collapsed_ocpn,
            subprocess_components=[{
                "id": "collapsed_sub_processes",
                "color": "#12ab34",
                "fillcolor": "#12ab34",
                "marker": "+",
                "transition_keys": frozenset(
                    _transition_key(object_type, transition)
                    for object_type, transition in inserted_transitions
                ),
                "place_keys": frozenset(),
                "arc_keys": frozenset(),
            }],
        )

        self.assertIn('fillcolor="#12ab34"', graphviz.source)
        self.assertIn("subprocess_1", graphviz.source)
        self.assertIn(">+</FONT>", graphviz.source)

    def test_hierarchy_row_rendering_uses_collapsed_subprocess_rendering(self):
        model_image = Image.new("RGBA", (240, 120), "white")
        model_data = {
            "ocpn": {"petri_nets": {"item": (object(), object(), object())}},
            "subprocess_components": [{"id": "subprocess_1"}],
            "activity_resources": {},
            "highlighted_activities": [],
        }

        with patch(
            "repo.helpers.vorbose.render_collapsed_sub_processes",
            return_value=model_image,
        ) as collapsed_render_patch, patch(
            "repo.helpers.vorbose._render_ocpn_image",
            return_value=model_image,
        ) as regular_render_patch:
            image = _render_model_image_for_hierarchy_row(
                model_data,
                {"item": "#123abc"},
            )

        self.assertIsNotNone(image)
        collapsed_render_patch.assert_called_once_with(
            model_data["ocpn"],
            model_data["subprocess_components"],
            max_size=(1400, 700),
            object_type_colors={"item": "#123abc"},
            activity_resource_types={},
            highlighted_activities=[],
        )
        regular_render_patch.assert_not_called()

    def test_collapsed_rendering_preserves_highlighted_activities_inputs(self):
        model_image = Image.new("RGBA", (240, 120), "white")
        model_data = {
            "ocpn": {"petri_nets": {"item": (object(), object(), object())}},
            "subprocess_components": [{"id": "subprocess_1"}],
            "activity_resources": {"a": ["item"]},
            "highlighted_activities": ["a"],
        }

        with patch(
            "repo.helpers.vorbose.render_collapsed_sub_processes",
            return_value=model_image,
        ) as collapsed_render_patch:
            image = _render_model_image_for_hierarchy_row(
                model_data,
                {"item": "#123abc"},
            )

        self.assertIs(image, model_image)
        collapsed_render_patch.assert_called_once_with(
            model_data["ocpn"],
            model_data["subprocess_components"],
            max_size=(1400, 700),
            object_type_colors={"item": "#123abc"},
            activity_resource_types={"a": ["item"]},
            highlighted_activities=["a"],
        )

    def test_indexed_subprocess_rendering_uses_global_indices_as_markers(self):
        model_image = Image.new("RGBA", (240, 120), "white")
        model_data = {
            "ocpn": {"petri_nets": {"item": (object(), object(), object())}},
            "subprocess_components": [
                {
                    "id": "subprocess_1",
                    "index": 7,
                    "global_id": "subprocess_7",
                    "transition_keys": frozenset({("activity", "a")}),
                    "place_keys": frozenset(),
                    "arc_keys": frozenset(),
                },
            ],
            "activity_resources": {"a": ["item"]},
            "highlighted_activities": ["a"],
        }

        with patch(
            "repo.helpers.vorbose._render_ocpn_image",
            return_value=model_image,
        ) as render_patch:
            image = _render_indexed_subprocess_image_for_hierarchy_row(
                model_data,
                {"item": "#123abc"},
            )

        self.assertIs(image, model_image)
        subprocess_components = render_patch.call_args.kwargs["subprocess_components"]
        self.assertEqual(len(subprocess_components), 1)
        self.assertEqual(subprocess_components[0]["id"], "subprocess_7")
        self.assertEqual(subprocess_components[0]["marker"], "7")
        self.assertEqual(
            subprocess_components[0]["transition_keys"],
            frozenset({("activity", "a")}),
        )

    def test_process_area_discovery_visualize_indexed_subprocesses_uses_helper(self):
        fake_pulp = ModuleType("pulp")
        for attribute in (
            "LpProblem",
            "LpVariable",
            "LpInteger",
            "value",
            "LpMinimize",
            "lpSum",
            "LpStatus",
            "PULP_CBC_CMD",
        ):
            setattr(fake_pulp, attribute, object())

        with patch.dict(sys.modules, {"pulp": fake_pulp}):
            from repo.discovery.discovery import ProcessAreaDiscoveryFramework

            discovery = ProcessAreaDiscoveryFramework(None, [])
            discovery.solution = {"item": 1}
            discovery.discovered_models = {1: {"subprocess_components": []}}

            with patch(
                "repo.discovery.discovery.visualize_hierarchy_with_indexed_subprocesses",
                return_value="visualized",
            ) as visualize_patch:
                result = discovery.visualize_indexed_subprocesses(
                    title="Indexed",
                    output_path="indexed.png",
                )

        self.assertEqual(result, "visualized")
        visualize_patch.assert_called_once_with(
            discovery.solution,
            discovery.discovered_models,
            title="Indexed",
            output_path="indexed.png",
        )

    def test_render_pruning_candidate_debug_returns_side_by_side_image(self):
        with patch("repo.helpers.vorbose.plt.figure") as figure_patch, patch(
            "repo.helpers.vorbose.plt.imshow"
        ) as imshow_patch, patch("repo.helpers.vorbose.plt.axis") as axis_patch, patch(
            "repo.helpers.vorbose.plt.tight_layout"
        ) as tight_layout_patch, patch("repo.helpers.vorbose.plt.show") as show_patch, patch(
            "repo.helpers.vorbose.plt.close"
        ) as close_patch:
            image = render_pruning_candidate_debug(
                _build_simple_debug_ocpn(),
                _build_simple_debug_ocpn(),
                title="Candidate debug",
                with_component_metrics={"Complexity": 10.0, "Precision": 0.8},
                without_component_metrics={"Complexity": 4.0, "Precision": 0.5},
                summary_metrics={"Simplicity gain": 0.6, "Precision loss": 0.3},
                show=True,
            )

        self.assertIsNotNone(image)
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)
        figure_patch.assert_called_once()
        imshow_patch.assert_called_once()
        axis_patch.assert_called_once_with("off")
        tight_layout_patch.assert_called_once()
        show_patch.assert_called_once()
        close_patch.assert_called_once()

    def test_select_best_pruning_candidate_skips_debug_output_by_default(self):
        candidates = [
            {"id": "a", "activities": ("a",)},
            {"id": "b", "activities": ("b",)},
        ]
        scores = {
            ("a",): 0.5,
            ("b",): 0.3,
            ("a", "b"): 0.1,
        }

        with patch(
            "repo.discovery.discovery_preparation._compute_simplicity_gain",
            return_value=0.0,
        ) as simplicity_patch, patch(
            "repo.discovery.discovery_preparation._compute_information_loss",
            side_effect=lambda *args: scores[tuple(args[3]["activities"])],
        ) as information_patch, patch(
            "repo.discovery.discovery_preparation._debug_pruning_candidate",
        ) as debug_patch:
            best_candidate, best_score = _select_best_pruning_candidate(
                current_model=_build_simple_debug_ocpn(),
                current_ocel=object(),
                lower_layer_ocel=None,
                candidates=candidates,
            )

        self.assertEqual(best_candidate["activities"], ("a", "b"))
        self.assertAlmostEqual(best_score, 0.1)
        self.assertEqual(simplicity_patch.call_count, 3)
        self.assertEqual(information_patch.call_count, 3)
        self.assertEqual(debug_patch.call_count, 0)

    def test_select_best_pruning_candidate_returns_none_when_no_positive_candidate_exists(self):
        with patch(
            "repo.discovery.discovery_preparation._compute_simplicity_gain",
            return_value=0.2,
        ) as simplicity_patch, patch(
            "repo.discovery.discovery_preparation._compute_information_loss",
            return_value=0.0,
        ) as information_patch, patch(
            "repo.discovery.discovery_preparation._debug_pruning_candidate",
        ) as debug_patch:
            best_candidate, best_score = _select_best_pruning_candidate(
                current_model=_build_simple_debug_ocpn(),
                current_ocel=object(),
                lower_layer_ocel=None,
                candidates=[{"id": "a", "activities": ("a",)}],
            )

        self.assertIsNone(best_candidate)
        self.assertEqual(best_score, 0.0)
        self.assertEqual(simplicity_patch.call_count, 1)
        self.assertEqual(information_patch.call_count, 1)
        self.assertEqual(debug_patch.call_count, 0)

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

    def test_layer_discovery_keeps_only_native_and_detected_subprocess_activities(self):
        ocel = _FakeInputOCEL(
            {
                "item_1": "item",
                "order_1": "order",
            },
            [
                {"event_id": "e1", "activity": "start", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
                {"event_id": "e2", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e3", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e4", "activity": "x", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e5", "activity": "end", "timestamp": pd.Timestamp("2024-01-01T00:04:00"), "event_objects": ["item_1"]},
            ],
        )
        layer_one_ocel = unittest.mock.Mock()
        layer_one_ocel.events = pd.DataFrame({"ocel:activity": ["a", "b", "x"]})
        layer_two_first_ocel = unittest.mock.Mock()
        layer_two_first_ocel.events = pd.DataFrame({"ocel:activity": ["start", "a", "b", "x", "end"]})
        layer_two_second_ocel = unittest.mock.Mock()
        layer_two_second_ocel.events = pd.DataFrame({"ocel:activity": ["start", "a", "b", "end"]})
        subprocess_component = {
            "id": "subprocess_1",
            "transitions": [
                {"kind": "activity", "label": "a"},
                {"kind": "activity", "label": "b"},
            ],
        }
        build_layer_calls = []

        def build_layer_ocel_side_effect(_, __, ___, selected_object_types, selected_activities):
            build_layer_calls.append((set(selected_object_types), set(selected_activities)))
            if set(selected_object_types) == {"order"}:
                return layer_one_ocel, [], ["a", "b", "x"]
            if len(build_layer_calls) == 2:
                return layer_two_first_ocel, [], ["a", "b", "end", "start", "x"]
            return layer_two_second_ocel, [], ["a", "b", "end", "start"]

        def discover_with_components_side_effect(layer_ocel, *_):
            if layer_ocel is layer_one_ocel:
                return {"activities": ["a", "b", "x"], "petri_nets": {}}, []
            if layer_ocel is layer_two_first_ocel:
                return {"activities": ["a", "b", "end", "start", "x"], "petri_nets": {}}, [subprocess_component]
            if layer_ocel is layer_two_second_ocel:
                return {"activities": ["a", "b", "end", "start"], "petri_nets": {}}, [subprocess_component]
            raise AssertionError("Unexpected layer OCEL")

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            side_effect=build_layer_ocel_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            side_effect=discover_with_components_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            side_effect=[
                [],
                [{"kind": "subprocess", "activities": ("a", "b")}],
            ],
        ), patch(
            "repo.discovery.discovery_preparation._select_best_pruning_candidate",
            return_value=({"activities": ("a", "b")}, 0.1),
        ), patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ):
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2},
            )

        layer_two_model = discovered_models[2]

        self.assertEqual(build_layer_calls[1][1], {"a", "b", "end", "start", "x"})
        self.assertEqual(build_layer_calls[2][1], {"a", "b", "end", "start"})
        self.assertEqual(layer_two_model["activities"], ["a", "b", "end", "start"])
        self.assertEqual(
            _component_activity_sets(layer_two_model["subprocess_components"]),
            {frozenset({"a", "b"})},
        )

    def test_layer_discovery_uses_metric_pruning_for_higher_layers(self):
        ocel = _FakeInputOCEL(
            {
                "item_1": "item",
                "order_1": "order",
            },
            [
                {"event_id": "e1", "activity": "start", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
                {"event_id": "e2", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e3", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e4", "activity": "end", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["item_1"]},
            ],
        )
        layer_one_ocel = unittest.mock.Mock()
        layer_one_ocel.events = pd.DataFrame({"ocel:activity": ["a", "b"]})
        layer_two_ocel = unittest.mock.Mock()
        layer_two_ocel.events = pd.DataFrame({"ocel:activity": ["start", "a", "b", "end"]})
        subprocess_component = {
            "id": "subprocess_1",
            "transitions": [
                {"kind": "activity", "label": "a"},
                {"kind": "activity", "label": "b"},
            ],
        }

        def build_layer_ocel_side_effect(_, __, ___, selected_object_types, ____):
            if set(selected_object_types) == {"order"}:
                return layer_one_ocel, [], ["a", "b"]
            return layer_two_ocel, [], ["a", "b", "end", "start"]

        def discover_with_components_side_effect(layer_ocel, *_):
            if layer_ocel is layer_one_ocel:
                return {"activities": ["a", "b"], "petri_nets": {}}, []
            if layer_ocel is layer_two_ocel:
                return {"activities": ["a", "b", "end", "start"], "petri_nets": {}}, [subprocess_component]
            raise AssertionError("Unexpected layer OCEL")

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            side_effect=build_layer_ocel_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            side_effect=discover_with_components_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            side_effect=[[], [{"kind": "subprocess", "activities": ("a", "b")}]],
        ), patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ), patch(
            "repo.discovery.discovery_preparation._select_best_pruning_candidate",
            return_value=({"activities": ("a", "b")}, 0.2),
        ) as select_patch:
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2},
            )

        self.assertEqual(discovered_models[2]["activities"], ["a", "b", "end", "start"])
        select_patch.assert_called_once()

    def test_layer_discovery_precomputes_precision_reference_for_higher_layers(self):
        ocel = _FakeInputOCEL(
            {
                "item_1": "item",
                "order_1": "order",
            },
            [
                {"event_id": "e1", "activity": "start", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
                {"event_id": "e2", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e3", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e4", "activity": "end", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["item_1"]},
            ],
        )
        layer_one_ocel = unittest.mock.Mock()
        layer_one_ocel.events = pd.DataFrame({"ocel:activity": ["a", "b"]})
        layer_two_ocel = unittest.mock.Mock()
        layer_two_ocel.events = pd.DataFrame({"ocel:activity": ["start", "a", "b", "end"]})
        precision_reference = {"precision": 0.75}

        def build_layer_ocel_side_effect(_, __, ___, selected_object_types, ____):
            if set(selected_object_types) == {"order"}:
                return layer_one_ocel, [], ["a", "b"]
            return layer_two_ocel, [], ["a", "b", "end", "start"]

        def discover_with_components_side_effect(layer_ocel, *_):
            if layer_ocel is layer_one_ocel:
                return {"activities": ["a", "b"], "petri_nets": {"order": (object(), object(), object())}}, []
            if layer_ocel is layer_two_ocel:
                return {"activities": ["a", "b", "end", "start"], "petri_nets": {"item": (object(), object(), object())}}, []
            raise AssertionError("Unexpected layer OCEL")

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            side_effect=build_layer_ocel_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            side_effect=discover_with_components_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            side_effect=[[], [{"kind": "activity", "activities": ("a",)}]],
        ), patch(
            "repo.discovery.discovery_preparation._build_precision_reference_bundle",
            return_value=precision_reference,
        ) as precision_patch, patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ), patch(
            "repo.discovery.discovery_preparation._select_best_pruning_candidate",
            return_value=({"activities": ("a",)}, 0.2),
        ) as select_patch:
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2},
            )

        precision_patch.assert_called_once()
        self.assertIs(
            select_patch.call_args.kwargs["precision_bundle"],
            precision_reference,
        )
        self.assertEqual(select_patch.call_args.kwargs["precision_bundle"], precision_reference)
        self.assertIs(discovered_models[2]["precision_reference"], precision_reference)
        self.assertIsNone(precision_patch.call_args.kwargs["precision_context_depth"])

    def test_layer_discovery_keeps_boundary_activities_attached_to_subprocess_candidates(self):
        ocel = _FakeInputOCEL(
            {
                "item_1": "item",
                "order_1": "order",
            },
            [
                {"event_id": "e1", "activity": "review", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
                {"event_id": "e2", "activity": "pre", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e3", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e4", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:03:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e5", "activity": "post", "timestamp": pd.Timestamp("2024-01-01T00:04:00"), "event_objects": ["item_1", "order_1"]},
                {"event_id": "e6", "activity": "x", "timestamp": pd.Timestamp("2024-01-01T00:05:00"), "event_objects": ["item_1", "order_1"]},
            ],
        )
        layer_one_ocel = unittest.mock.Mock()
        layer_one_ocel.events = pd.DataFrame({"ocel:activity": ["pre", "a", "b", "post", "x"]})
        layer_two_first_ocel = unittest.mock.Mock()
        layer_two_first_ocel.events = pd.DataFrame({"ocel:activity": ["review", "pre", "a", "b", "post", "x"]})
        layer_two_second_ocel = unittest.mock.Mock()
        layer_two_second_ocel.events = pd.DataFrame({"ocel:activity": ["review", "pre", "a", "b", "post"]})
        subprocess_component = {
            "id": "subprocess_1",
            "transitions": [
                {"kind": "activity", "label": "a"},
                {"kind": "activity", "label": "b"},
            ],
        }
        build_layer_calls = []

        def build_layer_ocel_side_effect(_, __, ___, selected_object_types, selected_activities):
            build_layer_calls.append((set(selected_object_types), set(selected_activities)))
            if set(selected_object_types) == {"order"}:
                return layer_one_ocel, [], ["a", "b", "post", "pre", "x"]
            if len(build_layer_calls) == 2:
                return layer_two_first_ocel, [], ["a", "b", "post", "pre", "review", "x"]
            return layer_two_second_ocel, [], ["a", "b", "post", "pre", "review"]

        def discover_with_components_side_effect(layer_ocel, *_):
            if layer_ocel is layer_one_ocel:
                return {"activities": ["a", "b", "post", "pre", "x"], "petri_nets": {}}, []
            if layer_ocel is layer_two_first_ocel:
                return {"activities": ["a", "b", "post", "pre", "review", "x"], "petri_nets": {}}, [subprocess_component]
            if layer_ocel is layer_two_second_ocel:
                return {"activities": ["a", "b", "post", "pre", "review"], "petri_nets": {}}, [subprocess_component]
            raise AssertionError("Unexpected layer OCEL")

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            side_effect=build_layer_ocel_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            side_effect=discover_with_components_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            side_effect=[
                [],
                [{"kind": "subprocess", "activities": ("pre", "a", "b", "post")}],
            ],
        ), patch(
            "repo.discovery.discovery_preparation._select_best_pruning_candidate",
            return_value=({"activities": ("pre", "a", "b", "post")}, 0.1),
        ), patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ):
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2},
            )

        layer_two_model = discovered_models[2]

        self.assertEqual(build_layer_calls[1][1], {"a", "b", "post", "pre", "review", "x"})
        self.assertEqual(build_layer_calls[2][1], {"a", "b", "post", "pre", "review"})
        self.assertEqual(layer_two_model["activities"], ["a", "b", "post", "pre", "review"])
        self.assertNotIn("x", layer_two_model["activities"])

    def test_layer_discovery_assigns_unique_subprocess_indices_across_layers(self):
        ocel = _FakeInputOCEL(
            {
                "order_1": "order",
                "item_1": "item",
            },
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["order_1"]},
                {"event_id": "e2", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1"]},
            ],
        )
        layer_one_ocel = unittest.mock.Mock()
        layer_one_ocel.events = pd.DataFrame({"ocel:activity": ["a"]})
        layer_two_ocel = unittest.mock.Mock()
        layer_two_ocel.events = pd.DataFrame({"ocel:activity": ["b"]})
        layer_one_component = {"id": "subprocess_1"}
        layer_two_component = {"id": "subprocess_1"}

        def build_layer_ocel_side_effect(_, __, ___, selected_object_types, ____):
            if set(selected_object_types) == {"order"}:
                return layer_one_ocel, [], ["a"]
            return layer_two_ocel, [], ["b"]

        def discover_with_components_side_effect(layer_ocel, *_):
            if layer_ocel is layer_one_ocel:
                return {"activities": ["a"], "petri_nets": {}}, [layer_one_component]
            if layer_ocel is layer_two_ocel:
                return {"activities": ["b"], "petri_nets": {}}, [layer_two_component]
            raise AssertionError("Unexpected layer OCEL")

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            side_effect=build_layer_ocel_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            side_effect=discover_with_components_side_effect,
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            return_value=[],
        ), patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ):
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2},
            )

        layer_one_subprocess = discovered_models[1]["subprocess_components"][0]
        layer_two_subprocess = discovered_models[2]["subprocess_components"][0]

        self.assertEqual(layer_one_subprocess["id"], "subprocess_1")
        self.assertEqual(layer_one_subprocess["index"], 1)
        self.assertEqual(layer_one_subprocess["global_id"], "subprocess_1")
        self.assertEqual(layer_two_subprocess["id"], "subprocess_1")
        self.assertEqual(layer_two_subprocess["index"], 2)
        self.assertEqual(layer_two_subprocess["global_id"], "subprocess_2")
        self.assertEqual(layer_one_component, {"id": "subprocess_1"})
        self.assertEqual(layer_two_component, {"id": "subprocess_1"})

    def test_layer_discovery_reuses_last_iteration_components(self):
        ocel = _FakeInputOCEL(
            {"item_1": "item"},
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
            ],
        )
        fake_layer_ocel = unittest.mock.Mock()
        fake_layer_ocel.events = pd.DataFrame({"ocel:activity": ["a"]})
        expected_components = [{"id": "subprocess_1"}]

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            return_value=(
                fake_layer_ocel,
                [("e1", "a", pd.Timestamp("2024-01-01T00:00:00"), ("item_1",))],
                ["a"],
            ),
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            return_value=({"activities": ["a"], "petri_nets": {}}, expected_components),
        ) as discover_patch, patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            return_value=[],
        ), patch(
            "repo.discovery.discovery_preparation._select_best_pruning_candidate",
            side_effect=AssertionError("Metric-based pruning should be bypassed"),
        ), patch(
            "repo.discovery.discovery_preparation.detect_subprocess_components",
            side_effect=AssertionError("Should reuse iteration components"),
        ):
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"item": 1},
            )

        self.assertEqual(discovered_models[1]["subprocess_components"][0]["id"], "subprocess_1")
        self.assertEqual(discovered_models[1]["subprocess_components"][0]["index"], 1)
        self.assertEqual(discovered_models[1]["subprocess_components"][0]["global_id"], "subprocess_1")
        discover_patch.assert_called_once()

    def test_layer_discovery_progress_mentions_pm4py_and_subprocess_detection(self):
        ocel = _FakeInputOCEL(
            {"item_1": "item"},
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["item_1"]},
            ],
        )
        fake_layer_ocel = unittest.mock.Mock()
        fake_layer_ocel.events = pd.DataFrame({"ocel:activity": ["a"]})
        fake_layer_ocel.relations = pd.DataFrame({"ocel:eid": ["e1"]})
        fake_ocpn = {"activities": ["a"], "petri_nets": {}}

        buffer = StringIO()
        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            return_value=(
                fake_layer_ocel,
                [("e1", "a", pd.Timestamp("2024-01-01T00:00:00"), ("item_1",))],
                ["a"],
            ),
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn",
            return_value=fake_ocpn,
        ), patch(
            "repo.discovery.discovery_preparation.detect_subprocess_components",
            return_value=[],
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            return_value=[],
        ), patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ), redirect_stdout(buffer):
            discover_models_for_hierarchy(
                ocel,
                {"item": 1},
                show_progress=True,
            )

        output = buffer.getvalue()
        self.assertIn("PM4Py OCPN discovery", output)
        self.assertIn("subprocess detection", output)

    def test_layer_discovery_limits_processing_to_first_two_layers(self):
        ocel = _FakeInputOCEL(
            {
                "order_1": "order",
                "item_1": "item",
                "package_1": "package",
            },
            [
                {"event_id": "e1", "activity": "a", "timestamp": pd.Timestamp("2024-01-01T00:00:00"), "event_objects": ["order_1"]},
                {"event_id": "e2", "activity": "b", "timestamp": pd.Timestamp("2024-01-01T00:01:00"), "event_objects": ["item_1"]},
                {"event_id": "e3", "activity": "c", "timestamp": pd.Timestamp("2024-01-01T00:02:00"), "event_objects": ["package_1"]},
            ],
        )
        fake_layer_ocel = unittest.mock.Mock()
        fake_layer_ocel.events = pd.DataFrame({"ocel:activity": ["a"]})
        fake_layer_ocel.relations = pd.DataFrame({"ocel:eid": ["e1"]})

        with patch(
            "repo.discovery.discovery_preparation._build_layer_ocel",
            return_value=(fake_layer_ocel, [], ["a"]),
        ), patch(
            "repo.discovery.discovery_preparation._discover_ocpn_with_subprocess_components",
            return_value=({"activities": ["a"], "petri_nets": {}}, []),
        ), patch(
            "repo.discovery.discovery_preparation._build_pruning_candidates",
            return_value=[],
        ), patch(
            "repo.discovery.discovery_preparation._discover_activity_resources",
            return_value={},
        ):
            _, discovered_models = discover_models_for_hierarchy(
                ocel,
                {"order": 1, "item": 2, "package": 3},
            )

        self.assertEqual(sorted(discovered_models), [1, 2])


if __name__ == "__main__":
    unittest.main()
