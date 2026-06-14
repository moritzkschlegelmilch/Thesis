import os
import unittest
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from repo.discovery.net_quality import NetQuality, _BindingStep, _ReplayEvent


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


def _build_single_type_ocel(traces, object_type="order"):
    base_timestamp = pd.Timestamp("2024-01-01 00:00:00")
    event_rows = []
    object_rows = []
    relation_rows = []
    event_counter = 1

    for case_index, activities in enumerate(traces, start=1):
        object_id = f"{object_type}_{case_index}"
        object_rows.append({
            "ocel:oid": object_id,
            "ocel:type": object_type,
        })

        for activity in activities:
            event_id = f"e{event_counter}"
            timestamp = base_timestamp + pd.Timedelta(minutes=event_counter)
            event_rows.append({
                "ocel:eid": event_id,
                "ocel:activity": activity,
                "ocel:timestamp": timestamp,
            })
            relation_rows.append({
                "ocel:eid": event_id,
                "ocel:activity": activity,
                "ocel:timestamp": timestamp,
                "ocel:oid": object_id,
                "ocel:type": object_type,
                "ocel:qualifier": None,
            })
            event_counter += 1

    return OCEL(
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
    )


def _build_flower_ocpn(activity_labels, object_type="order"):
    net = PetriNet(f"{object_type}_flower")
    place = PetriNet.Place("p")
    net.places.add(place)

    for activity in activity_labels:
        transition = PetriNet.Transition(activity, activity)
        net.transitions.add(transition)
        petri_utils.add_arc_from_to(place, transition, net)
        petri_utils.add_arc_from_to(transition, place, net)

    initial_marking = Marking({place: 1})
    final_marking = Marking({place: 1})

    return {
        "activities": list(activity_labels),
        "petri_nets": {
            object_type: (net, initial_marking, final_marking),
        },
        "double_arcs_on_activity": {
            object_type: {
                activity: False
                for activity in activity_labels
            },
        },
        "tbr_results": {},
    }


def _build_restrictive_ocpn(object_type="order"):
    net = PetriNet(f"{object_type}_restrictive")

    start = PetriNet.Place("start")
    after_create = PetriNet.Place("after_create")
    after_approve = PetriNet.Place("after_approve")
    end = PetriNet.Place("end")

    for place in (start, after_create, after_approve, end):
        net.places.add(place)

    create = PetriNet.Transition("create_t", "create")
    approve = PetriNet.Transition("approve_t", "approve")
    complete = PetriNet.Transition("complete_t", "complete")

    for transition in (create, approve, complete):
        net.transitions.add(transition)

    petri_utils.add_arc_from_to(start, create, net)
    petri_utils.add_arc_from_to(create, after_create, net)
    petri_utils.add_arc_from_to(after_create, approve, net)
    petri_utils.add_arc_from_to(approve, after_approve, net)
    petri_utils.add_arc_from_to(after_approve, complete, net)
    petri_utils.add_arc_from_to(complete, end, net)

    initial_marking = Marking({start: 1})
    final_marking = Marking({end: 1})

    return {
        "activities": ["create", "approve", "complete"],
        "petri_nets": {
            object_type: (net, initial_marking, final_marking),
        },
        "double_arcs_on_activity": {
            object_type: {
                "create": False,
                "approve": False,
                "complete": False,
            },
        },
        "tbr_results": {},
    }


class NetQualityTests(unittest.TestCase):
    def test_replay_terminal_states_honor_max_nodes_per_replay(self):
        quality = NetQuality(_build_flower_ocpn(["a"]), max_nodes_per_replay=2)
        replay_event = _ReplayEvent(
            context_key=(),
            binding_sequence=(),
            context_tokens_by_type=(),
        )
        successors_by_state = {
            "start": ["first", "second"],
            "first": ["third"],
            "second": ["fourth"],
            "third": [],
            "fourth": [],
        }

        with mock.patch.object(
            quality,
            "_initial_state",
            return_value="start",
        ), mock.patch.object(
            quality,
            "_state_key",
            side_effect=lambda state: state,
        ), mock.patch.object(
            quality,
            "_fire_silent",
            side_effect=lambda state: successors_by_state[state],
        ):
            terminal_states = quality._replay_terminal_states(replay_event)

        self.assertEqual(set(terminal_states), {"start", "first"})

    def test_replay_honors_max_nodes_per_replay(self):
        quality = NetQuality(_build_flower_ocpn(["a"]), max_nodes_per_replay=2)
        replay_event = _ReplayEvent(
            context_key=(),
            binding_sequence=(),
            context_tokens_by_type=(),
        )
        successors_by_state = {
            "start": ["first", "second"],
            "first": ["third"],
            "second": ["fourth"],
            "third": [],
            "fourth": [],
        }

        with mock.patch.object(
            quality,
            "_initial_state",
            return_value="start",
        ), mock.patch.object(
            quality,
            "_state_key",
            side_effect=lambda state: state,
        ), mock.patch.object(
            quality,
            "_enabled_labels",
            side_effect=lambda state: {state},
        ), mock.patch.object(
            quality,
            "_fire_silent",
            side_effect=lambda state: successors_by_state[state],
        ):
            enabled = quality._replay(replay_event)

        self.assertEqual(enabled, {"start", "first"})

    def test_discovered_deterministic_pm4py_ocpn_has_perfect_quality(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete"],
            ["create", "approve", "complete"],
        ])
        ocpn = pm4py.discover_oc_petri_net(ocel)

        quality = NetQuality(ocpn, ocel)

        self.assertAlmostEqual(quality.fitness(), 1.0, places=6)
        self.assertAlmostEqual(quality.precision(), 1.0, places=6)

    def test_flower_model_keeps_fitness_but_loses_precision(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete"],
            ["create", "approve", "complete"],
        ])
        ocpn = _build_flower_ocpn(["create", "approve", "complete"])

        quality = NetQuality(ocpn)

        self.assertAlmostEqual(quality.fitness(ocel), 1.0, places=6)
        self.assertAlmostEqual(quality.precision(ocel), 1 / 3, places=6)

    def test_restrictive_model_loses_fitness_but_keeps_precision(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete"],
            ["create", "reject", "complete"],
        ])
        ocpn = _build_restrictive_ocpn()

        quality = NetQuality(ocpn, ocel)

        self.assertAlmostEqual(quality.fitness(), 4 / 6, places=6)
        self.assertAlmostEqual(quality.precision(), 1.0, places=6)

    def test_precision_context_sampling_uses_weighted_draw_average(self):
        quality = NetQuality(
            _build_flower_ocpn(["create", "approve"]),
            precision_context_sample_size=4,
            random_seed=7,
        )
        prepared = {
            "events": ("e1", "e2", "e3"),
            "ctx": {
                "e1": "ctx_a",
                "e2": "ctx_a",
                "e3": "ctx_b",
            },
            "log": {
                "ctx_a": frozenset({"create"}),
                "ctx_b": frozenset({"approve"}),
            },
            "replay": {
                "ctx_a": [object(), object()],
                "ctx_b": [object()],
            },
        }

        with mock.patch.object(
            quality,
            "_prepare_log",
            return_value=prepared,
        ), mock.patch.object(
            quality,
            "_sample_precision_context_draw_counts",
            return_value={"ctx_a": 3, "ctx_b": 1},
        ), mock.patch.object(
            quality,
            "_build_model_enabled_by_context",
            return_value={
                "ctx_a": frozenset({"create", "approve"}),
                "ctx_b": frozenset({"approve"}),
            },
        ) as build_patch:
            precision = quality.precision()

        self.assertAlmostEqual(precision, (3 * 0.5 + 1 * 1.0) / 4, places=6)
        build_patch.assert_called_once_with(
            prepared,
            selected_contexts={"ctx_a": 3, "ctx_b": 1},
            show_progress=False,
            progress_desc="Replaying sampled contexts",
        )

    def test_precision_context_sampling_preserves_skip_behavior(self):
        quality = NetQuality(
            _build_flower_ocpn(["create", "approve"]),
            precision_context_sample_size=4,
            random_seed=7,
        )
        prepared = {
            "events": ("e1", "e2"),
            "ctx": {
                "e1": "ctx_a",
                "e2": "ctx_b",
            },
            "log": {
                "ctx_a": frozenset({"create"}),
                "ctx_b": frozenset({"approve"}),
            },
            "replay": {
                "ctx_a": [object()],
                "ctx_b": [object()],
            },
        }

        with mock.patch.object(
            quality,
            "_prepare_log",
            return_value=prepared,
        ), mock.patch.object(
            quality,
            "_sample_precision_context_draw_counts",
            return_value={"ctx_a": 1, "ctx_b": 3},
        ), mock.patch.object(
            quality,
            "_build_model_enabled_by_context",
            return_value={
                "ctx_a": frozenset({"create", "approve"}),
                "ctx_b": frozenset({"reject"}),
            },
        ):
            precision = quality.precision()

        self.assertAlmostEqual(precision, 0.5, places=6)

    def test_prepare_log_can_limit_context_predecessor_depth(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete", "ship"],
        ])
        ocpn = _build_flower_ocpn(["create", "approve", "complete", "ship"])
        depth_one_quality = NetQuality(ocpn, ocel, precision_context_depth=1)
        depth_two_quality = NetQuality(ocpn, ocel, precision_context_depth=2)
        full_quality = NetQuality(ocpn, ocel)

        depth_one_prepared = depth_one_quality._prepare_log(ocel)
        depth_two_prepared = depth_two_quality._prepare_log(ocel)
        full_prepared = full_quality._prepare_log(ocel)

        depth_one_ctx = depth_one_prepared["ctx"]["e4"]
        depth_two_ctx = depth_two_prepared["ctx"]["e4"]
        full_ctx = full_prepared["ctx"]["e4"]

        depth_one_binding = depth_one_prepared["replay"][depth_one_ctx][0].binding_sequence
        depth_two_binding = depth_two_prepared["replay"][depth_two_ctx][0].binding_sequence
        full_binding = full_prepared["replay"][full_ctx][0].binding_sequence

        self.assertEqual([step.label for step in depth_one_binding], ["complete"])
        self.assertEqual([step.label for step in depth_two_binding], ["approve", "complete"])
        self.assertEqual(
            [step.label for step in full_binding],
            ["create", "approve", "complete"],
        )

    def test_prepare_log_can_materialize_replay_lazily(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete"],
        ])
        quality = NetQuality(_build_flower_ocpn(["create", "approve", "complete"]), ocel)

        prepared = quality._prepare_log(ocel, materialize_replay=False)

        self.assertEqual(prepared["replay"], {})
        self.assertEqual(sum(prepared["context_weights"].values()), 3)

        selected_context = prepared["ctx"]["e3"]
        replay = quality._materialize_replay_events(
            prepared,
            selected_contexts=(selected_context,),
        )

        self.assertEqual(tuple(replay), (selected_context,))
        self.assertEqual(len(replay[selected_context]), 1)
        self.assertEqual(
            [step.label for step in replay[selected_context][0].binding_sequence],
            ["create", "approve"],
        )

    def test_replay_cache_alpha_normalizes_object_ids(self):
        quality = NetQuality(_build_flower_ocpn(["a"]))

        def replay_event(object_id):
            token = ("order", object_id)
            return _ReplayEvent(
                context_key=(),
                binding_sequence=(
                    _BindingStep(
                        label="a",
                        objects_by_type=(("order", (token,)),),
                    ),
                ),
                context_tokens_by_type=(("order", (token,)),),
            )

        self.assertEqual(quality._replay(replay_event("order_1")), frozenset({"a"}))
        self.assertEqual(len(quality._replay_cache), 1)
        self.assertEqual(len(quality._terminal_replay_cache), 1)

        self.assertEqual(quality._replay(replay_event("order_99")), frozenset({"a"}))
        self.assertEqual(len(quality._replay_cache), 1)
        self.assertEqual(len(quality._terminal_replay_cache), 1)

    def test_replay_alpha_normalization_preserves_object_equality(self):
        quality = NetQuality(_build_flower_ocpn(["a", "b"]))

        def signature(event):
            return quality._replay_signature(quality._canonical_replay_event(event))

        same_token = ("order", "order_1")
        first_token = ("order", "order_1")
        second_token = ("order", "order_2")

        same_object_event = _ReplayEvent(
            context_key=(),
            binding_sequence=(
                _BindingStep("a", (("order", (same_token,)),)),
                _BindingStep("b", (("order", (same_token,)),)),
            ),
            context_tokens_by_type=(("order", (same_token,)),),
        )
        different_object_event = _ReplayEvent(
            context_key=(),
            binding_sequence=(
                _BindingStep("a", (("order", (first_token,)),)),
                _BindingStep("b", (("order", (second_token,)),)),
            ),
            context_tokens_by_type=(("order", (first_token, second_token)),),
        )

        self.assertNotEqual(signature(same_object_event), signature(different_object_event))

    def test_progress_output_can_be_enabled(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete"],
            ["create", "approve", "complete"],
        ])
        ocpn = pm4py.discover_oc_petri_net(ocel)
        quality = NetQuality(ocpn, ocel)

        buffer = StringIO()
        with redirect_stdout(buffer):
            fitness = quality.fitness(show_progress=True)

        output = buffer.getvalue()

        self.assertAlmostEqual(fitness, 1.0, places=6)
        self.assertIn("Preparing OCPA-style event contexts", output)
        self.assertIn("Replaying contexts", output)
        self.assertIn("Aggregating scores", output)
        self.assertIn("Done.", output)

    def test_complexity_for_single_type_net(self):
        quality = NetQuality(_build_restrictive_ocpn())

        self.assertEqual(quality.complexity(), 13)

    def test_complexity_merges_visible_transitions_like_pm4py(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_out"],
            transitions={"pack_item": "pack"},
            arcs=[
                ("item_in", "pack_item"),
                ("pack_item", "item_out"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["order_in", "order_out"],
            transitions={"pack_order": "pack"},
            arcs=[
                ("order_in", "pack_order"),
                ("pack_order", "order_out"),
            ],
        )

        quality = NetQuality(_build_ocpn({"item": item_net, "order": order_net}))

        self.assertEqual(quality.complexity(), 13)

    def test_complexity_weights_silent_transitions_by_two(self):
        silent_net = _build_net(
            "order",
            places=["start", "mid", "end"],
            transitions={
                "tau": None,
                "a": "a",
            },
            arcs=[
                ("start", "tau"),
                ("tau", "mid"),
                ("mid", "a"),
                ("a", "end"),
            ],
        )

        quality = NetQuality(_build_ocpn({"order": silent_net}))

        self.assertEqual(quality.complexity(), 10)

    def test_complexity_counts_variable_arcs_from_double_arc_metadata(self):
        variable_net = _build_net(
            "order",
            places=["start", "end"],
            transitions={"a": "a"},
            arcs=[
                ("start", "a"),
                ("a", "end"),
            ],
        )
        ocpn = _build_ocpn({"order": variable_net})
        ocpn["double_arcs_on_activity"]["order"]["a"] = True

        quality = NetQuality(ocpn)

        self.assertEqual(quality.complexity(), 7)

    def test_complexity_doubles_silent_synchronization_penalty(self):
        item_net = _build_net(
            "item",
            places=["item_in", "item_out"],
            transitions={"tau": None},
            arcs=[
                ("item_in", "tau"),
                ("tau", "item_out"),
            ],
        )
        order_net = _build_net(
            "order",
            places=["order_in", "order_out"],
            transitions={"tau": None},
            arcs=[
                ("order_in", "tau"),
                ("tau", "order_out"),
            ],
        )

        quality = NetQuality(_build_ocpn({"item": item_net, "order": order_net}))

        self.assertEqual(quality.complexity(), 18)

    def test_prepare_log_groups_each_event_once(self):
        ocel = _build_single_type_ocel([
            ["create", "approve", "complete"],
        ])
        quality = NetQuality(_build_flower_ocpn(["create", "approve", "complete"]), ocel)
        original_group = quality._group_event_tokens_by_type

        with mock.patch.object(
            quality,
            "_group_event_tokens_by_type",
            wraps=original_group,
        ) as group_patch:
            prepared = quality._prepare_log(ocel)

        self.assertEqual(len(prepared["events"]), 3)
        self.assertEqual(group_patch.call_count, 5)


if __name__ == "__main__":
    unittest.main()
