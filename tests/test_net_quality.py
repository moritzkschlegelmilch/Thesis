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

from repo.discovery.net_quality import NetQuality, _ReplayEvent


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
        self.assertIn("Preparing event contexts", output)
        self.assertIn("Replaying contexts", output)
        self.assertIn("Aggregating scores", output)
        self.assertIn("Done.", output)

    def test_complexity_for_single_type_net(self):
        quality = NetQuality(_build_restrictive_ocpn())

        self.assertEqual(quality.complexity(), 12)

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

        self.assertEqual(quality.complexity(), 12)

    def test_complexity_uses_beta_for_silent_transitions(self):
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

        self.assertEqual(quality.complexity(), 12)
        self.assertEqual(quality.complexity(alpha=2, beta=3), 14)


if __name__ == "__main__":
    unittest.main()
