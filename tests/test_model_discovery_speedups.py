import unittest

import pandas as pd

from discovery.framework import AcceptingOCPN, LayerAssignment
from discovery.model_discovery import (
    ModelDiscovery,
    _log_contains_only_activities,
)


class _TinyOCEL:
    event_id_column = "ocel:eid"
    event_activity = "ocel:activity"
    event_timestamp = "ocel:timestamp"
    object_id_column = "ocel:oid"
    object_type_column = "ocel:type"
    o2o_graph_edges = ()

    def __init__(self):
        timestamps = pd.to_datetime([
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
        ])
        self.events = pd.DataFrame({
            "ocel:eid": ["e1", "e2"],
            "ocel:activity": ["a", "b"],
            "ocel:timestamp": timestamps,
        })
        self.objects = pd.DataFrame({
            "ocel:oid": ["item_1"],
            "ocel:type": ["item"],
        })
        self.relations = pd.DataFrame({
            "ocel:eid": ["e1", "e2"],
            "ocel:activity": ["a", "b"],
            "ocel:timestamp": timestamps,
            "ocel:oid": ["item_1", "item_1"],
            "ocel:type": ["item", "item"],
            "ocel:qualifier": [None, None],
        })


class _LayeredOCEL:
    event_id_column = "ocel:eid"
    event_activity = "ocel:activity"
    event_timestamp = "ocel:timestamp"
    object_id_column = "ocel:oid"
    object_type_column = "ocel:type"
    o2o_graph_edges = ()

    def __init__(self):
        timestamps = pd.to_datetime([
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:02:00",
        ])
        self.events = pd.DataFrame({
            "ocel:eid": ["e1", "e2", "e3"],
            "ocel:activity": ["order_only", "shared", "item_native"],
            "ocel:timestamp": timestamps,
        })
        self.objects = pd.DataFrame({
            "ocel:oid": ["order_1", "item_1"],
            "ocel:type": ["order", "item"],
        })
        self.relations = pd.DataFrame({
            "ocel:eid": ["e1", "e2", "e2", "e3"],
            "ocel:activity": ["order_only", "shared", "shared", "item_native"],
            "ocel:timestamp": [timestamps[0], timestamps[1], timestamps[1], timestamps[2]],
            "ocel:oid": ["order_1", "order_1", "item_1", "item_1"],
            "ocel:type": ["order", "order", "item", "item"],
            "ocel:qualifier": [None, None, None, None],
        })


class _FakeDiscovery:
    def __init__(self):
        self.calls = 0

    def mine(self, log):
        self.calls += 1
        activities = frozenset(str(activity) for activity in log.events["ocel:activity"].dropna())
        object_types = frozenset(str(object_type) for object_type in log.relations["ocel:type"].dropna())
        return AcceptingOCPN(
            raw={
                "activities": tuple(sorted(activities)),
                "petri_nets": {
                    object_type: None
                    for object_type in object_types
                },
                "double_arcs_on_activity": {
                    object_type: {}
                    for object_type in object_types
                },
            },
            object_types=object_types,
            activities=activities,
        )


class _FailingOptimization:
    def __init__(self):
        self.calls = 0

    def optimize(self, *_, **__):
        self.calls += 1
        raise AssertionError("Optimization should be skipped when there are no candidates.")


class _RecordingOptimization:
    def __init__(self):
        self.calls = []

    def optimize(
        self,
        log,
        hierarchy,
        candidate_activities,
        *,
        required_activities=frozenset(),
    ):
        self.calls.append({
            "log_activities": frozenset(log.events["ocel:activity"].dropna()),
            "candidate_activities": frozenset(candidate_activities),
            "required_activities": frozenset(required_activities),
        })
        return frozenset(required_activities | candidate_activities)


class _FakeSubprocessMiner:
    def mine(self, *_):
        return ()


class ModelDiscoverySpeedupTests(unittest.TestCase):
    def test_log_contains_only_activities_detects_already_projected_log(self):
        log = _TinyOCEL()

        self.assertTrue(_log_contains_only_activities(log, frozenset({"a", "b", "c"})))
        self.assertFalse(_log_contains_only_activities(log, frozenset({"a"})))

    def test_model_discovery_skips_optimization_when_no_candidate_activities(self):
        log = _TinyOCEL()
        discovery = _FakeDiscovery()
        optimization = _FailingOptimization()
        model_discovery = ModelDiscovery(
            discovery,
            optimization,
            subprocess_miner=_FakeSubprocessMiner(),
        )

        hierarchy = model_discovery.mine(
            log,
            LayerAssignment.from_mapping({"item": 1}),
            [1],
        )

        self.assertEqual(optimization.calls, 0)
        self.assertEqual(discovery.calls, 1)
        self.assertEqual(len(hierarchy.areas), 1)
        self.assertEqual(hierarchy.areas[0].activities, frozenset({"a", "b"}))

    def test_model_discovery_prunes_candidates_absent_from_projected_layer_log(self):
        log = _LayeredOCEL()
        optimization = _RecordingOptimization()
        model_discovery = ModelDiscovery(
            _FakeDiscovery(),
            optimization,
            subprocess_miner=_FakeSubprocessMiner(),
        )

        hierarchy = model_discovery.mine(
            log,
            LayerAssignment.from_mapping({"order": 1, "item": 2}),
            [1, 1],
        )

        self.assertEqual(len(optimization.calls), 1)
        self.assertEqual(
            optimization.calls[0]["log_activities"],
            frozenset({"shared", "item_native"}),
        )
        self.assertEqual(
            optimization.calls[0]["candidate_activities"],
            frozenset({"shared"}),
        )
        self.assertEqual(
            optimization.calls[0]["required_activities"],
            frozenset({"item_native"}),
        )
        self.assertEqual(len(hierarchy.areas), 2)
        self.assertEqual(
            hierarchy.areas[1].activities,
            frozenset({"shared", "item_native"}),
        )


if __name__ == "__main__":
    unittest.main()
