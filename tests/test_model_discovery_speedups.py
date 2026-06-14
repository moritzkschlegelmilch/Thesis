import unittest

import pandas as pd

from discovery.framework import AcceptingOCPN, LayerAssignment
from discovery.model_discovery import ModelDiscovery, _log_contains_only_activities


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


class _FakeDiscovery:
    def __init__(self):
        self.calls = 0

    def mine(self, log):
        self.calls += 1
        activities = frozenset(str(activity) for activity in log.events["ocel:activity"].dropna())
        return AcceptingOCPN(
            raw={"activities": tuple(sorted(activities)), "petri_nets": {"item": None}},
            object_types=frozenset({"item"}),
            activities=activities,
        )


class _FailingOptimization:
    def __init__(self):
        self.calls = 0

    def optimize(self, *_, **__):
        self.calls += 1
        raise AssertionError("Optimization should be skipped when there are no candidates.")


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


if __name__ == "__main__":
    unittest.main()
