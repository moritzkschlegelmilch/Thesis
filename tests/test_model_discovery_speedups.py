import unittest
from unittest.mock import patch

import pandas as pd

from discovery.framework import (
    AcceptingOCPN,
    AdvancedProcessArea,
    AdvancedProcessAreaHierarchy,
    LayerAssignment,
)
from discovery.model_discovery import (
    HierarchyQualityEvaluator,
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


class _TwoLayerOCEL:
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
            "2024-01-01T00:03:00",
        ])
        self.events = pd.DataFrame({
            "ocel:eid": ["e1", "e2", "e3", "e4"],
            "ocel:activity": ["a", "a", "b", "b"],
            "ocel:timestamp": timestamps,
        })
        self.objects = pd.DataFrame({
            "ocel:oid": ["order_1", "item_1"],
            "ocel:type": ["order", "item"],
        })
        self.relations = pd.DataFrame({
            "ocel:eid": ["e1", "e2", "e3", "e4"],
            "ocel:activity": ["a", "a", "b", "b"],
            "ocel:timestamp": timestamps,
            "ocel:oid": ["order_1", "order_1", "item_1", "item_1"],
            "ocel:type": ["order", "order", "item", "item"],
            "ocel:qualifier": [None, None, None, None],
        })


class _FakeDiscovery:
    def __init__(self):
        self.calls = 0
        self.activity_calls = []

    def mine(self, log):
        self.calls += 1
        activities = frozenset(str(activity) for activity in log.events["ocel:activity"].dropna())
        self.activity_calls.append(tuple(sorted(activities)))
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


class _FakeSubprocessMiner:
    def mine(self, *_):
        return ()


class _FakeCollapsedNetBuilder:
    def collapse(self, net, _):
        return net


class _RecordingPrecisionCalculator:
    def __init__(self):
        self.prepared_contexts = []
        self.precision_calls = []

    def prepare_log_context(self, log):
        activities = tuple(sorted(str(activity) for activity in log.events["ocel:activity"].dropna().unique()))
        context = {"activities": activities, "index": len(self.prepared_contexts)}
        self.prepared_contexts.append(context)
        return context

    def precision(self, log, net, *, prepared_context=None):
        activities = tuple(sorted(str(activity) for activity in log.events["ocel:activity"].dropna().unique()))
        self.precision_calls.append((activities, tuple(sorted(net.activities)), prepared_context))
        return 0.5


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

    def test_windowed_quality_uses_delta_windows_and_reuses_precision_contexts(self):
        log = _TwoLayerOCEL()
        discovery = _FakeDiscovery()
        precision = _RecordingPrecisionCalculator()
        evaluator = HierarchyQualityEvaluator(
            discovery,
            _FakeCollapsedNetBuilder(),
            precision,
        )
        hierarchy = AdvancedProcessAreaHierarchy((
            AdvancedProcessArea(
                object_types=frozenset({"order"}),
                activities=frozenset({"a"}),
                net=AcceptingOCPN(
                    raw={"activities": ("a",), "petri_nets": {"order": None}},
                    object_types=frozenset({"order"}),
                    activities=frozenset({"a"}),
                ),
                resources={},
                subprocesses=(),
            ),
            AdvancedProcessArea(
                object_types=frozenset({"item"}),
                activities=frozenset({"b"}),
                net=AcceptingOCPN(
                    raw={"activities": ("b",), "petri_nets": {"item": None}},
                    object_types=frozenset({"item"}),
                    activities=frozenset({"b"}),
                ),
                resources={},
                subprocesses=(),
            ),
        ))

        with patch("discovery.model_discovery._complexity", return_value=1.0):
            evaluator.evaluate(log, hierarchy, [0, 1])

        self.assertEqual(discovery.activity_calls, [("a",), ("a", "b")])
        self.assertEqual(
            [context["activities"] for context in precision.prepared_contexts],
            [("a",), ("a", "b")],
        )
        self.assertEqual(len(precision.precision_calls), 4)
        self.assertIs(precision.precision_calls[0][2], precision.prepared_contexts[0])
        self.assertIs(precision.precision_calls[1][2], precision.prepared_contexts[0])
        self.assertIs(precision.precision_calls[2][2], precision.prepared_contexts[1])
        self.assertIs(precision.precision_calls[3][2], precision.prepared_contexts[1])


if __name__ == "__main__":
    unittest.main()
