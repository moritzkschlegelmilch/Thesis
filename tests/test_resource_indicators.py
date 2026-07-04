import unittest

import pandas as pd

from discovery.scorables.TimeRelationScorer import TimeRelationScorer
from discovery.totem import clear_totem_cache


class _TemporalOCEL:
    o2o_graph_edges = ()

    def __init__(self, event_records, object_to_type):
        self.object_types = sorted(set(object_to_type.values()))
        self._event_records = {
            event_id: {
                "timestamp": timestamp,
                "objects": tuple(objects),
            }
            for event_id, timestamp, objects in event_records
        }
        self._object_to_type = dict(object_to_type)
        self.events = pd.DataFrame({"_eventId": [record[0] for record in event_records]})

    def get_event_timestamp(self, event_id):
        return self._event_records[event_id]["timestamp"]

    def get_value(self, event_id, attribute):
        if attribute != "event_objects":
            raise KeyError(attribute)
        return self._event_records[event_id]["objects"]

    def get_event_objects_by_type(self, event_id, object_type):
        return [
            obj
            for obj in self._event_records[event_id]["objects"]
            if self._object_to_type[obj] == object_type
        ]


class TemporalResourceIndicatorTests(unittest.TestCase):
    def setUp(self):
        clear_totem_cache()

    def tearDown(self):
        clear_totem_cache()

    def test_temporal_force_counts_lifespan_containment_in_forward_direction(self):
        ocel = _TemporalOCEL(
            [
                ("e1", 0, ("product_1",)),
                ("e2", 25, ("product_1", "item_1")),
                ("e3", 75, ("product_1", "item_1")),
                ("e4", 100, ("product_1",)),
            ],
            {
                "product_1": "product",
                "item_1": "item",
            },
        )

        scorer = TimeRelationScorer(1)
        scorer.prepare(ocel)

        self.assertEqual(scorer.assign_score_push("product", "item"), 1)
        self.assertEqual(scorer.assign_score_push("item", "product"), -1)
        self.assertEqual(scorer.assign_score_pull("product", "item"), 0)

    def test_temporal_pull_counts_any_partial_lifespan_overlap(self):
        ocel = _TemporalOCEL(
            [
                ("e1", 0, ("item_1",)),
                ("e2", 50, ("item_1", "package_1")),
                ("e3", 100, ("item_1",)),
                ("e4", 150, ("package_1",)),
            ],
            {
                "item_1": "item",
                "package_1": "package",
            },
        )

        scorer = TimeRelationScorer(1)
        scorer.prepare(ocel)

        self.assertEqual(scorer.assign_score_pull("item", "package"), 1)
        self.assertEqual(scorer.assign_score_pull("package", "item"), 1)
        self.assertEqual(scorer.assign_score_push("item", "package"), 0)


if __name__ == "__main__":
    unittest.main()
