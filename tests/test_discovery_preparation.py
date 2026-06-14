import unittest

from discovery.discovery_preparation import _build_projected_ocel


class DiscoveryPreparationTests(unittest.TestCase):
    def test_projected_ocel_prunes_object_types_without_event_couples(self):
        source_ocel = object()
        object_to_type = {
            "order_1": "Order",
            "dispute_1": "Dispute",
        }
        event_records = (
            ("e1", "start", 1727855715, ("order_1",)),
            ("e2", "review", 1727855775, ("order_1", "dispute_1")),
        )

        projected_ocel, included_records, included_activities = _build_projected_ocel(
            source_ocel,
            event_records,
            object_to_type,
            {"start", "review"},
        )

        self.assertEqual(projected_ocel.objects["ocel:type"].unique().tolist(), ["Order"])
        self.assertEqual(projected_ocel.relations["ocel:type"].unique().tolist(), ["Order"])
        self.assertEqual([record[0] for record in included_records], ["e1", "e2"])
        self.assertEqual(included_activities, ["review", "start"])

    def test_projected_ocel_drops_events_left_without_relations_after_pruning(self):
        source_ocel = object()
        object_to_type = {
            "order_1": "Order",
            "credit_note_1": "CreditNote",
        }
        event_records = (
            ("e1", "start", 1727855715, ("order_1",)),
            ("e2", "finish", 1727855775, ("order_1",)),
            ("e3", "issue credit", 1727855835, ("credit_note_1",)),
        )

        projected_ocel, included_records, included_activities = _build_projected_ocel(
            source_ocel,
            event_records,
            object_to_type,
            {"start", "finish", "issue credit"},
        )

        self.assertEqual(projected_ocel.events["ocel:eid"].tolist(), ["e1", "e2"])
        self.assertEqual(projected_ocel.objects["ocel:type"].unique().tolist(), ["Order"])
        self.assertEqual([record[0] for record in included_records], ["e1", "e2"])
        self.assertEqual(included_activities, ["finish", "start"])


if __name__ == "__main__":
    unittest.main()
