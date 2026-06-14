import os
import tempfile
import textwrap
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from totem_lib.ocel.importer import load_events_from_xml


class OCELImporterTests(unittest.TestCase):
    def test_xml_import_preserves_timezone_less_timestamps(self):
        xml = textwrap.dedent(
            """\
            <log>
              <events>
                <event id="event_1" type="Receive Order" time="2024-10-02T07:55:15.348555">
                  <objects>
                    <relationship object-id="order_1" qualifier=""/>
                  </objects>
                </event>
              </events>
            </log>
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
            handle.write(xml)
            path = handle.name

        try:
            events = load_events_from_xml(path)
        finally:
            os.unlink(path)

        self.assertEqual(events["_timestampUnix"].null_count(), 0)
        self.assertEqual(events["_timestampUnix"].item(), 1727855715)


if __name__ == "__main__":
    unittest.main()
