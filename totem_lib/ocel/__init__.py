from .ocel import ObjectCentricEventLog
from .importer import import_ocel

try:
    from .pm4py_adapter import convert_ocel_polars_to_pm4py, PolarsOCELAdapter
except ImportError as exc:
    def convert_ocel_polars_to_pm4py(*_, **__):
        raise ImportError(
            "PM4Py adapter support is unavailable in this environment."
        ) from exc

    class PolarsOCELAdapter:
        def __init__(self, *_, **__):
            raise ImportError(
                "PM4Py adapter support is unavailable in this environment."
            ) from exc
