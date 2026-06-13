from .discovery import ProcessAreaDiscoveryFramework
from .scorables import (
    CardinalityRelationScorer,
    DivergenceScorer,
    TimeRelationScorer,
)


class ProcessAreaDiscovery(ProcessAreaDiscoveryFramework):
    def __init__(self, ocel):
        super().__init__(ocel, [
            TimeRelationScorer(1),
            CardinalityRelationScorer(1),
            DivergenceScorer(1)
        ])
