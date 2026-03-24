from repo.discovery.scorables.CardinalityRelationScorer import CardinalityRelationScorer
from repo.discovery.scorables.TimeRelationScorer import TimeRelationScorer
from repo.discovery.discovery import ProcessAreaDiscoveryFramework


class ProcessAreaDiscovery(ProcessAreaDiscoveryFramework):
    def __init__(self, ocel):
        super().__init__(ocel, [
            TimeRelationScorer(1),
            CardinalityRelationScorer(1)
        ])
