from repo.discovery.Scorable import DataSource, Scorable
from repo.discovery.data_sources.totem_scorables.TimeRelationScorer import TimeRelationScorer
from repo.discovery.totem import totemDiscovery


class TotemDatasource(DataSource):

    def __init__(self, eps_time):
        self.temporal_relations = None
        self.cardinality_relations = None
        self.event_cardinality_relations = None

        super().__init__([
            TimeRelationScorer(eps_time)
        ])

    def prepare(self, ocel):
        self.temporal_relations = totemDiscovery(ocel)
