from repo.discovery.data_sources.totem_datasource import TotemDatasource
from repo.discovery.discovery import ProcessAreaDiscoveryFramework


class ProcessAreaDiscovery(ProcessAreaDiscoveryFramework):
    def __init__(self, ocel):
        super().__init__(ocel, [
            TotemDatasource(1)
        ])
