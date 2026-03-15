from discovery.discovery import ProcessAreaDiscovery
from totem_lib import import_ocel

ocel = import_ocel("example_data/ContainerLogistics.sqlite")

discovery = ProcessAreaDiscovery(ocel)
discovery.prepare()
discovery.assign_scores()
discovery.solve_ilp()

