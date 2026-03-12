from discovery.totem import totemDiscovery
from totem_lib import import_ocel

ocel = import_ocel("example_data/ContainerLogistics.sqlite")
temporalRelations = totemDiscovery(ocel)

print(temporalRelations)