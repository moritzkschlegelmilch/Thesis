from discovery.algorithm import ProcessAreaDiscovery
from totem_lib import import_ocel, totemDiscovery

ocel = import_ocel("example_data/ContainerLogistics.sqlite")

discovery = ProcessAreaDiscovery(ocel)

discovery.run()
# totem = totemDiscovery(ocel)
# print(totem.tempgraph)
# totem.visualize(
#     output_dir="figures", output_file="totem_example.pdf", ot_to_hex_color={}
# )