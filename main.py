import pm4py

from discovery.algorithm import ProcessAreaDiscovery
from repo.discovery.component_deletion_impact import render_component_deletion_impact, discover_component_and_edge_ocpns
from repo.discovery.subprocess_detection import collapse_sub_processes
from repo.helpers.vorbose import render_collapsed_sub_processes
from totem_lib import import_ocel


ocel = import_ocel("example_data/ContainerLogistics.sqlite")

discovery = ProcessAreaDiscovery(ocel)
discovery.get_layers()
discovery.discover_models()
discovery.visualize()

#render_collapsed_sub_processes(discovery.discovered_models[3]["ocpn"], discovery.discovered_models[3]["subprocess_components"]).show()


#discovery.visualize()

# component_activities = {"Drive to Terminal"}
#
# component_plus_edges_ocpn, edge_only_ocpn = discover_component_and_edge_ocpns(
#     ocel=discovery.discovered_models[3]["ocel"],
#     ocpn=discovery.discovered_models[3]["ocpn"],
#     component_activity_labels=component_activities,
#     lower_layer_ocel=discovery.discovered_models[2]["ocel"],
# )
#
# #
# # image = render_component_deletion_impact(
# #     discovery.discovered_models[3]['ocpn'],
# #     component_activities,
# #     highlight_color="#ffd966",
# # )
# #
# # if image is not None:
# #     image.show()
# #
# pm4py.view_ocpn(component_plus_edges_ocpn, format="png", bgcolor="white")
# pm4py.view_ocpn(edge_only_ocpn, format="png", bgcolor="white")

#discovery.visualize()

#print(test_log_folder("./example_simulation_data", "./testing_output", recursive=True))