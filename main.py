import pm4py

from discovery.algorithm import ProcessAreaDiscovery
from repo.helpers.testing import test_log_folder
from totem_lib import import_ocel
#
ocel = import_ocel("example_simulation_data/unfair/hiring_log_high.xml")

discovery = ProcessAreaDiscovery(ocel)
discovery.run()
discovery.visualize()

#print(test_log_folder("./example_simulation_data", "./testing_output", recursive=True))