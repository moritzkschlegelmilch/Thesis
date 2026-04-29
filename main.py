import matplotlib.pyplot as plt

from discovery.algorithm import ProcessAreaDiscovery
from discovery.region_detection import detect_object_centric_regions
from repo.helpers.vorbose import render_object_centric_regions
from totem_lib import import_ocel


ocel = import_ocel("example_data/angular_github_commits_ocel.xml")

discovery = ProcessAreaDiscovery(ocel)
discovery.get_layers()
discovery.discover_models()
discovery.visualize()