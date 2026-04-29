import matplotlib.pyplot as plt

from discovery.algorithm import ProcessAreaDiscovery
from discovery.region_detection import detect_object_centric_regions
from repo.helpers.vorbose import render_object_centric_regions
from totem_lib import import_ocel


ocel = import_ocel("example_data/ContainerLogistics.sqlite")

discovery = ProcessAreaDiscovery(ocel)
discovery.get_layers()
discovery.discover_models()

layer_3_model = discovery.discovered_models[2]
layer_3_regions = detect_object_centric_regions(
    layer_3_model["ocpn"],
    allowed_activities=set(layer_3_model["highlighted_activities"]),
)

layer_3_regions_image = render_object_centric_regions(
    layer_3_model["ocpn"],
    layer_3_regions,
    activity_resource_types=layer_3_model.get("activity_resources"),
    highlighted_activities=layer_3_model.get("highlighted_activities"),
)
if layer_3_regions_image is not None:
    plt.figure(figsize=(max(12, layer_3_regions_image.width / 150), max(7, layer_3_regions_image.height / 150)))
    plt.imshow(layer_3_regions_image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

print(layer_3_model["highlighted_activities"])
print(layer_3_regions)
