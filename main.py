from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo.discovery.algorithm import ProcessAreaDiscovery
from repo.totem_lib import import_ocel


def main():
    ocel = import_ocel("example_data/ContainerLogistics.sqlite")

    discovery = ProcessAreaDiscovery(ocel)
    discovery.get_layers(show_progress=True)
    discovery.discover_models(show_progress=True)
    discovery.visualize()
    discovery.visualize_subprocesses(output_path="output/hierarchy_with_subprocesses.png")


if __name__ == "__main__":
    main()
