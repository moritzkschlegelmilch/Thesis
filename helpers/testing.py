from pathlib import Path

from totem_lib import import_ocel

from repo.discovery.algorithm import ProcessAreaDiscovery
from repo.discovery.totem import clear_totem_cache

SUPPORTED_LOG_SUFFIXES = {
    ".json",
    ".jsonocel",
    ".sqlite",
    ".xml",
    ".xmlocel",
    ".xes",
}


def test_log_folder(input_dir, output_dir, recursive=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    results = []

    for log_path in sorted(input_dir.glob(pattern)):
        print("processing", log_path)
        if not log_path.is_file() or log_path.suffix.lower() not in SUPPORTED_LOG_SUFFIXES:
            continue

        relative_path = log_path.relative_to(input_dir)
        image_path = output_dir / relative_path.with_suffix(".png")
        ocel = None

        try:
            ocel = import_ocel(str(log_path))
            discovery = ProcessAreaDiscovery(ocel)
            print("running discovery for", log_path)
            discovery.run()
            print("visualizing results for", log_path)
            discovery.visualize(title=log_path.stem, output_path=image_path)
            results.append({
                "input_path": str(log_path),
                "output_path": str(image_path),
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "input_path": str(log_path),
                "output_path": str(image_path),
                "status": "error",
                "error": str(exc),
            })
        finally:
            if ocel is not None:
                clear_totem_cache(ocel)

    return results
