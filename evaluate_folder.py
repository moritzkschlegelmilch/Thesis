#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
REPO_PARENT = REPO_ROOT.parent

if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import evaluate


def _normalize_extensions(extensions):
    normalized = []
    for extension in extensions:
        if not extension:
            continue
        normalized.append(extension if extension.startswith(".") else f".{extension}")
    return tuple(sorted(set(extension.lower() for extension in normalized)))


def _collect_input_paths(input_dir, *, recursive, extensions):
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run process-area discovery for every matching log in a folder and "
            "save both the final model visualization and the indexed-subprocess view."
        ),
    )
    parser.add_argument("input_dir", help="Directory containing the input log files.")
    parser.add_argument("output_dir", help="Directory where per-log outputs should be written.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".sqlite", ".xml", ".xmlocel", ".json", ".jsonocel"],
        help="File extensions to include. Defaults to .sqlite, .xml, .xmlocel, .json, and .jsonocel.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for matching files.",
    )
    parser.add_argument(
        "--layer-context",
        type=int,
        nargs="*",
        help="Optional per-layer context values passed to discover_models().",
    )
    parser.add_argument(
        "--precision-context-sample-size",
        type=int,
        default=512,
        help="PPS-with-replacement sample size for precision contexts during pruning. Defaults to 512.",
    )
    parser.add_argument(
        "--precision-context-depth",
        type=int,
        default=5,
        help="Maximum predecessor depth in the event-object graph when preparing precision contexts. Defaults to 5.",
    )
    parser.add_argument(
        "--precision-context-sample-seed",
        type=int,
        help="Optional random seed for precision context sampling.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    extensions = _normalize_extensions(args.extensions)
    input_paths = _collect_input_paths(
        input_dir,
        recursive=args.recursive,
        extensions=extensions,
    )

    if not input_paths:
        parser.error(
            f"No input files found in {input_dir} for extensions: {', '.join(extensions)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress

    file_bar = None
    if show_progress:
        file_bar = tqdm(
            total=len(input_paths),
            desc="Evaluate folder",
            dynamic_ncols=True,
            file=sys.stdout,
        )

    summaries = []
    try:
        for input_path in input_paths:
            if file_bar is not None:
                file_bar.set_postfix_str(input_path.name, refresh=False)

            file_output_dir = output_dir / input_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            summary = evaluate.evaluate_log(
                input_path,
                file_output_dir / "hierarchy.png",
                title=input_path.stem,
                layer_context=args.layer_context,
                precision_context_sample_size=args.precision_context_sample_size,
                precision_context_depth=args.precision_context_depth,
                precision_context_sample_seed=args.precision_context_sample_seed,
                subprocess_output=file_output_dir / "indexed_subprocesses.png",
                show_progress=show_progress,
            )
            summaries.append(summary)

            if file_bar is not None:
                file_bar.update(1)
            else:
                print(
                    f"Processed {input_path.name}: "
                    f"{summary['output_path']} | {summary['subprocess_output']}"
                )
    finally:
        if file_bar is not None:
            file_bar.close()

    print(f"Processed {len(summaries)} files into {output_dir}")


if __name__ == "__main__":
    main()
