#!/usr/bin/env python3
"""Generate balanced EIS samples without using a Jupyter notebook.

The script can be launched from any working directory because project paths
are resolved relative to this file. Each accepted sample is written
immediately under ``<output_dir>/<relabel_ecm>/csv`` and
``<output_dir>/<relabel_ecm>/png``.

Example
-------
python generate_data_pipline/generate_samples.py

Custom target counts can be supplied repeatedly::

    python generate_data_pipline/generate_samples.py \
        --target 'R1-[P1,R2]=55' \
        --target 'R1-[P1,R2]-P2=100' \
        --target 'R1-[P1,R2]-[P2,R3]=100'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple


PIPELINE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PIPELINE_DIR.parent

if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))


DEFAULT_TARGET_COUNTS = {
    "R1-[P1,R2]": 55,
    "R1-[P1,R2]-P2": 100,
    "R1-[P1,R2]-[P2,R3]": 100,
}


def parse_target(value: str) -> Tuple[str, int]:
    """Parse one ``ECM=COUNT`` command-line target.

    Parameters
    ----------
    value : str
        Target specification containing an ECM label and positive count.

    Returns
    -------
    tuple of str and int
        Parsed ECM label and target count.

    Raises
    ------
    argparse.ArgumentTypeError
        If the specification is malformed or the count is not positive.
    """
    try:
        label, count_text = value.rsplit("=", maxsplit=1)
        count = int(count_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected ECM=COUNT, received {value!r}"
        ) from exc

    label = label.strip()
    if not label or count < 1:
        raise argparse.ArgumentTypeError(
            "The ECM label must be non-empty and COUNT must be at least 1"
        )

    return label, count


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for data generation.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for generation settings.
    """
    parser = argparse.ArgumentParser(
        description="Generate balanced AutoREC EIS samples into per-ECM folders."
    )
    parser.add_argument(
        "--source-ecm",
        default="R1-[P2,R3]-[P4,R5]",
        help="Source ECM used for random parameter sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PIPELINE_DIR / "data",
        help="Output root. Defaults to generate_data_pipline/data.",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        metavar="ECM=COUNT",
        help="Target count for one final ECM. Repeat for multiple ECMs.",
    )
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--random-candidates", type=int, default=5000)
    parser.add_argument("--selected-curves", type=int, default=150)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run balanced generation and immediate per-sample export.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments excluding the executable name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_argument_parser().parse_args(argv)

    from generate_data_pipline import DataGen

    target_counts = dict(args.target) if args.target else DEFAULT_TARGET_COUNTS.copy()
    output_dir = args.output_dir.expanduser().resolve()

    generator = DataGen(
        random_ecm_circuit=args.source_ecm,
        output_dir=output_dir,
        n_random_candidates=args.random_candidates,
        max_selected_curves=args.selected_curves,
        excluded_simplified_ecms=("R1", "R1-C2"),
        fim_refit_max_iters=10,
        fim_refit_min_iters=1,
        fim_refit_max_nfev=100,
        verbose=True,
    )

    balanced_df, batch_infos = generator.generate_data(
        target_per_relabel=target_counts,
        target_relabel_ecms=tuple(target_counts),
        min_batches=1,
        max_batches=args.max_batches,
        seed_start=args.seed_start,
        n_random_candidates=args.random_candidates,
        max_selected_curves=args.selected_curves,
        export=False,
        export_dataprep=True,
    )

    generated_counts = generator.relabel_group_counts(
        balanced_df,
        target_labels=tuple(target_counts),
    )
    print(f"\nGeneration finished after {len(batch_infos)} batch(es).")
    print(f"Data directory: {output_dir}")
    print(generated_counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
