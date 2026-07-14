#!/usr/bin/env python3
"""Generate balanced EIS samples with live export and live progress plotting."""

from __future__ import annotations

import sys
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PIPELINE_DIR.parent

if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from generate_data_pipline import DataGen


SOURCE_ECM = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
TARGET_NUM = 150

OUTPUT_DIR = PIPELINE_DIR / "data"

MAX_BATCHES = 150
SEED_START = 2026
N_RANDOM_CANDIDATES = 5000
MAX_SELECTED_CURVES = 150

EXCLUDED_SIMPLIFIED_ECMS = ("R1", "R1-C2")

FIM_REFIT_MAX_ITERS = 100
FIM_REFIT_MIN_ITERS = 1
FIM_REFIT_MAX_NFEV = 100


start_time = time.perf_counter()

generator = DataGen(
    random_ecm_circuit=SOURCE_ECM,
    output_dir=OUTPUT_DIR,
    n_random_candidates=N_RANDOM_CANDIDATES,
    max_selected_curves=MAX_SELECTED_CURVES,
    excluded_simplified_ecms=EXCLUDED_SIMPLIFIED_ECMS,
    fim_refit_max_iters=FIM_REFIT_MAX_ITERS,
    fim_refit_min_iters=FIM_REFIT_MIN_ITERS,
    fim_refit_max_nfev=FIM_REFIT_MAX_NFEV,
    verbose=False,
)

balanced_df, batch_infos = generator.generate_data(
    target_num=TARGET_NUM,
    min_batches=1,
    max_batches=MAX_BATCHES,
    seed_start=SEED_START,
    n_random_candidates=N_RANDOM_CANDIDATES,
    max_selected_curves=MAX_SELECTED_CURVES,
    export=False,
    export_dataprep=True,
    live_plot=True,
)

elapsed = time.perf_counter() - start_time
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)

print(
    f"Total runtime: "
    f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
)
