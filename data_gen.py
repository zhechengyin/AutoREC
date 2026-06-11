"""
AutoREC EIS random generation, relabelling, balancing, and export utilities.

This module contains the non-PCA/non-UMAP workflow from the notebook:
- random ECM parameter sampling
- impedance simulation
- high-frequency filtering
- greedy diverse-curve selection
- simplification + FIM relabelling
- postprocessing/canonicalization
- balanced relabel dataset generation
- CSV / table-data export

Example
-------
from autorec_eis_generator import AutoRECEISGenerator

generator = AutoRECEISGenerator(
    random_ecm_circuit="R1-[P2,R3]-[P4,R5]-[P6,R7]",
    output_dir="data",
    simplification_dir="ecm_simplification_functions",
)

balanced_df, batch_infos = generator.generate_data(
    target_per_relabel=5,
    n_random_candidates=5000,
    max_selected_curves=100,
    max_batches=200,
    seed_start=1000,
    export=True,
)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import autoeis as ae
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from src.autorec.utils import validity_check as default_validity_check
except Exception:
    default_validity_check = None


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Simulating circuit based on initial parameters",
)


DEFAULT_PARAM_BOUNDS = {
    "R": (1e0, 1e6),
    "Pw": (1e-8, 1e-2),
    "Pn": (0.45, 1.0),
    "C": (1e-8, 1e-2),
    "L": (1e-9, 1e1),
}


@dataclass
class DataGen:
    """Generate, relabel, balance, and export EIS data for one source ECM."""

    random_ecm_circuit: str = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
    random_ecm_freq: np.ndarray = field(default_factory=lambda: np.logspace(5, -2, 80))
    output_dir: str | Path = "data"
    simplification_dir: str | Path = "ecm_simplification_functions"
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PARAM_BOUNDS))

    n_random_candidates: int = 5000
    max_selected_curves: int = 100
    random_seed: int = 42
    selection_distance_threshold: Optional[float] = None

    high_frequency_index: int = 0
    max_high_frequency_minus_im_norm: float = 0.1

    fim_fit_ecm: bool = True
    fim_refit_max_iters: int = 5
    fim_refit_min_iters: int = 2
    fim_refit_max_nfev: int = 200
    fim_identifiability_thresh: float = 1e-6

    r1_value: float = 0.01
    drop_pp_series: bool = True
    drop_invalid_ecms: bool = True
    validity_check_fn: Optional[Callable[[str], bool]] = default_validity_check

    verbose: bool = True

    def __post_init__(self) -> None:
        self.random_ecm_freq = np.asarray(self.random_ecm_freq, dtype=float)
        self.output_dir = Path(self.output_dir)
        self.simplification_dir = Path(self.simplification_dir).resolve()

        if str(self.simplification_dir) not in sys.path:
            sys.path.insert(0, str(self.simplification_dir))

        self.ecm_parser_simplifier = importlib.import_module("parser")
        self.drop_ecm_redundancy = importlib.import_module("drop_ecm_redundancy")
        self.full_simplify_redundant_circuit = getattr(
            self.drop_ecm_redundancy,
            "full_simplify_redundant_circuit",
        )

        self.random_ecm_param_names = ae.parser.get_parameter_labels(self.random_ecm_circuit)
        self.random_ecm_fn = ae.utils.generate_circuit_fn(self.random_ecm_circuit)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # General utilities
    # ------------------------------------------------------------------
    @staticmethod
    def source_ecm_filename_slug(circuit: str) -> str:
        source = str(circuit).strip()
        readable = re.sub(r"\s+", "", source)
        readable = re.sub(r"[^A-Za-z0-9._,\[\]-]+", "_", readable).strip("._-")
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        return f"{readable}_{digest}" if readable else f"unknown_ecm_{digest}"

    @staticmethod
    def split_top_level(text: str, sep: str = ",") -> List[str]:
        parts, current, depth = [], [], 0

        for ch in str(text):
            depth += ch == "["
            depth -= ch == "]"

            if ch == sep and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)

        parts.append("".join(current))
        return parts

    @staticmethod
    def _parallel_sort_key(item: str) -> int:
        first = item.strip()[:1]
        if first == "P":
            return 0
        if first == "R":
            return 1
        return 2

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Sampling / simulation / filtering / selection
    # ------------------------------------------------------------------
    def sample_params(
        self,
        n_candidates: Optional[int] = None,
        seed: Optional[int] = None,
        param_names: Optional[Sequence[str]] = None,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[Dict[str, float]]:
        n_candidates = self.n_random_candidates if n_candidates is None else n_candidates
        seed = self.random_seed if seed is None else seed
        param_names = self.random_ecm_param_names if param_names is None else list(param_names)
        param_bounds = self.param_bounds if param_bounds is None else param_bounds

        sampled = ae.utils.sample_circuit_parameters(
            self.random_ecm_circuit,
            bounds=param_bounds,
            seed=seed,
            log=True,
            num_samples=n_candidates,
        )
        sampled = np.atleast_2d(sampled)

        return [
            {name: float(value) for name, value in zip(param_names, row)}
            for row in sampled
        ]

    def params_to_array(
        self,
        params: Dict[str, float],
        param_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        param_names = self.random_ecm_param_names if param_names is None else list(param_names)
        return np.array([params[name] for name in param_names], dtype=float)

    @staticmethod
    def normalize_curve(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        re_part = np.asarray(Z).real
        minus_im = -np.asarray(Z).imag

        re_rng = re_part.max() - re_part.min()
        im_rng = minus_im.max() - minus_im.min()

        re_norm = (re_part - re_part.min()) / re_rng if re_rng > 0 else np.zeros_like(re_part)
        minus_im_norm = (minus_im - minus_im.min()) / im_rng if im_rng > 0 else np.zeros_like(minus_im)

        return re_norm, minus_im_norm

    def circuit_params_to_array(self, circuit: str, params: Dict[str, float]) -> np.ndarray:
        if not isinstance(params, dict):
            raise TypeError("Circuit parameters must be a dict.")

        labels = ae.parser.get_parameter_labels(circuit)
        missing = [name for name in labels if name not in params]
        if missing:
            raise KeyError(f"Missing parameter(s) for {circuit}: {missing}")

        return np.array([params[name] for name in labels], dtype=float)

    def simulate_circuit_with_params(
        self,
        circuit: str,
        params: Dict[str, float],
        frequencies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        frequencies = self.random_ecm_freq if frequencies is None else np.asarray(frequencies, dtype=float)
        circuit_fn = ae.utils.generate_circuit_fn(circuit)
        return circuit_fn(frequencies, self.circuit_params_to_array(circuit, params))

    def simulate_relabel_impedance(self, row: pd.Series, frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        return self.simulate_circuit_with_params(
            row["relabel_ecm"],
            row["relabel_params"],
            frequencies=frequencies,
        )

    def simulate_relabel_normalized_curve(
        self,
        row: pd.Series,
        frequencies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.normalize_curve(self.simulate_relabel_impedance(row, frequencies=frequencies))

    def simulate_candidates(
        self,
        params_list: Sequence[Dict[str, float]],
        circuit_fn: Optional[Callable] = None,
        frequencies: Optional[np.ndarray] = None,
        param_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, float]], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        circuit_fn = self.random_ecm_fn if circuit_fn is None else circuit_fn
        frequencies = self.random_ecm_freq if frequencies is None else np.asarray(frequencies, dtype=float)
        param_names = self.random_ecm_param_names if param_names is None else list(param_names)

        valid_params, valid_Z, valid_curves = [], [], []

        for params in tqdm(params_list, desc="Simulating random candidates"):
            try:
                Z = circuit_fn(frequencies, self.params_to_array(params, param_names=param_names))
            except Exception:
                continue

            if np.all(np.isfinite(Z)):
                valid_params.append(params)
                valid_Z.append(Z)
                valid_curves.append(self.normalize_curve(Z))

        return valid_params, valid_Z, valid_curves

    def filter_high_frequency(
        self,
        params_list: Sequence[Dict[str, float]],
        Z_list: Sequence[np.ndarray],
        curves: Sequence[Tuple[np.ndarray, np.ndarray]],
        high_freq_index: Optional[int] = None,
        max_minus_im_norm: Optional[float] = None,
    ) -> Tuple[List[Dict[str, float]], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        high_freq_index = self.high_frequency_index if high_freq_index is None else high_freq_index
        max_minus_im_norm = (
            self.max_high_frequency_minus_im_norm
            if max_minus_im_norm is None
            else max_minus_im_norm
        )

        values = np.array([float(curve[1][high_freq_index]) for curve in curves])
        keep_idx = np.flatnonzero(values <= max_minus_im_norm)

        if len(keep_idx) == 0:
            raise ValueError("High-frequency filter removed every candidate.")

        info = {
            "raw_valid_candidate_count": len(params_list),
            "filtered_candidate_count": len(keep_idx),
            "removed_candidate_count": int(len(params_list) - len(keep_idx)),
            "high_frequency_index": high_freq_index,
            "max_high_frequency_minus_im_norm": max_minus_im_norm,
            "kept_original_idx": keep_idx.tolist(),
            "high_frequency_minus_im_norm": values.tolist(),
        }

        return (
            [params_list[i] for i in keep_idx],
            [Z_list[i] for i in keep_idx],
            [curves[i] for i in keep_idx],
            info,
        )

    @staticmethod
    def stack_curves(curves: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        return np.stack([np.column_stack(curve) for curve in curves]).astype(np.float32)

    @staticmethod
    def mean_curve_distance(curve_points: np.ndarray, reference_curve: np.ndarray) -> np.ndarray:
        return np.linalg.norm(curve_points - reference_curve[None, :, :], axis=2).mean(axis=1)

    def greedy_select(
        self,
        params_list: Sequence[Dict[str, float]],
        Z_list: Sequence[np.ndarray],
        curves: Sequence[Tuple[np.ndarray, np.ndarray]],
        k: Optional[int] = None,
        min_distance: Optional[float] = None,
    ) -> Tuple[List[Dict[str, float]], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], List[int], List[float]]:
        k = self.max_selected_curves if k is None else k
        min_distance = self.selection_distance_threshold if min_distance is None else min_distance

        if not curves or k <= 0:
            return [], [], [], [], []

        curve_points = self.stack_curves(curves)
        first = int(np.argmax(self.mean_curve_distance(curve_points, curve_points.mean(axis=0))))

        selected, scores = [first], [np.inf]
        min_dist = self.mean_curve_distance(curve_points, curve_points[first])
        min_dist[first] = -np.inf

        while len(selected) < min(k, len(curves)):
            nxt, score = int(np.argmax(min_dist)), float(np.max(min_dist))

            if min_distance is not None and score < min_distance:
                break

            selected.append(nxt)
            scores.append(score)

            min_dist = np.minimum(min_dist, self.mean_curve_distance(curve_points, curve_points[nxt]))
            min_dist[selected] = -np.inf

        return (
            [params_list[i] for i in selected],
            [Z_list[i] for i in selected],
            [curves[i] for i in selected],
            selected,
            scores,
        )

    def generate_filter_select(
        self,
        seed: Optional[int] = None,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
    ) -> Dict[str, Any]:
        candidates = self.sample_params(n_candidates=n_random_candidates, seed=seed)
        params, Z, curves = self.simulate_candidates(candidates)

        filtered_params, filtered_Z, filtered_curves, filter_info = self.filter_high_frequency(
            params,
            Z,
            curves,
        )

        selected_params, selected_Z, selected_curves, selected_idx, selected_scores = self.greedy_select(
            filtered_params,
            filtered_Z,
            filtered_curves,
            k=max_selected_curves,
        )

        return {
            "candidates": candidates,
            "params": params,
            "Z": Z,
            "curves": curves,
            "filtered_params": filtered_params,
            "filtered_Z": filtered_Z,
            "filtered_curves": filtered_curves,
            "filter_info": filter_info,
            "selected_params": selected_params,
            "selected_Z": selected_Z,
            "selected_curves": selected_curves,
            "selected_idx": selected_idx,
            "selected_scores": selected_scores,
        }

    # ------------------------------------------------------------------
    # Relabelling and postprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def choose_fim_relabel(
        fim_result: Any,
        fallback_circuit: str,
        fallback_params: Dict[str, float],
    ) -> Tuple[str, Optional[Dict[str, float]], Any]:
        if isinstance(fim_result, str):
            return fim_result, fallback_params, []

        if isinstance(fim_result, list) and fim_result:
            first = fim_result[0]
            if isinstance(first, tuple):
                return first[0], first[1], fim_result
            return first, None, fim_result

        return fallback_circuit, fallback_params, fim_result

    def simplify_relabel_circuit(
        self,
        circuit: Optional[str],
        params: Optional[Dict[str, float]],
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        if not circuit:
            return circuit, params

        try:
            if isinstance(params, dict) and params:
                simplified = self.ecm_parser_simplifier.simplify(circuit, params)
            else:
                simplified = self.ecm_parser_simplifier.simplify(circuit)

            return simplified if isinstance(simplified, tuple) else (simplified, params)

        except Exception:
            return circuit, params

    def run_relabel(
        self,
        params_list: Sequence[Dict[str, float]],
        Z_list: Sequence[np.ndarray],
        curves: Sequence[Tuple[np.ndarray, np.ndarray]],
        batch_seed: Optional[int] = None,
        batch_id: Optional[int] = None,
    ) -> pd.DataFrame:
        rows = []

        for selected_position, (params, Z, curve) in enumerate(tqdm(
            zip(params_list, Z_list, curves),
            total=len(params_list),
            desc="Simplify + FIM relabel selected curves",
        )):
            record = {
                "batch_id": batch_id,
                "batch_seed": batch_seed,
                "selected_position": selected_position,
                "original_ecm": self.random_ecm_circuit,
                "simplified_ecm": None,
                "fim_relabel_ecm": None,
                "post_fim_simplified_ecm": None,
                "relabel_ecm": None,
                "params": params,
                "simplified_params": None,
                "fim_relabel_params": None,
                "post_fim_simplified_params": None,
                "relabel_params": None,
                "fim_candidates": None,
                "Z": Z,
                "curve": curve,
                "relabel_failed": False,
                "failure_reason": None,
            }

            try:
                simplified_ecm, simplified_params = self.ecm_parser_simplifier.simplify(
                    self.random_ecm_circuit,
                    params,
                )

                fim_result = self.full_simplify_redundant_circuit(
                    simplified_ecm,
                    self.random_ecm_freq,
                    Z,
                    simplified_params,
                    refit_ecm=self.fim_fit_ecm,
                    identifiability_thresh=self.fim_identifiability_thresh,
                    fit_kwargs={
                        "max_iters": self.fim_refit_max_iters,
                        "min_iters": self.fim_refit_min_iters,
                        "max_nfev": self.fim_refit_max_nfev,
                    },
                    verbose=False,
                )

                fim_relabel_ecm, fim_relabel_params, fim_candidates = self.choose_fim_relabel(
                    fim_result,
                    simplified_ecm,
                    simplified_params,
                )

                relabel_ecm, relabel_params = self.simplify_relabel_circuit(
                    fim_relabel_ecm,
                    fim_relabel_params,
                )

                record.update({
                    "simplified_ecm": simplified_ecm,
                    "fim_relabel_ecm": fim_relabel_ecm,
                    "post_fim_simplified_ecm": relabel_ecm,
                    "relabel_ecm": relabel_ecm,
                    "simplified_params": simplified_params,
                    "fim_relabel_params": fim_relabel_params,
                    "post_fim_simplified_params": relabel_params,
                    "relabel_params": relabel_params,
                    "fim_candidates": fim_candidates,
                })

            except Exception as exc:
                record.update({
                    "simplified_ecm": self.random_ecm_circuit,
                    "fim_relabel_ecm": self.random_ecm_circuit,
                    "post_fim_simplified_ecm": self.random_ecm_circuit,
                    "relabel_ecm": self.random_ecm_circuit,
                    "relabel_failed": True,
                    "failure_reason": repr(exc),
                })

            rows.append(record)

        return pd.DataFrame(rows)

    def has_series_p_chain(self, circuit: Optional[str]) -> bool:
        if circuit is None or pd.isna(circuit):
            return False

        parts = [p.strip() for p in self.split_top_level(str(circuit), sep="-")]

        for i in range(len(parts) - 1):
            if re.fullmatch(r"P\d+", parts[i]) and re.fullmatch(r"P\d+", parts[i + 1]):
                return True

        return False

    def drop_series_p_chain_after_relabel(
        self,
        df: pd.DataFrame,
        circuit_col: str = "relabel_ecm",
    ) -> pd.DataFrame:
        df = df.copy()
        bad_mask = df[circuit_col].apply(self.has_series_p_chain)

        self._log(f"Dropping {int(bad_mask.sum())} final relabelled ECMs with P-P series")
        self._log(f"Keeping {int((~bad_mask).sum())} final relabelled ECMs")

        return df.loc[~bad_mask].reset_index(drop=True)

    def drop_invalid_relabelled_ecms(
        self,
        df: pd.DataFrame,
        circuit_col: str = "relabel_ecm",
    ) -> pd.DataFrame:
        if self.validity_check_fn is None:
            self._log("Skipping validity_check because no validity_check_fn is available.")
            return df.reset_index(drop=True)

        df = df.copy()

        def safe_validity_check(circuit: Any) -> bool:
            try:
                if circuit is None or pd.isna(circuit):
                    return False
                return bool(self.validity_check_fn(str(circuit)))
            except Exception:
                return False

        valid_mask = df[circuit_col].apply(safe_validity_check)

        self._log(f"Dropping {int((~valid_mask).sum())} invalid relabelled ECMs")
        self._log(f"Keeping {int(valid_mask.sum())} valid relabelled ECMs")

        return df.loc[valid_mask].reset_index(drop=True)

    def reorder_parallel_blocks_and_series_p(self, circuit: Optional[str]) -> Optional[str]:
        if circuit is None:
            return circuit

        circuit = str(circuit)
        out, i = [], 0

        while i < len(circuit):
            if circuit[i] != "[":
                out.append(circuit[i])
                i += 1
                continue

            start, depth, i = i, 1, i + 1

            while i < len(circuit) and depth:
                depth += circuit[i] == "["
                depth -= circuit[i] == "]"
                i += 1

            items = self.split_top_level(circuit[start + 1:i - 1])
            items = [self.reorder_parallel_blocks_and_series_p(item) for item in items]
            items = sorted(items, key=self._parallel_sort_key)
            out.append("[" + ",".join(items) + "]")

        circuit = "".join(out)
        series_parts = self.split_top_level(circuit, sep="-")

        p_parts = [
            part for part in series_parts
            if re.fullmatch(r"\s*P\d+\s*", part)
        ]
        non_p_parts = [
            part for part in series_parts
            if not re.fullmatch(r"\s*P\d+\s*", part)
        ]

        return "-".join(non_p_parts + p_parts)

    @staticmethod
    def convert_capacitors_to_cpes(
        circuit: Optional[str],
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        if circuit is None:
            return circuit, params

        circuit = re.sub(r"C(\d+)", r"P\1", str(circuit))

        if not isinstance(params, dict):
            return circuit, params

        converted = {}

        for key, value in params.items():
            key = str(key)
            match = re.fullmatch(r"C(\d+)", key)

            if match:
                cpe_base = f"P{match.group(1)}"
                converted[f"{cpe_base}w"] = value
                converted[f"{cpe_base}n"] = 1.0
            else:
                converted[key] = value

        return circuit, converted

    def ensure_series_r1(
        self,
        circuit: Optional[str],
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        if circuit is None:
            return circuit, params

        circuit = str(circuit)
        series_parts = self.split_top_level(circuit, sep="-")

        has_front_series_resistor = bool(
            series_parts and re.fullmatch(r"\s*R\d+\s*", series_parts[0])
        )

        if has_front_series_resistor:
            old_front = series_parts[0].strip()
            circuit = "R1-" + "-".join(series_parts[1:]) if len(series_parts) > 1 else "R1"

            if isinstance(params, dict):
                params = dict(params)
                params.pop(old_front, None)
                params["R1"] = self.r1_value

            return circuit, params

        circuit = "R1-" + circuit if circuit else "R1"

        if isinstance(params, dict):
            params = dict(params)
            params["R1"] = self.r1_value

        return circuit, params

    @staticmethod
    def reindex_parameter_dict(
        params: Optional[Dict[str, float]],
        mapping: Dict[str, str],
    ) -> Optional[Dict[str, float]]:
        if not isinstance(params, dict):
            return params

        renamed = {}

        for key, value in params.items():
            match = re.match(r"([A-Z]\d+)(.*)", str(key))

            if match:
                new_key = mapping.get(match.group(1), match.group(1)) + match.group(2)
            else:
                new_key = key

            renamed[new_key] = value

        return renamed

    @staticmethod
    def reindex_components(circuit: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if circuit is None:
            return circuit, {}

        circuit = str(circuit)

        mapping = {}
        used_names = {"R1"} if re.search(r"\bR1\b", circuit) else set()
        counters = {"R": 2}

        for match in re.finditer(r"([A-Z])(\d+)", circuit):
            old_name = match.group(0)
            prefix = match.group(1)

            if old_name == "R1":
                mapping[old_name] = "R1"
                continue

            if old_name in mapping:
                continue

            counters.setdefault(prefix, 1)

            while f"{prefix}{counters[prefix]}" in used_names:
                counters[prefix] += 1

            new_name = f"{prefix}{counters[prefix]}"
            mapping[old_name] = new_name
            used_names.add(new_name)
            counters[prefix] += 1

        new_circuit = re.sub(
            r"([A-Z])(\d+)",
            lambda match: mapping[match.group(0)],
            circuit,
        )

        return new_circuit, mapping

    def normalize_circuit_and_params(
        self,
        circuit: Optional[str],
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        if circuit is None or pd.isna(circuit):
            return circuit, params

        circuit, params = self.convert_capacitors_to_cpes(circuit, params)
        circuit = self.reorder_parallel_blocks_and_series_p(circuit)
        circuit, params = self.ensure_series_r1(circuit, params)

        circuit, mapping = self.reindex_components(circuit)
        params = self.reindex_parameter_dict(params, mapping)

        if isinstance(params, dict):
            params = dict(params)
            params["R1"] = self.r1_value

        return circuit, params

    def postprocess_relabels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for circuit_col, params_col in (
            ("simplified_ecm", "simplified_params"),
            ("fim_relabel_ecm", "fim_relabel_params"),
            ("post_fim_simplified_ecm", "post_fim_simplified_params"),
            ("relabel_ecm", "relabel_params"),
        ):
            if circuit_col not in df.columns:
                continue

            normalized = [
                self.normalize_circuit_and_params(
                    row[circuit_col],
                    row.get(params_col) if params_col in df.columns else None,
                )
                for _, row in df.iterrows()
            ]

            df[circuit_col] = [item[0] for item in normalized]

            if params_col in df.columns:
                df[params_col] = [item[1] for item in normalized]

        return df

    # ------------------------------------------------------------------
    # Balanced dataset generation
    # ------------------------------------------------------------------
    @staticmethod
    def relabel_group_counts(df: pd.DataFrame, target_labels: Optional[Sequence[str]] = None) -> pd.Series:
        source = df.dropna(subset=["relabel_ecm"]).copy()

        if "relabel_failed" in source.columns:
            source = source.loc[~source["relabel_failed"]]

        counts = source["relabel_ecm"].value_counts().sort_index()

        if target_labels is not None:
            counts = counts.reindex(target_labels, fill_value=0)

        return counts

    def run_selected_relabel_batch(
        self,
        seed: int,
        batch_id: int,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        n_random_candidates = self.n_random_candidates if n_random_candidates is None else n_random_candidates
        max_selected_curves = self.max_selected_curves if max_selected_curves is None else max_selected_curves

        candidates = self.sample_params(
            n_candidates=n_random_candidates,
            seed=seed,
        )

        params, Z, curves = self.simulate_candidates(candidates)

        filtered_params, filtered_Z, filtered_curves, filter_info = self.filter_high_frequency(
            params,
            Z,
            curves,
        )

        selected_params, selected_Z, selected_curves, selected_idx, selected_scores = self.greedy_select(
            filtered_params,
            filtered_Z,
            filtered_curves,
            k=max_selected_curves,
        )

        batch_df = self.run_relabel(
            selected_params,
            selected_Z,
            selected_curves,
            batch_seed=seed,
            batch_id=batch_id,
        )

        batch_df = self.postprocess_relabels(batch_df)

        if self.drop_pp_series:
            batch_df = self.drop_series_p_chain_after_relabel(
                batch_df,
                circuit_col="relabel_ecm",
            )

        if self.drop_invalid_ecms:
            batch_df = self.drop_invalid_relabelled_ecms(
                batch_df,
                circuit_col="relabel_ecm",
            )

        batch_df["selected_position_in_batch"] = batch_df["selected_position"]

        selected_idx_map = dict(enumerate(selected_idx))
        selected_score_map = dict(enumerate(selected_scores))

        batch_df["selected_filtered_idx"] = batch_df["selected_position"].map(selected_idx_map)
        batch_df["selection_score"] = batch_df["selected_position"].map(selected_score_map)

        batch_info = {
            "batch_id": batch_id,
            "seed": seed,
            "valid_candidate_count": len(params),
            "filtered_candidate_count": len(filtered_params),
            "selected_curve_count": len(selected_curves),
            "after_final_filters_count": len(batch_df),
            "removed_by_high_frequency_filter": filter_info["removed_candidate_count"],
        }

        return batch_df, batch_info

    def take_needed_target_rows(
        self,
        batch_df: pd.DataFrame,
        counts: pd.Series,
        target_labels: Sequence[str],
        target_per_relabel: int,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        kept = []

        usable = batch_df.dropna(subset=["relabel_ecm"]).copy()

        if "relabel_failed" in usable.columns:
            usable = usable.loc[~usable["relabel_failed"]]

        for label in target_labels:
            needed = target_per_relabel - int(counts.get(label, 0))

            if needed <= 0:
                continue

            label_rows = usable.loc[usable["relabel_ecm"] == label]

            if not label_rows.empty:
                selected = label_rows.head(needed)
                kept.append(selected)
                counts[label] = int(counts.get(label, 0)) + len(selected)

        if not kept:
            return batch_df.iloc[0:0].copy(), counts

        return pd.concat(kept, ignore_index=True), counts

    def run_balanced_relabel_dataset(
        self,
        target_per_relabel: int = 5,
        target_relabel_ecms: Optional[Sequence[str]] = None,
        min_batches: int = 1,
        max_batches: int = 200,
        seed_start: int = 1000,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        use_manual_targets = target_relabel_ecms is not None
        target_labels = sorted(set(target_relabel_ecms)) if use_manual_targets else None

        kept_batches = []
        batch_infos = []
        counts = pd.Series(dtype=int)

        for batch_id in range(max_batches):
            seed = seed_start + batch_id

            batch_df, batch_info = self.run_selected_relabel_batch(
                seed=seed,
                batch_id=batch_id,
                n_random_candidates=n_random_candidates,
                max_selected_curves=max_selected_curves,
            )
            batch_infos.append(batch_info)

            discovered_labels = set(self.relabel_group_counts(batch_df).index)

            if target_labels is None:
                target_labels = sorted(discovered_labels)
                counts = pd.Series(0, index=target_labels, dtype=int)
            elif counts.empty:
                counts = pd.Series(0, index=target_labels, dtype=int)

            before_counts = counts.copy()

            kept_df, counts = self.take_needed_target_rows(
                batch_df,
                counts,
                target_labels,
                target_per_relabel,
            )

            if not kept_df.empty:
                kept_batches.append(kept_df)

            added_counts = counts.subtract(before_counts, fill_value=0).astype(int)
            under_target = counts[counts < target_per_relabel]
            target_note = "manual target" if use_manual_targets else "first-batch target"

            self._log(
                f"Batch {batch_id + 1}: {target_note} groups {len(counts)}, "
                f"discovered groups {len(discovered_labels)}"
            )
            self._log("Added this batch:")
            self._log(added_counts.to_string())
            self._log("Accumulated target counts:")
            self._log(counts.to_string())

            if batch_id + 1 >= min_batches and under_target.empty and len(counts) > 0:
                return pd.concat(kept_batches, ignore_index=True), batch_infos

        raise RuntimeError(
            f"Balanced relabel dataset did not reach {target_per_relabel} per group "
            f"within {max_batches} batches. Current counts:\n{counts.to_string()}"
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def build_final_relabel_df(
        self,
        balanced_relabel_results_df: pd.DataFrame,
        target_per_relabel: int,
        target_relabel_ecms: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        source_df = balanced_relabel_results_df.dropna(subset=["relabel_ecm"]).copy()
        source_df = source_df.loc[~source_df["relabel_failed"]]

        if target_relabel_ecms is not None:
            source_df = source_df.loc[source_df["relabel_ecm"].isin(target_relabel_ecms)]

        final_df = (
            source_df
            .sort_values(["relabel_ecm", "batch_id", "selected_position"])
            .groupby("relabel_ecm", group_keys=False)
            .head(target_per_relabel)
            .reset_index(drop=True)
        )

        final_df.insert(0, "global_position", np.arange(len(final_df)))

        final_counts = self.relabel_group_counts(final_df)
        missing = final_counts[final_counts < target_per_relabel]

        if not missing.empty:
            raise RuntimeError(f"Final balanced export still has under-target groups:\n{missing.to_string()}")

        return final_df

    def build_curve_parameter_export_df(self, final_relabel_df: pd.DataFrame) -> pd.DataFrame:
        frequency_json = json.dumps([float(value) for value in self.random_ecm_freq])
        rows = []

        for _, row in final_relabel_df.sort_values(["relabel_ecm", "global_position"]).iterrows():
            original_params = (
                {str(k): float(v) for k, v in row["params"].items()}
                if isinstance(row["params"], dict)
                else {}
            )
            relabel_params = (
                {str(k): float(v) for k, v in row["relabel_params"].items()}
                if isinstance(row["relabel_params"], dict)
                else {}
            )

            Z = np.asarray(row["Z"])

            rows.append({
                "original_ecm": row["original_ecm"],
                "original_params_json": json.dumps(original_params, sort_keys=True),
                "final_simplified_ecm": row["relabel_ecm"],
                "final_simplified_params_json": json.dumps(relabel_params, sort_keys=True),
                "frequency_hz_json": frequency_json,
                "impedance_data_json": json.dumps({
                    "real_ohm": [float(value) for value in np.real(Z)],
                    "imag_ohm": [float(value) for value in np.imag(Z)],
                }),
            })

        return pd.DataFrame(rows)

    def export_table_csv(
        self,
        balanced_relabel_results_df: pd.DataFrame,
        target_per_relabel: int = 5,
        target_relabel_ecms: Optional[Sequence[str]] = None,
        csv_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
        final_df = self.build_final_relabel_df(
            balanced_relabel_results_df,
            target_per_relabel=target_per_relabel,
            target_relabel_ecms=target_relabel_ecms,
        )

        export_df = self.build_curve_parameter_export_df(final_df)

        if csv_path is None:
            slug = self.source_ecm_filename_slug(self.random_ecm_circuit)
            csv_path = self.output_dir / f"balanced_final_params_{slug}.csv"
        else:
            csv_path = Path(csv_path)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(csv_path, index=False)

        return final_df, export_df, csv_path

    def export_eis_plot_zip(
        self,
        final_relabel_df: pd.DataFrame,
        zip_path: Optional[str | Path] = None,
    ) -> Path:
        import matplotlib.pyplot as plt

        if zip_path is None:
            slug = self.source_ecm_filename_slug(self.random_ecm_circuit)
            zip_path = self.output_dir / f"balanced_eis_plots_{slug}.zip"
        else:
            zip_path = Path(zip_path)

        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for _, row in final_relabel_df.sort_values(["relabel_ecm", "global_position"]).iterrows():
                Z = np.asarray(self.simulate_relabel_impedance(row, self.random_ecm_freq))
                label = str(row["relabel_ecm"])
                safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "unknown_ecm"

                fig, ax = plt.subplots(figsize=(5.8, 5.2))
                ax.plot(np.real(Z), -np.imag(Z), linewidth=1.8)
                ax.set_title(label, fontsize=10, wrap=True)
                ax.set_xlabel("Re(Z) / ohm")
                ax.set_ylabel("-Im(Z) / ohm")
                ax.grid(True, linestyle=":", alpha=0.35)
                ax.set_aspect("equal", adjustable="datalim")
                fig.tight_layout()

                with BytesIO() as buffer:
                    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
                    archive.writestr(
                        f"curve_{int(row['global_position']):04d}_{safe_label}.png",
                        buffer.getvalue(),
                    )

                plt.close(fig)

        return zip_path

    def export_dataprep_folder(
        self,
        final_relabel_df: pd.DataFrame,
        output_dir: Optional[str | Path] = None,
        use_relabel_simulation: bool = True,
    ) -> Path:
        """Export one CSV per curve under output_dir/<final_ecm>/.

        CSV schema:
            freq, Z_img, Z_real

        This is meant to match the common AutoREC DataPrep pattern where each
        final ECM label has a subdirectory containing individual EIS curve CSVs.
        """

        output_dir = self.output_dir if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for _, row in final_relabel_df.sort_values(["relabel_ecm", "global_position"]).iterrows():
            label = str(row["relabel_ecm"])
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "unknown_ecm"
            label_dir = output_dir / safe_label
            label_dir.mkdir(parents=True, exist_ok=True)

            if use_relabel_simulation:
                Z = np.asarray(self.simulate_relabel_impedance(row, self.random_ecm_freq))
            else:
                Z = np.asarray(row["Z"])

            curve_df = pd.DataFrame({
                "freq": self.random_ecm_freq.astype(float),
                "Z_img": np.imag(Z).astype(float),
                "Z_real": np.real(Z).astype(float),
            })

            curve_df.to_csv(
                label_dir / f"curve_{int(row['global_position']):04d}.csv",
                index=False,
            )

        return output_dir

    def generate_data(
        self,
        target_per_relabel: int = 5,
        target_relabel_ecms: Optional[Sequence[str]] = None,
        min_batches: int = 1,
        max_batches: int = 200,
        seed_start: int = 1000,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
        export: bool = True,
        export_plots: bool = False,
        export_dataprep: bool = False,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Run balanced generation + relabelling.

        Returns
        -------
        balanced_relabel_results_df, batch_infos
        """

        balanced_df, batch_infos = self.run_balanced_relabel_dataset(
            target_per_relabel=target_per_relabel,
            target_relabel_ecms=target_relabel_ecms,
            min_batches=min_batches,
            max_batches=max_batches,
            seed_start=seed_start,
            n_random_candidates=n_random_candidates,
            max_selected_curves=max_selected_curves,
        )

        counts = self.relabel_group_counts(balanced_df)
        self._log("Generated relabel counts before final export trimming:")
        self._log(counts.to_string())
        self._log(f"Generated rows: {len(balanced_df)}")

        if export:
            final_df, export_df, csv_path = self.export_table_csv(
                balanced_df,
                target_per_relabel=target_per_relabel,
                target_relabel_ecms=target_relabel_ecms,
            )
            self._log(f"Saved table CSV: {csv_path}")

            if export_plots:
                zip_path = self.export_eis_plot_zip(final_df)
                self._log(f"Saved EIS plot zip: {zip_path}")

            if export_dataprep:
                dataprep_dir = self.export_dataprep_folder(final_df)
                self._log(f"Saved DataPrep folder: {dataprep_dir}")

        return balanced_df, batch_infos
