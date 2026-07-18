"""
AutoREC EIS random generation, relabelling, balancing, and export utilities.

This module contains the non-PCA/non-UMAP workflow from the notebook:
- random ECM parameter sampling
- impedance simulation
- high-frequency filtering
- greedy diverse-curve selection
- parser full simplification and final relabelling
- postprocessing/canonicalization
- balanced relabel dataset generation
- CSV / table-data export

Example
-------
from generate_data_pipline.data_gen import DataGen

generator = DataGen(
    random_ecm_circuit="R1-[P2,R3]-[P4,R5]-[P6,R7]",
    output_dir="data",
)

balanced_df, batch_infos = generator.generate_data(
    target_num=150,
    n_random_candidates=5000,
    max_selected_curves=100,
    max_batches=200,
    seed_start=1000,
    export=True,
)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import autoeis as ae
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from autorec import parser as autorec_parser

try:
    from autorec.utils import validity_check as default_validity_check
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

DEFAULT_GENERATION_FIGSIZE = (7.5, 5.0)

PIPELINE_DIR = Path(__file__).resolve().parent


class DataGen:
    """Generate, relabel, balance, and export EIS data for one source ECM.

    The class also owns the CLI entry-point helpers used by the standalone
    sample-generation script.

    The class wraps the notebook workflow for random ECM parameter sampling,
    impedance simulation, curve filtering, diverse curve selection, parser
    full simplification, final relabelling, balanced row collection, and
    export.

    Parameters
    ----------
    random_ecm_circuit : str, default="R1-[P2,R3]-[P4,R5]-[P6,R7]"
        Source ECM used to sample random circuit parameters and simulate EIS
        curves.
    random_ecm_freq : numpy.ndarray, optional
        Frequencies in Hz used for simulation and fitting.
    output_dir : str or pathlib.Path, default="data"
        Directory where CSV, plot, and dataprep exports are written.
    param_bounds : dict, optional
        Sampling bounds keyed by component type.
    n_random_candidates : int, default=5000
        Number of random parameter sets sampled per batch.
    max_selected_curves : int, default=100
        Maximum number of diverse curves selected per batch before relabelling.
    random_seed : int, default=42
        Default random seed used when a method does not receive an explicit
        seed.
    selection_distance_threshold : float, optional
        Minimum greedy-selection distance. If ``None``, selection continues
        until ``max_selected_curves`` or all candidates are exhausted.
    high_frequency_index : int, default=0
        Index used for high-frequency endpoint filtering.
    max_high_frequency_minus_im_norm : float, default=0.1
        Maximum allowed normalized ``-Im(Z)`` at ``high_frequency_index``.
    fim_fit_ecm : bool, default=True
        Whether FIM-simplified circuits are refit to obtain updated parameters.
    fim_refit_max_iters : int, default=5
        Maximum outer iterations passed to AutoEIS refitting.
    fim_refit_min_iters : int, default=2
        Minimum outer iterations passed to AutoEIS refitting.
    fim_refit_max_nfev : int, default=200
        Maximum function evaluations passed to AutoEIS refitting.
    fim_identifiability_thresh : float, default=1e-5
        Eigenvalue threshold used by FIM redundancy analysis.
    final_consistency_max_iters : int, default=10
        Maximum number of final-circuit simplification checks performed after
        canonicalization.
    r1_value : float, default=1e-4
        Value assigned to the generated source ``R1`` and to a newly inserted
        canonical ``R1``. Existing fitted series-resistor values are preserved.
    drop_pp_series : bool, default=True
        Whether final relabelled ECMs with series P-P chains are removed.
    drop_invalid_ecms : bool, default=True
        Whether final relabelled ECMs failing ``validity_check_fn`` are removed.
    excluded_simplified_ecms : sequence of str or None, default=("R1", "R1-C2")
        Final simplified/relabelled ECM labels to exclude from generated data.
        Pass an empty sequence to disable this filter.
    validity_check_fn : callable, optional
        Function receiving a circuit string and returning whether it is valid.
    verbose : bool, default=True
        Whether progress messages are printed.
    """

    def __init__(
        self,
        random_ecm_circuit: str = "R1-[P2,R3]-[P4,R5]-[P6,R7]",
        random_ecm_freq: Optional[np.ndarray] = None,
        output_dir: str | Path = "data",
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_random_candidates: int = 5000,
        max_selected_curves: int = 100,
        random_seed: int = 42,
        selection_distance_threshold: Optional[float] = None,
        high_frequency_index: int = 0,
        max_high_frequency_minus_im_norm: float = 0.01,
        fim_fit_ecm: bool = True,
        fim_refit_max_iters: int = 5,
        fim_refit_min_iters: int = 2,
        fim_refit_max_nfev: int = 200,
        fim_identifiability_thresh: float = 1e-5,
        final_consistency_max_iters: int = 10,
        r1_value: float = 1e-4,
        drop_pp_series: bool = True,
        drop_invalid_ecms: bool = True,
        excluded_simplified_ecms: Optional[Sequence[str]] = ("R1", "R1-C2"),
        validity_check_fn: Optional[Callable[[str], bool]] = default_validity_check,
        verbose: bool = True,
    ) -> None:
        """Initialize the data generator and cache source-circuit helpers."""
        self.random_ecm_circuit = random_ecm_circuit
        self.random_ecm_freq = np.asarray(
            np.logspace(5, -2, 80) if random_ecm_freq is None else random_ecm_freq,
            dtype=float,
        )
        self.output_dir = Path(output_dir)
        self.param_bounds = dict(
            DEFAULT_PARAM_BOUNDS if param_bounds is None else param_bounds
        )

        self.n_random_candidates = n_random_candidates
        self.max_selected_curves = max_selected_curves
        self.random_seed = random_seed
        self.selection_distance_threshold = selection_distance_threshold

        self.high_frequency_index = high_frequency_index
        self.max_high_frequency_minus_im_norm = max_high_frequency_minus_im_norm

        self.fim_fit_ecm = fim_fit_ecm
        self.fim_refit_max_iters = fim_refit_max_iters
        self.fim_refit_min_iters = fim_refit_min_iters
        self.fim_refit_max_nfev = fim_refit_max_nfev
        self.fim_identifiability_thresh = fim_identifiability_thresh
        if final_consistency_max_iters < 1:
            raise ValueError("final_consistency_max_iters must be at least 1")
        self.final_consistency_max_iters = final_consistency_max_iters

        self.r1_value = r1_value
        self.drop_pp_series = drop_pp_series
        self.drop_invalid_ecms = drop_invalid_ecms
        self.excluded_simplified_ecms = excluded_simplified_ecms
        self.validity_check_fn = validity_check_fn
        self.verbose = verbose

        self.ecm_parser_simplifier = autorec_parser
        self.full_simplify_fn = getattr(autorec_parser, "full_simplify", None)
        self.full_simplify_import_error = None

        if self.full_simplify_fn is None:
            try:
                from .ecm_simplification_functions.parser import full_simplify

                self.full_simplify_fn = full_simplify
            except Exception as exc:
                self.full_simplify_import_error = exc

        self.random_ecm_param_names = ae.parser.get_parameter_labels(self.random_ecm_circuit)
        self.random_ecm_fn = ae.utils.generate_circuit_fn(self.random_ecm_circuit)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_target(value: str) -> Tuple[str, int]:
        """Parse one ``ECM=COUNT`` command-line target."""
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

    @staticmethod
    def build_argument_parser() -> argparse.ArgumentParser:
        """Build the command-line parser for data generation."""
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
            "--target-num",
            type=int,
            default=150,
            help="Number of samples required for every ECM discovered in the first batch.",
        )
        parser.add_argument("--max-batches", type=int, default=100)
        parser.add_argument("--seed-start", type=int, default=2026)
        parser.add_argument("--random-candidates", type=int, default=5000)
        parser.add_argument("--selected-curves", type=int, default=150)
        return parser

    def run_from_args(self, args: argparse.Namespace) -> int:
        """Execute the standalone generation workflow from parsed CLI args."""
        output_dir = args.output_dir.expanduser().resolve()

        self.random_ecm_circuit = args.source_ecm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_ecm_param_names = ae.parser.get_parameter_labels(self.random_ecm_circuit)
        self.random_ecm_fn = ae.utils.generate_circuit_fn(self.random_ecm_circuit)

        self.generate_data(
            target_num=args.target_num,
            min_batches=1,
            max_batches=args.max_batches,
            seed_start=args.seed_start,
            n_random_candidates=args.random_candidates,
            max_selected_curves=args.selected_curves,
            export=False,
            export_dataprep=True,
        )
        return 0

    def main(self, argv: Optional[Sequence[str]] = None) -> int:
        """Run the CLI entry point for balanced sample generation."""
        args = self.build_argument_parser().parse_args(argv)
        return self.run_from_args(args)

    # ------------------------------------------------------------------
    # General utilities
    # ------------------------------------------------------------------
    @staticmethod
    def source_ecm_filename_slug(circuit: str) -> str:
        """Create a deterministic filesystem-safe slug for a source ECM.

        Parameters
        ----------
        circuit : str
            Circuit string used to build a readable slug and hash suffix.

        Returns
        -------
        str
            Sanitized slug containing a short SHA1 digest.
        """
        source = str(circuit).strip()
        readable = re.sub(r"\s+", "", source)
        readable = re.sub(r"[^A-Za-z0-9._,\[\]-]+", "_", readable).strip("._-")
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        return f"{readable}_{digest}" if readable else f"unknown_ecm_{digest}"

    @staticmethod
    def split_top_level(text: str, sep: str = ",") -> List[str]:
        """Split a circuit string at a separator outside nested brackets.

        Parameters
        ----------
        text : str
            Circuit fragment to split.
        sep : str, default=","
            Separator to split on when bracket depth is zero.

        Returns
        -------
        list of str
            Top-level fragments preserving nested bracket content.
        """
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
        """Return the canonical sort group for a parallel-branch item.

        Parameters
        ----------
        item : str
            Branch item from a parallel block.

        Returns
        -------
        int
            Sort group, with CPE elements before resistors and other elements.
        """
        first = item.strip()[:1]
        if first == "P":
            return 0
        if first == "R":
            return 1
        return 2

    def _log(self, message: str) -> None:
        """Print a progress message when verbose logging is enabled.

        Parameters
        ----------
        message : str
            Message to print.
        """
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
        """Sample random source-circuit parameter dictionaries.

        Parameters
        ----------
        n_candidates : int, optional
            Number of random samples. Defaults to ``self.n_random_candidates``.
        seed : int, optional
            Random seed. Defaults to ``self.random_seed``.
        param_names : sequence of str, optional
            Parameter labels to use when converting sampled arrays to dicts.
        param_bounds : dict, optional
            Sampling bounds keyed by component type.

        Returns
        -------
        list of dict
            Parameter dictionaries keyed by AutoEIS parameter label.
        """
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

        params_list = [
            {name: float(value) for name, value in zip(param_names, row)} for row in sampled
        ]

        # Force generated source-circuit R1 to fixed value
        for params in params_list:
            if "R1" in params:
                params["R1"] = self.r1_value

        return params_list

    def params_to_array(
        self,
        params: Dict[str, float],
        param_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Convert a parameter dictionary to an array ordered by labels.

        Parameters
        ----------
        params : dict
            Parameter dictionary keyed by circuit parameter label.
        param_names : sequence of str, optional
            Desired output order. Defaults to source-circuit labels.

        Returns
        -------
        numpy.ndarray
            Parameter values in ``param_names`` order.
        """
        param_names = self.random_ecm_param_names if param_names is None else list(param_names)
        return np.array([params[name] for name in param_names], dtype=float)

    @staticmethod
    def normalize_curve(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize impedance data into unit-range Nyquist coordinates.

        Parameters
        ----------
        Z : numpy.ndarray
            Complex impedance values.

        Returns
        -------
        tuple of numpy.ndarray
            Normalized real part and normalized ``-imaginary`` part.
        """
        re_part = np.asarray(Z).real
        minus_im = -np.asarray(Z).imag

        re_rng = re_part.max() - re_part.min()
        im_rng = minus_im.max() - minus_im.min()

        re_norm = (re_part - re_part.min()) / re_rng if re_rng > 0 else np.zeros_like(re_part)
        minus_im_norm = (
            (minus_im - minus_im.min()) / im_rng if im_rng > 0 else np.zeros_like(minus_im)
        )

        return re_norm, minus_im_norm

    def circuit_params_to_array(self, circuit: str, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters to the order required by a specific circuit.

        Parameters
        ----------
        circuit : str
            Circuit whose AutoEIS parameter-label order should be used.
        params : dict
            Parameter values keyed by label.

        Returns
        -------
        numpy.ndarray
            Parameter values ordered according to ``circuit``.

        Raises
        ------
        TypeError
            If ``params`` is not a dictionary.
        KeyError
            If parameters required by ``circuit`` are missing.
        """
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
        """Simulate impedance for an arbitrary circuit and parameter set.

        Parameters
        ----------
        circuit : str
            AutoEIS circuit string to simulate.
        params : dict
            Circuit parameters keyed by AutoEIS parameter label.
        frequencies : numpy.ndarray, optional
            Frequencies in Hz. Defaults to ``self.random_ecm_freq``.

        Returns
        -------
        numpy.ndarray
            Complex impedance values.
        """
        frequencies = (
            self.random_ecm_freq
            if frequencies is None
            else np.asarray(frequencies, dtype=float)
        )
        circuit_fn = ae.utils.generate_circuit_fn(circuit)
        return circuit_fn(frequencies, self.circuit_params_to_array(circuit, params))

    def simulate_relabel_impedance(
        self, row: pd.Series, frequencies: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate impedance using the final relabelled ECM in a result row.

        Parameters
        ----------
        row : pandas.Series
            Result row containing ``relabel_ecm`` and ``relabel_params``.
        frequencies : numpy.ndarray, optional
            Frequencies in Hz. Defaults to ``self.random_ecm_freq``.

        Returns
        -------
        numpy.ndarray
            Complex impedance values for the relabelled ECM.
        """
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
        """Simulate and normalize a relabelled result row.

        Parameters
        ----------
        row : pandas.Series
            Result row containing final relabelled circuit information.
        frequencies : numpy.ndarray, optional
            Frequencies in Hz. Defaults to ``self.random_ecm_freq``.

        Returns
        -------
        tuple of numpy.ndarray
            Normalized real and ``-imaginary`` Nyquist coordinates.
        """
        return self.normalize_curve(
            self.simulate_relabel_impedance(row, frequencies=frequencies)
        )

    def simulate_candidates(
        self,
        params_list: Sequence[Dict[str, float]],
        circuit_fn: Optional[Callable] = None,
        frequencies: Optional[np.ndarray] = None,
        param_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, float]], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """Simulate random candidate parameters and keep finite curves.

        Parameters
        ----------
        params_list : sequence of dict
            Candidate parameter dictionaries.
        circuit_fn : callable, optional
            Precompiled AutoEIS circuit function. Defaults to source circuit.
        frequencies : numpy.ndarray, optional
            Frequencies in Hz. Defaults to ``self.random_ecm_freq``.
        param_names : sequence of str, optional
            Parameter-label order for conversion to arrays.

        Returns
        -------
        tuple
            ``(valid_params, valid_Z, valid_curves)`` where curves are
            normalized Nyquist coordinates.
        """
        circuit_fn = self.random_ecm_fn if circuit_fn is None else circuit_fn
        frequencies = (
            self.random_ecm_freq
            if frequencies is None
            else np.asarray(frequencies, dtype=float)
        )
        param_names = self.random_ecm_param_names if param_names is None else list(param_names)

        valid_params, valid_Z, valid_curves = [], [], []

        for params in tqdm(
            params_list,
            desc="Simulating random candidates",
            disable=True,
        ):
            try:
                Z = circuit_fn(
                    frequencies, self.params_to_array(params, param_names=param_names)
                )
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
    ) -> Tuple[
        List[Dict[str, float]],
        List[np.ndarray],
        List[Tuple[np.ndarray, np.ndarray]],
        Dict[str, Any],
    ]:
        """Filter candidates with high-frequency ``-Im(Z)`` above a threshold.

        Parameters
        ----------
        params_list : sequence of dict
            Candidate parameter dictionaries.
        Z_list : sequence of numpy.ndarray
            Complex impedance curves aligned with ``params_list``.
        curves : sequence of tuple
            Normalized Nyquist coordinates aligned with ``params_list``.
        high_freq_index : int, optional
            Frequency index used for filtering.
        max_minus_im_norm : float, optional
            Maximum allowed normalized ``-Im(Z)`` value.

        Returns
        -------
        tuple
            Filtered parameters, impedance curves, normalized curves, and a
            dictionary describing the filter operation.

        Raises
        ------
        ValueError
            If every candidate is removed.
        """
        high_freq_index = (
            self.high_frequency_index if high_freq_index is None else high_freq_index
        )
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
        """Stack normalized curves into a 3D point array.

        Parameters
        ----------
        curves : sequence of tuple
            Normalized real and ``-imaginary`` curve arrays.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_curves, n_frequencies, 2)``.
        """
        return np.stack([np.column_stack(curve) for curve in curves]).astype(np.float32)

    @staticmethod
    def mean_curve_distance(
        curve_points: np.ndarray, reference_curve: np.ndarray
    ) -> np.ndarray:
        """Compute mean pointwise distance from each curve to a reference.

        Parameters
        ----------
        curve_points : numpy.ndarray
            Stacked curves with shape ``(n_curves, n_frequencies, 2)``.
        reference_curve : numpy.ndarray
            Reference curve with shape ``(n_frequencies, 2)``.

        Returns
        -------
        numpy.ndarray
            Mean Euclidean distance for each candidate curve.
        """
        return np.linalg.norm(curve_points - reference_curve[None, :, :], axis=2).mean(axis=1)

    def greedy_select(
        self,
        params_list: Sequence[Dict[str, float]],
        Z_list: Sequence[np.ndarray],
        curves: Sequence[Tuple[np.ndarray, np.ndarray]],
        k: Optional[int] = None,
        min_distance: Optional[float] = None,
        reference_curves: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[
        List[Dict[str, float]],
        List[np.ndarray],
        List[Tuple[np.ndarray, np.ndarray]],
        List[int],
        List[float],
    ]:
        """Select a diverse subset of normalized curves greedily.

        Parameters
        ----------
        params_list : sequence of dict
            Candidate parameter dictionaries.
        Z_list : sequence of numpy.ndarray
            Complex impedance curves aligned with ``params_list``.
        curves : sequence of tuple
            Normalized Nyquist curves aligned with ``params_list``.
        k : int, optional
            Maximum number of curves to select.
        min_distance : float, optional
            Stop selection when the next best curve is closer than this value.
        reference_curves : sequence of tuple, optional
            Previously kept normalized Nyquist curves. When provided, candidates
            are selected by maximizing their minimum distance to both these
            references and the curves already selected in the current batch.

        Returns
        -------
        tuple
            Selected parameters, impedance curves, normalized curves, original
            filtered indices, and greedy distance scores.
        """
        k = self.max_selected_curves if k is None else k
        min_distance = (
            self.selection_distance_threshold if min_distance is None else min_distance
        )

        if not curves or k <= 0:
            return [], [], [], [], []

        curve_points = self.stack_curves(curves)
        reference_curves = [] if reference_curves is None else list(reference_curves)

        if reference_curves:
            reference_points = self.stack_curves(reference_curves)
            reference_distances = [
                self.mean_curve_distance(curve_points, reference_curve)
                for reference_curve in reference_points
            ]
            min_dist = np.min(np.stack(reference_distances), axis=0)
            first, first_score = int(np.argmax(min_dist)), float(np.max(min_dist))

            if min_distance is not None and first_score < min_distance:
                return [], [], [], [], []
        else:
            first = int(
                np.argmax(self.mean_curve_distance(curve_points, curve_points.mean(axis=0)))
            )
            first_score = np.inf
            min_dist = self.mean_curve_distance(curve_points, curve_points[first])

        selected, scores = [first], [first_score]
        min_dist[first] = -np.inf

        while len(selected) < min(k, len(curves)):
            nxt, score = int(np.argmax(min_dist)), float(np.max(min_dist))

            if min_distance is not None and score < min_distance:
                break

            selected.append(nxt)
            scores.append(score)

            min_dist = np.minimum(
                min_dist, self.mean_curve_distance(curve_points, curve_points[nxt])
            )
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
        reference_curves: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """Run sampling, simulation, high-frequency filtering, and selection.

        Parameters
        ----------
        seed : int, optional
            Random seed for parameter sampling.
        n_random_candidates : int, optional
            Number of random samples to generate.
        max_selected_curves : int, optional
            Maximum number of curves selected after filtering.
        reference_curves : sequence of tuple, optional
            Previously kept normalized Nyquist curves used as diversity
            references during greedy selection.

        Returns
        -------
        dict
            Intermediate arrays and metadata for the generate/filter/select
            portion of the workflow.
        """
        candidates = self.sample_params(n_candidates=n_random_candidates, seed=seed)
        params, Z, curves = self.simulate_candidates(candidates)

        filtered_params, filtered_Z, filtered_curves, filter_info = self.filter_high_frequency(
            params,
            Z,
            curves,
        )

        selected_params, selected_Z, selected_curves, selected_idx, selected_scores = (
            self.greedy_select(
                filtered_params,
                filtered_Z,
                filtered_curves,
                k=max_selected_curves,
                reference_curves=reference_curves,
            )
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
    def validate_and_order_circuit_params(
        circuit: str,
        params: Dict[str, float],
    ) -> Dict[str, float]:
        """Validate parameter labels and order them to match the circuit."""
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary keyed by parameter label")

        expected_labels = ae.parser.get_parameter_labels(str(circuit))
        expected_set = set(expected_labels)
        actual_set = set(params)

        if actual_set != expected_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            raise ValueError(
                "Circuit and parameter labels do not match. "
                f"Missing labels: {missing}; extra labels: {extra}."
            )

        return {label: params[label] for label in expected_labels}

    def run_parser_full_simplify(
        self,
        circuit: str,
        params: Dict[str, float],
        Z: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[str, Dict[str, float], Optional[Dict[str, Any]]]:
        """Run the parser-provided full simplification for one EIS curve.

        Parameters
        ----------
        circuit : str
            Source circuit to simplify.
        params : dict
            Source-circuit parameters.
        Z : numpy.ndarray
            Complex impedance curve used by parser-level FIM analysis.

        Returns
        -------
        tuple
            Fully simplified circuit, parameter dictionary, and optional FIM
            simplification information. The information value is ``None`` for
            older simplifier implementations that return only two values.
        """
        if self.full_simplify_fn is None:
            raise RuntimeError(
                "No parser full_simplify function is available. "
                f"Fallback import error: {self.full_simplify_import_error!r}"
            )

        params = self.validate_and_order_circuit_params(circuit, params)

        result = self.full_simplify_fn(
            circuit,
            self.random_ecm_freq,
            Z,
            params,
            identifiability_thresh=self.fim_identifiability_thresh,
            refit_ecm=self.fim_fit_ecm,
            fit_kwargs={
                "max_iters": self.fim_refit_max_iters,
                "min_iters": self.fim_refit_min_iters,
                "max_nfev": self.fim_refit_max_nfev,
            },
            verbose=verbose,
        )

        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise TypeError(
                "full_simplify must return (circuit, params) or "
                "(circuit, params, info)"
            )

        simplified_circuit, simplified_params = result[:2]
        info = result[2] if len(result) == 3 else None
        return simplified_circuit, simplified_params, info

    def canonicalize_relabel_result(
        self,
        circuit: str,
        params: Dict[str, float],
    ) -> Tuple[str, Dict[str, float]]:
        """Convert a simplification result into the exported ECM convention."""
        circuit, params = self.ensure_series_r1(circuit, params)
        circuit, params = self.convert_capacitors_to_cpes(circuit, params)
        circuit = self.reorder_parallel_blocks_and_series_p(circuit)
        circuit, mapping = self.reindex_components(circuit)
        params = self.reindex_parameter_dict(params, mapping)
        params = self.validate_and_order_circuit_params(circuit, params)
        return circuit, params

    def enforce_final_relabel_consistency(
        self,
        circuit: str,
        params: Dict[str, float],
    ) -> Tuple[str, Dict[str, float], str, Dict[str, float], List[Dict[str, Any]]]:
        """Simplify the final simulated ECM repeatedly until its topology is stable."""
        current_circuit, current_params = self.canonicalize_relabel_result(
            circuit,
            params,
        )
        seen_circuits = {current_circuit}
        trace = []

        for iteration in range(self.final_consistency_max_iters):
            final_Z = self.simulate_circuit_with_params(
                current_circuit,
                current_params,
                self.random_ecm_freq,
            )
            checked_circuit, checked_params, checked_info = (
                self.run_parser_full_simplify(
                    current_circuit,
                    current_params,
                    final_Z,
                )
            )
            next_circuit, next_params = self.canonicalize_relabel_result(
                checked_circuit,
                checked_params,
            )
            trace.append(
                {
                    "iteration": iteration,
                    "input_circuit": current_circuit,
                    "simplified_circuit": checked_circuit,
                    "canonical_circuit": next_circuit,
                    "fim_identifiability_info": checked_info,
                }
            )

            if next_circuit == current_circuit:
                return (
                    next_circuit,
                    next_params,
                    checked_circuit,
                    checked_params,
                    trace,
                )

            if next_circuit in seen_circuits:
                raise RuntimeError(
                    "Final relabel consistency check entered a circuit cycle: "
                    f"{next_circuit}"
                )

            seen_circuits.add(next_circuit)
            current_circuit, current_params = next_circuit, next_params

        raise RuntimeError(
            "Final relabel consistency check did not stabilize within "
            f"{self.final_consistency_max_iters} iterations."
        )

    def run_relabel(
        self,
        params_list: Sequence[Dict[str, float]],
        Z_list: Sequence[np.ndarray],
        curves: Sequence[Tuple[np.ndarray, np.ndarray]],
        batch_seed: Optional[int] = None,
        batch_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Relabel selected curves with parser full simplification metadata.

        Parameters
        ----------
        params_list : sequence of dict
            Selected source-circuit parameter dictionaries.
        Z_list : sequence of numpy.ndarray
            Selected complex impedance curves.
        curves : sequence of tuple
            Selected normalized Nyquist curves.
        batch_seed : int, optional
            Seed used to generate the batch.
        batch_id : int, optional
            Batch identifier.

        Returns
        -------
        pandas.DataFrame
            One row per selected curve with original data, relabelled ECMs,
            relabelled parameters, parser simplification output, and failure
            metadata.
        """
        rows = []

        for selected_position, (params, Z, curve) in enumerate(
            tqdm(
                zip(params_list, Z_list, curves),
                total=len(params_list),
                desc="Parser full simplify selected curves",
                disable=True,
            )
        ):
            record = {
                "batch_id": batch_id,
                "batch_seed": batch_seed,
                "selected_position": selected_position,
                "original_ecm": self.random_ecm_circuit,
                "simplified_ecm": None,
                "fim_relabel_ecm": None,
                "post_fim_simplified_ecm": None,
                "fim_identifiability_info": None,
                "relabel_ecm": None,
                "params": params,
                "simplified_params": None,
                "fim_relabel_params": None,
                "post_fim_simplified_params": None,
                "relabel_params": None,
                "fim_candidates": None,
                "Z": Z,
                "curve": curve,
                "final_Z": None,
                "final_curve": None,
                "relabel_failed": False,
                "failure_reason": None,
            }

            try:
                simplified_ecm, simplified_params, info = self.run_parser_full_simplify(
                    self.random_ecm_circuit, params, Z
                )
                (
                    relabel_ecm,
                    relabel_params,
                    post_fim_simplified_ecm,
                    post_fim_simplified_params,
                    consistency_trace,
                ) = self.enforce_final_relabel_consistency(
                    simplified_ecm,
                    simplified_params,
                )
                final_Z = self.simulate_circuit_with_params(
                    relabel_ecm,
                    relabel_params,
                    self.random_ecm_freq,
                )
                final_curve = self.normalize_curve(final_Z)

                record.update(
                    {
                        "simplified_ecm": simplified_ecm,
                        "fim_relabel_ecm": simplified_ecm,
                        "post_fim_simplified_ecm": post_fim_simplified_ecm,
                        "relabel_ecm": relabel_ecm,
                        "simplified_params": simplified_params,
                        "fim_relabel_params": simplified_params,
                        "post_fim_simplified_params": post_fim_simplified_params,
                        "fim_identifiability_info": info,
                        "relabel_params": relabel_params,
                        "fim_candidates": consistency_trace,
                        "final_Z": final_Z,
                        "final_curve": final_curve,
                    }
                )

            except Exception as exc:
                record.update(
                    {
                        "simplified_ecm": self.random_ecm_circuit,
                        "fim_relabel_ecm": self.random_ecm_circuit,
                        "post_fim_simplified_ecm": self.random_ecm_circuit,
                        "relabel_ecm": self.random_ecm_circuit,
                        "relabel_failed": True,
                        "failure_reason": repr(exc),
                    }
                )

            rows.append(record)

        return pd.DataFrame(rows)

    def has_series_p_chain(self, circuit: Optional[str]) -> bool:
        """Check whether a circuit contains adjacent series CPE elements.

        Parameters
        ----------
        circuit : str or None
            Circuit string to inspect.

        Returns
        -------
        bool
            ``True`` if a top-level series chain contains adjacent ``P``
            elements.
        """
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
        """Remove rows whose final ECM contains a series P-P chain.

        Parameters
        ----------
        df : pandas.DataFrame
            Relabelled result table.
        circuit_col : str, default="relabel_ecm"
            Column containing circuit strings to inspect.

        Returns
        -------
        pandas.DataFrame
            Filtered result table with reset index.
        """
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
        """Remove rows whose final ECM fails the configured validity check.

        Parameters
        ----------
        df : pandas.DataFrame
            Relabelled result table.
        circuit_col : str, default="relabel_ecm"
            Column containing circuit strings to validate.

        Returns
        -------
        pandas.DataFrame
            Valid relabelled rows with reset index.
        """
        if self.validity_check_fn is None:
            self._log("Skipping validity_check because no validity_check_fn is available.")
            return df.reset_index(drop=True)

        df = df.copy()

        def safe_validity_check(circuit: Any) -> bool:
            """Return validity-check output while treating exceptions as invalid."""
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

    @staticmethod
    def circuit_key(circuit: Any) -> Optional[str]:
        """Build a whitespace-insensitive circuit key.

        Parameters
        ----------
        circuit : object
            Circuit-like value to normalize.

        Returns
        -------
        str or None
            Circuit string with whitespace removed, or ``None`` for null
            values.
        """
        if circuit is None:
            return None

        try:
            if pd.isna(circuit):
                return None
        except (TypeError, ValueError):
            pass

        return re.sub(r"\s+", "", str(circuit))

    def excluded_simplified_ecm_keys(
        self,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
    ) -> set[str]:
        """Build comparable keys for excluded simplified ECM labels.

        Parameters
        ----------
        excluded_simplified_ecms : sequence of str or None, optional
            Exclusion labels. ``None`` uses ``self.excluded_simplified_ecms``.

        Returns
        -------
        set of str
            Whitespace-insensitive original and canonicalized exclusion keys.
        """
        excluded_simplified_ecms = (
            self.excluded_simplified_ecms
            if excluded_simplified_ecms is None
            else excluded_simplified_ecms
        )

        if excluded_simplified_ecms is None:
            return set()

        if isinstance(excluded_simplified_ecms, str):
            excluded_simplified_ecms = [excluded_simplified_ecms]

        excluded_keys = set()

        for circuit in excluded_simplified_ecms:
            key = self.circuit_key(circuit)
            if key is not None:
                excluded_keys.add(key)

                cpe_key = self.circuit_key(re.sub(r"C(\d+)", r"P\1", str(circuit)))
                if cpe_key is not None:
                    excluded_keys.add(cpe_key)

            try:
                normalized_circuit, _ = self.normalize_circuit_and_params(circuit)
                normalized_key = self.circuit_key(normalized_circuit)
            except Exception:
                normalized_key = None

            if normalized_key is not None:
                excluded_keys.add(normalized_key)

        return excluded_keys

    def drop_excluded_simplified_ecms(
        self,
        df: pd.DataFrame,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
        circuit_col: str = "relabel_ecm",
    ) -> pd.DataFrame:
        """Remove rows whose final simplified ECM is in the exclusion list.

        Parameters
        ----------
        df : pandas.DataFrame
            Relabelled result table.
        excluded_simplified_ecms : sequence of str or None, optional
            Exclusion labels. ``None`` uses ``self.excluded_simplified_ecms``.
        circuit_col : str, default="relabel_ecm"
            Column containing final simplified/relabelled ECM labels.

        Returns
        -------
        pandas.DataFrame
            Filtered result table with reset index.
        """
        excluded_keys = self.excluded_simplified_ecm_keys(excluded_simplified_ecms)

        if not excluded_keys:
            return df.reset_index(drop=True)

        df = df.copy()
        excluded_mask = df[circuit_col].apply(
            lambda circuit: self.circuit_key(circuit) in excluded_keys
        )

        self._log(f"Dropping {int(excluded_mask.sum())} excluded simplified ECMs")
        self._log(f"Keeping {int((~excluded_mask).sum())} rows after simplified ECM exclusion")

        return df.loc[~excluded_mask].reset_index(drop=True)

    def reorder_parallel_blocks_and_series_p(self, circuit: Optional[str]) -> Optional[str]:
        """Canonicalize branch order in parallel blocks and series CPE position.

        Parameters
        ----------
        circuit : str or None
            Circuit string to reorder.

        Returns
        -------
        str or None
            Circuit with parallel block items sorted and top-level series CPE
            elements moved after non-CPE series elements.
        """
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

            items = self.split_top_level(circuit[start + 1 : i - 1])
            items = [self.reorder_parallel_blocks_and_series_p(item) for item in items]
            items = sorted(items, key=self._parallel_sort_key)
            out.append("[" + ",".join(items) + "]")

        circuit = "".join(out)
        series_parts = self.split_top_level(circuit, sep="-")

        p_parts = [part for part in series_parts if re.fullmatch(r"\s*P\d+\s*", part)]
        non_p_parts = [part for part in series_parts if not re.fullmatch(r"\s*P\d+\s*", part)]

        return "-".join(non_p_parts + p_parts)

    @staticmethod
    def convert_capacitors_to_cpes(
        circuit: Optional[str],
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Convert capacitor components to equivalent CPE components.

        Parameters
        ----------
        circuit : str or None
            Circuit string that may contain ``C`` components.
        params : dict, optional
            Parameters for ``circuit``. Capacitor values are converted to
            ``Pw`` values with ``Pn`` set to ``1.0``.

        Returns
        -------
        tuple
            Converted circuit and converted parameter dictionary.
        """
        if circuit is None:
            return circuit, params

        circuit = re.sub(r"C(\d+)", r"P\1", str(circuit))

        if not isinstance(params, dict):
            return circuit, params

        converted_params = {}

        for key, value in params.items():
            key = str(key)
            match = re.fullmatch(r"C(\d+)", key)

            if match:
                cpe_base = f"P{match.group(1)}"
                converted_params[f"{cpe_base}w"] = value
                converted_params[f"{cpe_base}n"] = 1.0
            else:
                converted_params[key] = value

        return circuit, converted_params

    def ensure_series_r1(
        self,
        circuit: Optional[str],
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Ensure the canonical circuit starts with a series ``R1``.

        Parameters
        ----------
        circuit : str or None
            Circuit string to normalize.
        params : dict, optional
            Parameters aligned with ``circuit``.

        Returns
        -------
        tuple
            Circuit with front series resistor renamed or inserted as ``R1``
            and parameters updated. Existing fitted resistor values are
            preserved; ``self.r1_value`` is used only for a newly inserted R1.
        """
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
                if old_front in params:
                    fitted_value = params.pop(old_front)
                    params["R1"] = fitted_value

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
        """Rename parameter dictionary keys using a component mapping.

        Parameters
        ----------
        params : dict or None
            Parameter dictionary to rename.
        mapping : dict
            Component-name mapping, for example ``{"R5": "R2"}``.

        Returns
        -------
        dict or None
            Parameter dictionary with renamed component prefixes.
        """
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
        """Renumber all circuit components in one canonical global sequence.

        Component types share a single left-to-right counter. For example,
        ``R1-[P3,R5]`` becomes ``R1-[P2,R3]`` rather than using separate
        counters for ``R`` and ``P`` components.

        Parameters
        ----------
        circuit : str or None
            Circuit string to reindex.

        Returns
        -------
        tuple
            Reindexed circuit and mapping from old component names to new
            component names.
        """
        if circuit is None:
            return circuit, {}

        circuit = str(circuit)

        component_labels = list(dict.fromkeys(ae.parser.get_component_labels(circuit)))
        mapping = {
            old_name: f"{old_name[0]}{new_index}"
            for new_index, old_name in enumerate(component_labels, start=1)
        }

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
        """Canonicalize a circuit string and its parameter dictionary.

        Parameters
        ----------
        circuit : str or None
            Circuit string to canonicalize.
        params : dict, optional
            Parameter dictionary aligned with ``circuit``.

        Returns
        -------
        tuple
            Canonical circuit and parameter dictionary after ``R1``
            normalization, branch reordering, and reindexing.
        """
        if circuit is None or pd.isna(circuit):
            return circuit, params

        circuit, params = self.ensure_series_r1(circuit, params)
        circuit = self.reorder_parallel_blocks_and_series_p(circuit)

        circuit, mapping = self.reindex_components(circuit)
        params = self.reindex_parameter_dict(params, mapping)

        if isinstance(params, dict):
            params = self.validate_and_order_circuit_params(circuit, params)

        return circuit, params

    def postprocess_relabels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Canonicalize final relabel circuit and parameter columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Relabelled result table.

        Returns
        -------
        pandas.DataFrame
            Copy of ``df`` with canonicalized final relabel columns where
            present. Parser simplification columns are left unchanged for
            debugging.
        """
        df = df.copy()

        for circuit_col, params_col in (("relabel_ecm", "relabel_params"),):
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
    def relabel_group_counts(
        df: pd.DataFrame, target_labels: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """Count valid rows per final relabelled ECM.

        Parameters
        ----------
        df : pandas.DataFrame
            Relabelled result table.
        target_labels : sequence of str, optional
            Labels to include in the output, with missing labels filled as
            zero.

        Returns
        -------
        pandas.Series
            Counts indexed by final ``relabel_ecm`` label.
        """
        source = df.dropna(subset=["relabel_ecm"]).copy()

        if "relabel_failed" in source.columns:
            source = source.loc[~source["relabel_failed"]]

        counts = source["relabel_ecm"].value_counts().sort_index()

        if target_labels is not None:
            counts = counts.reindex(target_labels, fill_value=0)

        return counts

    @staticmethod
    def target_count_series(
        target_per_relabel: int | Mapping[str, int],
        target_labels: Sequence[str],
    ) -> pd.Series:
        """Build per-label target counts from a scalar or explicit mapping.

        Parameters
        ----------
        target_per_relabel : int or mapping
            Uniform count for every label, or explicit count by relabelled ECM.
        target_labels : sequence of str
            Labels that must have target counts.

        Returns
        -------
        pandas.Series
            Target counts indexed by relabelled ECM label.
        """
        labels = list(target_labels)

        if isinstance(target_per_relabel, Mapping):
            missing = [label for label in labels if label not in target_per_relabel]
            if missing:
                raise ValueError(
                    f"target_per_relabel is missing target count(s) for label(s): {missing}"
                )
            return pd.Series(
                {label: int(target_per_relabel[label]) for label in labels},
                dtype=int,
            )

        return pd.Series(int(target_per_relabel), index=labels, dtype=int)

    def run_selected_relabel_batch(
        self,
        seed: int,
        batch_id: int,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
        reference_curves: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate, select, relabel, and filter one random batch.

        Parameters
        ----------
        seed : int
            Random seed for this batch.
        batch_id : int
            Batch identifier recorded in output rows.
        n_random_candidates : int, optional
            Number of random source-parameter samples.
        max_selected_curves : int, optional
            Maximum number of selected curves passed to relabelling.
        excluded_simplified_ecms : sequence of str or None, optional
            Final simplified/relabelled ECM labels to exclude. ``None`` uses
            ``self.excluded_simplified_ecms``.
        reference_curves : sequence of tuple, optional
            Previously kept normalized Nyquist curves used as diversity
            references during greedy selection.

        Returns
        -------
        tuple
            Batch result DataFrame and batch metadata dictionary.
        """
        n_random_candidates = (
            self.n_random_candidates if n_random_candidates is None else n_random_candidates
        )
        max_selected_curves = (
            self.max_selected_curves if max_selected_curves is None else max_selected_curves
        )

        candidates = self.sample_params(n_candidates=n_random_candidates, seed=seed)

        params, Z, curves = self.simulate_candidates(candidates)

        filtered_params, filtered_Z, filtered_curves, filter_info = self.filter_high_frequency(
            params, Z, curves
        )

        selected_params, selected_Z, selected_curves, selected_idx, selected_scores = (
            self.greedy_select(
                filtered_params,
                filtered_Z,
                filtered_curves,
                k=max_selected_curves,
                reference_curves=reference_curves,
            )
        )

        batch_df = self.run_relabel(
            selected_params, selected_Z, selected_curves, batch_seed=seed, batch_id=batch_id
        )

        batch_df = self.postprocess_relabels(batch_df)

        before_excluded_filter_count = len(batch_df)
        batch_df = self.drop_excluded_simplified_ecms(
            batch_df,
            excluded_simplified_ecms=excluded_simplified_ecms,
            circuit_col="relabel_ecm",
        )
        removed_by_excluded_simplified_ecm_filter = before_excluded_filter_count - len(
            batch_df
        )

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
            "reference_curve_count": 0 if reference_curves is None else len(reference_curves),
            "after_final_filters_count": len(batch_df),
            "removed_by_high_frequency_filter": filter_info["removed_candidate_count"],
            "removed_by_excluded_simplified_ecm_filter": removed_by_excluded_simplified_ecm_filter,
        }

        return batch_df, batch_info

    def take_needed_target_rows(
        self,
        batch_df: pd.DataFrame,
        counts: pd.Series,
        target_labels: Sequence[str],
        target_per_relabel: int | Mapping[str, int],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Keep only rows still needed to meet target counts.

        Parameters
        ----------
        batch_df : pandas.DataFrame
            Candidate relabelled rows from one batch.
        counts : pandas.Series
            Current accumulated count per target label. Updated in place.
        target_labels : sequence of str
            Final ECM labels being balanced.
        target_per_relabel : int or mapping
            Desired number of rows per target label.

        Returns
        -------
        tuple
            Kept rows from ``batch_df`` and updated ``counts``.
        """
        kept = []

        usable = batch_df.dropna(subset=["relabel_ecm"]).copy()

        if "relabel_failed" in usable.columns:
            usable = usable.loc[~usable["relabel_failed"]]

        target_counts = self.target_count_series(target_per_relabel, target_labels)

        for label in target_labels:
            needed = int(target_counts[label]) - int(counts.get(label, 0))

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

    def set_source_ecm(self, circuit: str) -> None:
        """Switch the source ECM and rebuild cached AutoEIS helpers."""
        self.random_ecm_circuit = str(circuit)
        self.random_ecm_param_names = ae.parser.get_parameter_labels(self.random_ecm_circuit)
        self.random_ecm_fn = ae.utils.generate_circuit_fn(self.random_ecm_circuit)

    def relabel_complexity(self, circuit: str) -> int:
        """Count fitted parameters in a relabelled ECM.

        AutoEIS expands each CPE into its ``Pw`` and ``Pn`` parameters while
        resistors, capacitors, and inductors each contribute one parameter.
        """
        return ae.parser.count_parameters(str(circuit))

    def run_balanced_relabel_dataset(
        self,
        target_per_relabel: int = 150,
        target_relabel_ecms: Optional[Sequence[str]] = None,
        min_batches: int = 1,
        max_batches: int = 200,
        seed_start: int = 1000,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
        export_selected_samples: bool = False,
        export_output_dir: Optional[str | Path] = None,
        use_relabel_simulation: bool = True,
        live_plot: bool = True,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Discover groups and choose the next source from unfinished counts.

        After every batch, the collected sample counts are inspected. The next
        source ECM is selected directly from the unfinished relabelled ECM
        groups, choosing the most complex unfinished group. No preset source
        sequence is used. Accepted curves remain diversity references after a
        source-ECM switch so selection stays diverse across the full dataset.
        """
        if not isinstance(target_per_relabel, int) or target_per_relabel < 1:
            raise ValueError("target_per_relabel must be a positive integer.")

        original_source_ecm = self.random_ecm_circuit
        active_source_ecm = original_source_ecm

        kept_batches = []
        kept_reference_curves = []
        batch_infos = []
        counts = pd.Series(dtype=int)
        target_labels = (
            list(dict.fromkeys(target_relabel_ecms)) if target_relabel_ecms is not None else []
        )

        export_counts: Dict[str, int] = {}
        export_root = None

        if export_selected_samples:
            export_root = self.prepare_eis_data_folder(export_output_dir)

        live_plot_path = self.output_dir / "generation_analysis.png"
        stable_batches = 0
        discovery_patience = 3

        for batch_id in range(max_batches):
            seed = seed_start + batch_id

            batch_df, batch_info = self.run_selected_relabel_batch(
                seed=seed,
                batch_id=batch_id,
                n_random_candidates=n_random_candidates,
                max_selected_curves=max_selected_curves,
                excluded_simplified_ecms=excluded_simplified_ecms,
                reference_curves=kept_reference_curves,
            )

            batch_info["source_ecm"] = active_source_ecm
            batch_infos.append(batch_info)

            discovered_now = sorted(
                str(label) for label in self.relabel_group_counts(batch_df).index
            )
            new_labels = [label for label in discovered_now if label not in target_labels]

            if new_labels:
                target_labels.extend(new_labels)
                target_labels.sort(
                    key=lambda label: (
                        -self.relabel_complexity(label),
                        label,
                    )
                )
                stable_batches = 0
            else:
                stable_batches += 1

            counts = counts.reindex(target_labels, fill_value=0).astype(int)
            target_counts = pd.Series(
                target_per_relabel,
                index=target_labels,
                dtype=int,
            )
            before_counts = counts.copy()

            kept_df, counts = self.take_needed_target_rows(
                batch_df,
                counts,
                target_labels,
                target_per_relabel,
            )

            if not kept_df.empty:
                kept_batches.append(kept_df)
                reference_column = (
                    "final_curve" if "final_curve" in kept_df.columns else "curve"
                )
                kept_reference_curves.extend(kept_df[reference_column].tolist())

            exported_sample_count = 0

            if export_selected_samples and export_root is not None:
                for _, row in kept_df.iterrows():
                    label = self.ecm_output_folder_name(row["relabel_ecm"])
                    sample_index = export_counts.get(label, 0)

                    self.export_eis_sample(
                        row,
                        sample_index=sample_index,
                        output_dir=export_root,
                        use_relabel_simulation=use_relabel_simulation,
                    )

                    export_counts[label] = sample_index + 1
                    exported_sample_count += 1

            added_counts = counts.subtract(before_counts, fill_value=0).astype(int)

            batch_info["exported_sample_count"] = exported_sample_count
            batch_info["added_counts"] = {
                str(label): int(value) for label, value in added_counts.items()
            }
            batch_info["accumulated_counts"] = {
                str(label): int(value) for label, value in counts.items()
            }
            batch_info["discovered_group_count"] = len(target_labels)
            batch_info["new_groups"] = list(new_labels)

            print(f"\nBatch {batch_id + 1}/{max_batches}")
            print(f"Source ECM          : {active_source_ecm}")
            print(f"Accepted this batch : {len(kept_df)}")
            print("Current progress:")

            for label in target_labels:
                current = int(counts.get(label, 0))
                print(f"  {label:<45} {current:>4}/{target_per_relabel}")

            print("Remaining:")

            for label in target_labels:
                current = int(counts.get(label, 0))
                remaining = max(0, target_per_relabel - current)
                print(f"  {label:<45} {remaining:>4}")

            print("-" * 72)

            if live_plot and target_labels:
                import matplotlib

                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt

                batch_numbers = np.arange(1, len(batch_infos) + 1)
                fig, ax = plt.subplots(figsize=DEFAULT_GENERATION_FIGSIZE)

                for label in target_labels:
                    values = [
                        int(info.get("added_counts", {}).get(label, 0)) for info in batch_infos
                    ]
                    ax.plot(
                        batch_numbers,
                        values,
                        marker="o",
                        linewidth=1.8,
                        label=label,
                    )

                ax.set_title("Accepted samples generated per iteration")
                ax.set_xlabel("Generation iteration")
                ax.set_ylabel("New accepted samples")
                ax.set_xticks(batch_numbers)
                ax.grid(True, linestyle=":", alpha=0.35)
                ax.legend(
                    title="Relabelled ECM",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )

                fig.tight_layout()
                fig.savefig(
                    live_plot_path,
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

            unfinished_labels = [
                label
                for label in target_labels
                if int(counts.get(label, 0)) < target_per_relabel
            ]

            if unfinished_labels:
                next_source_ecm = max(
                    unfinished_labels,
                    key=lambda label: (
                        self.relabel_complexity(label),
                        label,
                    ),
                )

                if next_source_ecm != active_source_ecm:
                    previous_source_ecm = active_source_ecm
                    self.set_source_ecm(next_source_ecm)
                    active_source_ecm = next_source_ecm

                    # Retain curves accepted under earlier source ECMs so the
                    # next batch is selected against the full dataset history.

                    print(
                        "Generation target changed: "
                        f"{previous_source_ecm} -> {next_source_ecm}"
                    )

            all_targets_complete = bool(target_labels) and not unfinished_labels

            if (
                batch_id + 1 >= min_batches
                and all_targets_complete
                and stable_batches >= discovery_patience
            ):
                self.set_source_ecm(original_source_ecm)

                if not kept_batches:
                    return batch_df.iloc[0:0].copy(), batch_infos

                return pd.concat(
                    kept_batches,
                    ignore_index=True,
                ), batch_infos

        self.set_source_ecm(original_source_ecm)

        target_counts = pd.Series(
            target_per_relabel,
            index=target_labels,
            dtype=int,
        )

        raise RuntimeError(
            "Balanced relabel dataset did not reach all dynamically discovered "
            f"targets within {max_batches} batches.\n"
            f"Target counts:\n{target_counts.to_string()}\n"
            f"Current counts:\n{counts.to_string()}"
        )

    def plot_generation_analysis(
        self,
        batch_infos: Sequence[Mapping[str, Any]],
        target_labels: Optional[Sequence[str]] = None,
        output_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> Path:
        """Plot newly accepted samples per generation iteration as lines.

        Each relabelled ECM is shown as one line. The plot is saved and,
        by default, displayed as the function output.
        """
        import matplotlib.pyplot as plt

        if not batch_infos:
            raise ValueError("Cannot plot generation analysis without batch information.")

        if output_path is None:
            output_path = self.output_dir / "generation_analysis.png"
        else:
            output_path = Path(output_path)

        if target_labels is None:
            target_labels = []
            for info in batch_infos:
                for label in info.get("added_counts", {}):
                    if label not in target_labels:
                        target_labels.append(label)
        else:
            target_labels = list(target_labels)

        batch_numbers = np.arange(1, len(batch_infos) + 1)

        fig, ax = plt.subplots(figsize=DEFAULT_GENERATION_FIGSIZE)

        for label in target_labels:
            values = [int(info.get("added_counts", {}).get(label, 0)) for info in batch_infos]
            ax.plot(
                batch_numbers,
                values,
                marker="o",
                linewidth=1.8,
                label=label,
            )

        ax.set_title("Accepted samples generated per iteration")
        ax.set_xlabel("Generation iteration")
        ax.set_ylabel("New accepted samples")
        ax.set_xticks(batch_numbers)
        ax.grid(True, linestyle=":", alpha=0.35)

        if target_labels:
            ax.legend(
                title="Relabelled ECM",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def build_final_relabel_df(
        self,
        balanced_relabel_results_df: pd.DataFrame,
        target_per_relabel: int | Mapping[str, int],
        target_relabel_ecms: Optional[Sequence[str]] = None,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Trim balanced relabel rows to the final export set.

        Parameters
        ----------
        balanced_relabel_results_df : pandas.DataFrame
            Accumulated balanced relabel results.
        target_per_relabel : int or mapping
            Number of rows to keep per final relabelled ECM.
        target_relabel_ecms : sequence of str, optional
            Optional final label filter.
        excluded_simplified_ecms : sequence of str or None, optional
            Final simplified/relabelled ECM labels to exclude. ``None`` uses
            ``self.excluded_simplified_ecms``.

        Returns
        -------
        pandas.DataFrame
            Final rows sorted by label and annotated with ``global_position``.

        Raises
        ------
        RuntimeError
            If the final trimmed rows do not satisfy ``target_per_relabel``.
        """
        source_df = balanced_relabel_results_df.dropna(subset=["relabel_ecm"]).copy()
        source_df = source_df.loc[~source_df["relabel_failed"]]

        if target_relabel_ecms is not None:
            source_df = source_df.loc[source_df["relabel_ecm"].isin(target_relabel_ecms)]

        source_df = self.drop_excluded_simplified_ecms(
            source_df,
            excluded_simplified_ecms=excluded_simplified_ecms,
            circuit_col="relabel_ecm",
        )

        if target_relabel_ecms is not None:
            target_labels = list(dict.fromkeys(target_relabel_ecms))
        elif isinstance(target_per_relabel, Mapping):
            target_labels = list(target_per_relabel)
            source_df = source_df.loc[source_df["relabel_ecm"].isin(target_labels)]
        else:
            target_labels = sorted(source_df["relabel_ecm"].dropna().unique())

        target_counts = self.target_count_series(target_per_relabel, target_labels)
        sorted_df = source_df.sort_values(["relabel_ecm", "batch_id", "selected_position"])
        final_parts = [
            sorted_df.loc[sorted_df["relabel_ecm"] == label].head(int(target_counts[label]))
            for label in target_labels
        ]
        final_df = (
            pd.concat(final_parts, ignore_index=True)
            if final_parts
            else sorted_df.iloc[0:0].copy()
        )

        final_df.insert(0, "global_position", np.arange(len(final_df)))

        final_counts = self.relabel_group_counts(final_df, target_labels=target_labels)
        missing = target_counts.subtract(final_counts, fill_value=0)
        missing = missing[missing > 0]

        if not missing.empty:
            raise RuntimeError(
                f"Final balanced export still has under-target groups:\n{missing.to_string()}"
            )

        return final_df

    def build_curve_parameter_export_df(self, final_relabel_df: pd.DataFrame) -> pd.DataFrame:
        """Build the compact CSV export table for final rows.

        Parameters
        ----------
        final_relabel_df : pandas.DataFrame
            Final relabel rows from ``build_final_relabel_df``.

        Returns
        -------
        pandas.DataFrame
            Table containing original ECM, original parameters, final ECM,
            final parameters, frequency grid, and impedance data as JSON
            strings.
        """
        frequency_json = json.dumps([float(value) for value in self.random_ecm_freq])
        rows = []

        for _, row in final_relabel_df.sort_values(
            ["relabel_ecm", "global_position"]
        ).iterrows():
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

            rows.append(
                {
                    "original_ecm": row["original_ecm"],
                    "original_params_json": json.dumps(original_params, sort_keys=True),
                    "final_simplified_ecm": row["relabel_ecm"],
                    "final_simplified_params_json": json.dumps(relabel_params, sort_keys=True),
                    "frequency_hz_json": frequency_json,
                    "impedance_data_json": json.dumps(
                        {
                            "real_ohm": [float(value) for value in np.real(Z)],
                            "imag_ohm": [float(value) for value in np.imag(Z)],
                        }
                    ),
                }
            )

        return pd.DataFrame(rows)

    def export_table_csv(
        self,
        balanced_relabel_results_df: pd.DataFrame,
        target_per_relabel: int | Mapping[str, int] = 5,
        target_relabel_ecms: Optional[Sequence[str]] = None,
        csv_path: Optional[str | Path] = None,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
        """Export final relabelled rows to a CSV table.

        Parameters
        ----------
        balanced_relabel_results_df : pandas.DataFrame
            Balanced relabel results before final trimming.
        target_per_relabel : int or mapping, default=5
            Number of rows to keep per final relabelled ECM.
        target_relabel_ecms : sequence of str, optional
            Optional final label filter.
        csv_path : str or pathlib.Path, optional
            Destination CSV path. Defaults to a source-ECM-derived filename in
            ``self.output_dir``.
        excluded_simplified_ecms : sequence of str or None, optional
            Final simplified/relabelled ECM labels to exclude. ``None`` uses
            ``self.excluded_simplified_ecms``.

        Returns
        -------
        tuple
            Final relabel DataFrame, export DataFrame, and written CSV path.
        """
        final_df = self.build_final_relabel_df(
            balanced_relabel_results_df,
            target_per_relabel=target_per_relabel,
            target_relabel_ecms=target_relabel_ecms,
            excluded_simplified_ecms=excluded_simplified_ecms,
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
        """Export Nyquist plots for final relabelled rows into a ZIP archive.

        Parameters
        ----------
        final_relabel_df : pandas.DataFrame
            Final rows containing ``relabel_ecm`` and ``relabel_params``.
        zip_path : str or pathlib.Path, optional
            Destination ZIP path. Defaults to a source-ECM-derived filename in
            ``self.output_dir``.

        Returns
        -------
        pathlib.Path
            Path to the written ZIP archive.
        """
        import matplotlib.pyplot as plt

        if zip_path is None:
            slug = self.source_ecm_filename_slug(self.random_ecm_circuit)
            zip_path = self.output_dir / f"balanced_eis_plots_{slug}.zip"
        else:
            zip_path = Path(zip_path)

        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for _, row in final_relabel_df.sort_values(
                ["relabel_ecm", "global_position"]
            ).iterrows():
                Z = np.asarray(self.simulate_relabel_impedance(row, self.random_ecm_freq))
                label = str(row["relabel_ecm"])
                safe_label = (
                    re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "unknown_ecm"
                )

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

    def export_eis_plot_folder(
        self,
        final_relabel_df: pd.DataFrame,
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """Export one Nyquist plot PNG per final relabelled row.

        Parameters
        ----------
        final_relabel_df : pandas.DataFrame
            Final rows containing ``relabel_ecm`` and ``relabel_params``.
        output_dir : str or pathlib.Path, optional
            Destination directory. Defaults to a source-ECM-derived plot
            directory in ``self.output_dir``.

        Returns
        -------
        pathlib.Path
            Directory containing one PNG file per generated EIS curve.
        """
        import matplotlib.pyplot as plt

        if output_dir is None:
            slug = self.source_ecm_filename_slug(self.random_ecm_circuit)
            output_dir = self.output_dir / f"eis_plots_{slug}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        for old_png in output_dir.glob("*.png"):
            old_png.unlink()

        for _, row in final_relabel_df.sort_values(
            ["relabel_ecm", "global_position"]
        ).iterrows():
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
            fig.savefig(
                output_dir / f"eis_{int(row['global_position']):04d}_{safe_label}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

        return output_dir

    def export_dataprep_folder(
        self,
        final_relabel_df: pd.DataFrame,
        output_dir: Optional[str | Path] = None,
        use_relabel_simulation: bool = True,
    ) -> Path:
        """Export final rows as AutoREC DataPrep-style curve CSV files.

        Parameters
        ----------
        final_relabel_df : pandas.DataFrame
            Final rows containing relabelled ECM metadata and impedance data.
        output_dir : str or pathlib.Path, optional
            Root output directory. Defaults to ``self.output_dir``.
        use_relabel_simulation : bool, default=True
            If ``True``, regenerate impedance from final relabelled ECMs.
            Otherwise write the original stored ``Z`` values.

        Returns
        -------
        pathlib.Path
            Root directory containing one subdirectory per final ECM label. Each
            subdirectory contains files named ``sample_0000.csv``,
            ``sample_0001.csv``, and so on, with matching pickle metadata files.

        Notes
        -----
        The output is directly compatible with ``EISDataPrep(mode="process")``:
        ``<output_dir>/<relabel_ecm>/sample_N.csv`` with columns ``freq``,
        ``Z_real``, and ``Z_imag``. Frequencies are written in ascending order.
        """

        output_dir = self.output_dir if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sorted_df = final_relabel_df.sort_values(["relabel_ecm", "global_position"])
        freq = self.random_ecm_freq.copy()
        freq_order = np.argsort(freq, kind="stable")
        exported_frequency = freq[freq_order]

        for label, label_df in sorted_df.groupby("relabel_ecm", sort=True, dropna=False):
            label = "unknown_ecm" if pd.isna(label) else str(label)
            safe_label = re.sub(r"[\\/]+", "_", label).strip() or "unknown_ecm"
            label_dir = output_dir / safe_label
            label_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ("sample_*.csv", "sample_*.pkl"):
                for old_export in label_dir.glob(pattern):
                    old_export.unlink()

            for eis_idx, (_, row) in enumerate(label_df.iterrows()):
                if use_relabel_simulation:
                    Z = np.asarray(self.simulate_relabel_impedance(row, self.random_ecm_freq))
                else:
                    Z = np.asarray(row["Z"])

                exported_impedance = Z[freq_order]

                curve_df = pd.DataFrame(
                    {
                        "freq": exported_frequency,
                        "Z_real": np.real(exported_impedance),
                        "Z_imag": np.imag(exported_impedance),
                    }
                )

                sample_stem = f"sample_{eis_idx:04d}"
                curve_df.to_csv(label_dir / f"{sample_stem}.csv", index=False)
                self.write_sample_metadata(
                    row=row,
                    sample_index=eis_idx,
                    metadata_path=label_dir / f"{sample_stem}.pkl",
                    original_frequency=freq,
                    exported_frequency=exported_frequency,
                    exported_impedance=exported_impedance,
                    use_relabel_simulation=use_relabel_simulation,
                )

        return output_dir

    @staticmethod
    def ecm_output_folder_name(ecm: Any) -> str:
        """Return the dynamic output-folder name for an ECM label.

        Parameters
        ----------
        ecm : object
            Final relabelled ECM value. Missing values use ``unknown_ecm``.

        Returns
        -------
        str
            ECM label with only filesystem path separators replaced. Circuit
            brackets, component labels, and indices are preserved.
        """
        if pd.isna(ecm):
            return "unknown_ecm"

        return re.sub(r"[\\/]+", "_", str(ecm)).strip() or "unknown_ecm"

    @staticmethod
    def write_sample_metadata(
        row: pd.Series,
        sample_index: int,
        metadata_path: Path,
        original_frequency: np.ndarray,
        exported_frequency: np.ndarray,
        exported_impedance: np.ndarray,
        use_relabel_simulation: bool,
    ) -> Path:
        """Write metadata with every frequency-aligned array in ascending order."""
        original_frequency = np.asarray(original_frequency, dtype=float)
        original_impedance = np.asarray(row.get("Z"))
        exported_frequency = np.asarray(exported_frequency, dtype=float)
        exported_impedance = np.asarray(exported_impedance)

        if original_frequency.ndim != 1 or exported_frequency.ndim != 1:
            raise ValueError("Metadata frequency arrays must be one-dimensional")
        if original_impedance.shape[:1] != original_frequency.shape:
            raise ValueError(
                "Original frequency and impedance arrays must have the same length"
            )
        if exported_impedance.shape[:1] != exported_frequency.shape:
            raise ValueError(
                "Exported frequency and impedance arrays must have the same length"
            )

        original_order = np.argsort(original_frequency, kind="stable")
        exported_order = np.argsort(exported_frequency, kind="stable")
        original_frequency = original_frequency[original_order].copy()
        original_impedance = original_impedance[original_order].copy()
        exported_frequency = exported_frequency[exported_order].copy()
        exported_impedance = exported_impedance[exported_order].copy()

        normalized_curve = row.get("curve")
        if isinstance(normalized_curve, (list, tuple)) and len(normalized_curve) == 2:
            normalized_curve = tuple(
                np.asarray(axis)[original_order].copy() for axis in normalized_curve
            )

        metadata = {
            "schema_version": 2,
            "sample_index": int(sample_index),
            "initial_circuit": row.get("original_ecm"),
            "initial_parameters": row.get("params"),
            "final_circuit": row.get("relabel_ecm"),
            "final_parameters": row.get("relabel_params"),
            "intermediate": {
                "simplified_circuit": row.get("simplified_ecm"),
                "simplified_parameters": row.get("simplified_params"),
                "fim_relabel_circuit": row.get("fim_relabel_ecm"),
                "fim_relabel_parameters": row.get("fim_relabel_params"),
                "post_fim_simplified_circuit": row.get("post_fim_simplified_ecm"),
                "post_fim_simplified_parameters": row.get("post_fim_simplified_params"),
                "fim_identifiability_info": row.get("fim_identifiability_info"),
                "fim_candidates": row.get("fim_candidates"),
            },
            "provenance": {
                "batch_id": row.get("batch_id"),
                "batch_seed": row.get("batch_seed"),
                "selected_position": row.get("selected_position"),
                "global_position": row.get("global_position"),
            },
            "data": {
                "stored_original_frequency_hz": original_frequency,
                "stored_original_impedance": original_impedance,
                "normalized_curve": normalized_curve,
                "exported_frequency_hz": exported_frequency,
                "exported_impedance": exported_impedance,
                "exported_from_final_simulation": bool(use_relabel_simulation),
            },
            "status": {
                "relabel_failed": bool(row.get("relabel_failed", False)),
                "failure_reason": row.get("failure_reason"),
            },
        }

        metadata_path = Path(metadata_path)
        with metadata_path.open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file, protocol=pickle.HIGHEST_PROTOCOL)

        return metadata_path

    def prepare_eis_data_folder(
        self,
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """Prepare an empty root directory for immediate EIS exports.

        Parameters
        ----------
        output_dir : str or pathlib.Path, optional
            Root output directory. Defaults to ``self.output_dir``.

        Returns
        -------
        pathlib.Path
            Prepared output directory.

        Notes
        -----
        Existing subdirectories are removed at the start of a generation run
        so rerunning generation cannot leave stale ECM groups or sample files.
        Files directly inside the root directory are preserved.
        """
        output_dir = self.output_dir if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        return output_dir

    def export_eis_sample(
        self,
        row: pd.Series,
        sample_index: int,
        output_dir: Optional[str | Path] = None,
        use_relabel_simulation: bool = True,
    ) -> Tuple[Path, Path]:
        """Immediately export one selected EIS sample as CSV and PNG.

        Parameters
        ----------
        row : pandas.Series
            Selected row containing ``relabel_ecm``, ``relabel_params``, and
            stored impedance data.
        sample_index : int
            Zero-based sample number within this row's ECM group.
        output_dir : str or pathlib.Path, optional
            Root output directory. Defaults to ``self.output_dir``.
        use_relabel_simulation : bool, default=True
            If ``True``, simulate impedance from the final relabelled ECM and
            parameters. Otherwise export the stored ``Z`` values.

        Returns
        -------
        tuple of pathlib.Path
            Written CSV path and PNG path. A same-stem pickle metadata file is
            written alongside the CSV.
        """
        import matplotlib.pyplot as plt

        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")

        output_dir = self.output_dir if output_dir is None else Path(output_dir)
        label = "unknown_ecm" if pd.isna(row["relabel_ecm"]) else str(row["relabel_ecm"])
        label_dir = output_dir / self.ecm_output_folder_name(label)
        png_dir = label_dir / "png"
        label_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        if use_relabel_simulation:
            Z = np.asarray(self.simulate_relabel_impedance(row, self.random_ecm_freq))
        else:
            Z = np.asarray(row["Z"])

        freq = self.random_ecm_freq.copy()
        freq_order = np.argsort(freq, kind="stable")
        exported_frequency = freq[freq_order]
        exported_impedance = Z[freq_order]
        curve_df = pd.DataFrame(
            {
                "freq": exported_frequency,
                "Z_real": np.real(exported_impedance),
                "Z_imag": np.imag(exported_impedance),
            }
        )

        sample_stem = f"sample_{sample_index:04d}"
        csv_path = label_dir / f"{sample_stem}.csv"
        metadata_path = label_dir / f"{sample_stem}.pkl"
        png_path = png_dir / f"{sample_stem}.png"
        curve_df.to_csv(csv_path, index=False)
        self.write_sample_metadata(
            row=row,
            sample_index=sample_index,
            metadata_path=metadata_path,
            original_frequency=freq,
            exported_frequency=exported_frequency,
            exported_impedance=exported_impedance,
            use_relabel_simulation=use_relabel_simulation,
        )

        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.plot(np.real(Z), -np.imag(Z), linewidth=1.8)
        ax.set_title(label, fontsize=10, wrap=True)
        ax.set_xlabel("Re(Z) / ohm")
        ax.set_ylabel("-Im(Z) / ohm")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return csv_path, png_path

    def export_eis_data_folder(
        self,
        final_relabel_df: pd.DataFrame,
        output_dir: Optional[str | Path] = None,
        use_relabel_simulation: bool = True,
    ) -> Path:
        """Export generated EIS data as per-ECM CSV and PNG folders.

        Parameters
        ----------
        final_relabel_df : pandas.DataFrame
            Final rows containing relabelled ECM metadata and impedance data.
        output_dir : str or pathlib.Path, optional
            Root output directory. Defaults to ``self.output_dir``.
        use_relabel_simulation : bool, default=True
            If ``True``, regenerate impedance from final relabelled ECMs.
            Otherwise write and plot the original stored ``Z`` values.

        Returns
        -------
        pathlib.Path
            Root directory with ``<relabel_ecm>/sample_N.csv``, matching
            ``sample_N.pkl`` metadata, and ``<relabel_ecm>/png/sample_N.png``
            for each generated curve.
        """
        output_dir = self.prepare_eis_data_folder(output_dir)
        sorted_df = final_relabel_df.sort_values(["relabel_ecm", "global_position"])
        sample_counts: Dict[str, int] = {}

        for _, row in sorted_df.iterrows():
            label = self.ecm_output_folder_name(row["relabel_ecm"])
            sample_index = sample_counts.get(label, 0)
            self.export_eis_sample(
                row,
                sample_index=sample_index,
                output_dir=output_dir,
                use_relabel_simulation=use_relabel_simulation,
            )
            sample_counts[label] = sample_index + 1

        return output_dir

    def generate_data(
        self,
        target_num: int = 150,
        min_batches: int = 1,
        max_batches: int = 200,
        seed_start: int = 1000,
        n_random_candidates: Optional[int] = None,
        max_selected_curves: Optional[int] = None,
        export: bool = True,
        export_plots: bool = False,
        export_dataprep: bool = False,
        live_plot: bool = True,
        excluded_simplified_ecms: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Generate the same target number of samples for each discovered ECM."""
        if not isinstance(target_num, int) or target_num < 1:
            raise ValueError("target_num must be a positive integer.")

        balanced_df, batch_infos = self.run_balanced_relabel_dataset(
            target_per_relabel=target_num,
            target_relabel_ecms=None,
            excluded_simplified_ecms=excluded_simplified_ecms,
            min_batches=min_batches,
            max_batches=max_batches,
            seed_start=seed_start,
            n_random_candidates=n_random_candidates,
            max_selected_curves=max_selected_curves,
            export_selected_samples=export_dataprep,
            export_output_dir=self.output_dir,
            live_plot=live_plot,
        )

        target_labels = list(batch_infos[0].get("added_counts", {}).keys())
        if not target_labels:
            target_labels = list(self.relabel_group_counts(balanced_df).index)

        if export:
            final_df, _, _ = self.export_table_csv(
                balanced_df,
                target_per_relabel=target_num,
                target_relabel_ecms=target_labels,
                excluded_simplified_ecms=excluded_simplified_ecms,
            )

            if export_plots:
                self.export_eis_plot_folder(final_df)

        return balanced_df, batch_infos
