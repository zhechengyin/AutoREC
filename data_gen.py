"""Data generation utilities and pipeline helpers.

This module defines `data_gen` which collects utility functions used across the
notebook workflow (relabeling, normalization, postprocessing). PCA/UMAP
functionality is intentionally excluded from this consolidation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class data_gen:
    """End-to-end EIS dataset generation pipeline helpers.

    Pipeline:
        sample_params
        -> simulate
        -> filter
        -> select
        -> relabel
        -> postprocess
        -> validate
        -> balance
        -> export
    """

    @staticmethod
    def source_ecm_filename_slug(circuit: str) -> str:
        source = str(circuit).strip()
        readable = re.sub(r"\s+", "", source)
        readable = re.sub(r"[^A-Za-z0-9._,\[\]-]+", "_", readable).strip("._-")
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        return f"{readable}_{digest}" if readable else f"unknown_ecm_{digest}"

    @staticmethod
    def relabel_group_counts(df: pd.DataFrame, target_labels: Optional[List[str]] = None) -> pd.Series:
        source = df.dropna(subset=["relabel_ecm"]).copy()
        if "relabel_failed" in source.columns:
            source = source.loc[~source["relabel_failed"]]
        counts = source["relabel_ecm"].value_counts().sort_index()
        if target_labels is not None:
            counts = counts.reindex(target_labels, fill_value=0)
        return counts

    @staticmethod
    def choose_fim_relabel(fim_result: Any, fallback_circuit: str, fallback_params: Any) -> Tuple[Any, Any, Any]:
        if isinstance(fim_result, str):
            return fim_result, fallback_params, []
        if isinstance(fim_result, list) and fim_result:
            first = fim_result[0]
            return (first[0], first[1], fim_result) if isinstance(first, tuple) else (first, None, fim_result)
        return fallback_circuit, fallback_params, fim_result

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

    @staticmethod
    def reorder_parallel_blocks_and_series_p(circuit: Optional[str]) -> Optional[str]:
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
            items = data_gen.split_top_level(circuit[start + 1 : i - 1])
            items = [data_gen.reorder_parallel_blocks_and_series_p(item) for item in items]
            items = sorted(items, key=data_gen._parallel_sort_key)
            out.append("[" + ",".join(items) + "]")

        circuit = "".join(out)
        series_parts = data_gen.split_top_level(circuit, sep="-")
        p_parts = [part for part in series_parts if re.fullmatch(r"\s*P\d+\s*", part)]
        non_p_parts = [part for part in series_parts if not re.fullmatch(r"\s*P\d+\s*", part)]
        return "-".join(non_p_parts + p_parts)

    @staticmethod
    def convert_capacitors_to_cpes(circuit: Optional[str], params: Any = None) -> Tuple[Optional[str], Any]:
        if circuit is None:
            return circuit, params
        circuit = re.sub(r"C(\d+)", r"P\1", str(circuit))
        if not isinstance(params, dict):
            return circuit, params

        converted: Dict[str, Any] = {}
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

    @staticmethod
    def ensure_series_r1(circuit: Optional[str], params: Any = None) -> Tuple[Optional[str], Any]:
        # strip leading R0 if present and drop 'R0' param
        if circuit is None:
            return circuit, params
        circuit = str(circuit)
        if circuit.startswith("R0-"):
            circuit = circuit[3:]
        if isinstance(params, dict):
            params = dict(params)
            params.pop("R0", None)
        return circuit, params

    @staticmethod
    def reindex_components(circuit: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if circuit is None:
            return circuit, {}
        mapping: Dict[str, str] = {}
        for match in re.finditer(r"([A-Z])(\d+)", circuit):
            mapping.setdefault(match.group(0), f"{match.group(1)}{len(mapping) + 1}")
        replaced = re.sub(r"([A-Z])(\d+)", lambda m: mapping[m.group(0)], circuit)
        return replaced, mapping

    @staticmethod
    def reindex_parameter_dict(params: Any, mapping: Dict[str, str]) -> Any:
        if not isinstance(params, dict):
            return params
        renamed: Dict[str, Any] = {}
        for key, value in params.items():
            match = re.match(r"([A-Z]\d+)(.*)", str(key))
            new_key = mapping.get(match.group(1), match.group(1)) + match.group(2) if match else key
            renamed[new_key] = value
        return renamed

    @staticmethod
    def normalize_circuit_and_params(circuit: Optional[str], params: Any = None) -> Tuple[Optional[str], Any]:
        if circuit is None:
            return circuit, params
        circuit, params = data_gen.convert_capacitors_to_cpes(circuit, params)
        circuit = data_gen.reorder_parallel_blocks_and_series_p(circuit)
        circuit, params = data_gen.ensure_series_r1(circuit, params)
        circuit, mapping = data_gen.reindex_components(circuit)
        return circuit, data_gen.reindex_parameter_dict(params, mapping)

    @staticmethod
    def postprocess_relabels(df: pd.DataFrame) -> pd.DataFrame:
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
                data_gen.normalize_circuit_and_params(row[circuit_col], row.get(params_col) if params_col in df.columns else None)
                for _, row in df.iterrows()
            ]
            df[circuit_col] = [item[0] for item in normalized]
            if params_col in df.columns:
                df[params_col] = [item[1] for item in normalized]
        return df



    
    