from pprint import pprint
import re
from itertools import product

import numpy as np
import autoeis as ae

from .fim_utils import compute_fim
from .eis_objectives import EISObjective


_EMPTY = object()


def suggest_unidentifiable_params(
    circuit,
    freq,
    Z,
    params,
    Rct_to_Z_scale=0.01,
    identifiability_thresh=1e-4,
    participation_thresh=0.2,
    mode="bode",
    verbose=False,
    full_output=False,
):
    """
    Suggest parameters in the circuit that possibly be unidentifiable and can be dropped.

    The analysis is based on the FIM of the circuit parameters. The parameters that have
    low eigenvalues in the FIM are likely to be unidentifiable, and the corresponding
    eigenvectors can be used to identify which parameters are likely to be unidentifiable.

    Parameters
    ----------
    circuit : str
        The circuit string.
    freq : array-like
        The frequencies at which the impedance is measured.
    Z : array-like
        The measured impedance values at the given frequencies.
    params : dict
        The dictionary of circuit parameters and their values.
    Rct_to_Z_scale : float, optional
        The scale factor to shift the ohmic resistor value. If Rct is too small, it can be
        falsely identified as an unidentifiable parameter. By shifting it to be comparable
        to the impedance values, we can avoid this issue.
    identifiability_thresh : float, optional
        The threshold for identifying unidentifiable parameters based on the eigenvalues
        of the FIM. Directions corresponding to eigenvalues below this threshold are
        considered unidentifiable.
    participation_thresh : float, optional
        The threshold for identifying which parameters are likely to be unidentifiable based on
        the eigenvectors of the FIM. Parameters with a squared participation above this
        threshold in the unidentifiable directions are considered candidates for
        redundancy.
    mode : {'nyquist', 'bode'}, optional
        The mode for evaluating the circuit function. 'nyquist' uses the real and
        imaginary parts of the impedance, while 'bode' uses the magnitude and phase.
        The choice of mode can affect the sensitivity of the FIM analysis to different
        parameters.
    verbose : bool, optional
        Whether to print detailed information about the analysis.
    full_output : bool, optional
        Whether to return the full analysis results including eigenvalues and
        eigenvectors.

    Returns
    -------
    suggestions : list of tuples
        A list of tuples, where each tuple contains the parameters that are suggested to
        be dropped together.
    """
    param_names = ae.parser.get_parameter_labels(circuit)
    params_array = np.array([params[name] for name in param_names])
    # Shift ohmic resistor as needed
    try:
        Rct = ae.parser.find_ohmic_resistors(circuit)[0]
        Rct_val = params[Rct]
        Rct_idx = param_names.index(Rct)
        Rct_shift = (
            Rct_to_Z_scale * max(Z.real) if Rct_val <= Rct_to_Z_scale * max(Z.real) else 0.0
        )
        params_array[Rct_idx] += Rct_shift
    except IndexError:
        # No ohmic resistor found, do nothing
        pass
    if verbose:
        print("Circuit:", circuit)
        print("Original parameters:")
        pprint(params, sort_dicts=False)

    # Compute FIM
    fim, eigvals, eigvecs = compute_fim(circuit, freq, Z, params_array, mode, verbose)

    # Use FIM to identify components that can possibly be dropped
    nparams = len(params)
    fim_rank = sum(eigvals > identifiability_thresh)
    if verbose:
        print(f"FIM rank: {fim_rank} (# parameters:  {nparams})")
    suggestions = []
    if fim_rank < nparams:
        # Find possibly unidentifiable component
        nunidentifiable = sum(eigvals < identifiability_thresh)
        idx_red = np.arange(nunidentifiable)
        # First, let's find possible suggestions
        red_params = []
        for ir in idx_red:
            part = eigvecs[:, ir] ** 2
            idx_params = np.where(part > participation_thresh)[0]
            candidate_params = [param_names[ip] for ip in idx_params]
            # For P*w and P*n, we just need to list them as one parameter as P*.
            candidate_params_ = []
            for cp in candidate_params:
                m = re.fullmatch(r"P(\d+)[wn]", cp)
                if m:
                    p = f"P{m.group(1)}"
                else:
                    p = cp
                if p not in candidate_params_:
                    candidate_params_.append(p)
            red_params.append(candidate_params_)
        suggestions = list(product(*red_params))

    if verbose and suggestions:
        print("Suggested parameters to drop together:")
        for s in suggestions:
            print(s)
    if full_output:
        return suggestions, eigvals, eigvecs
    else:
        return suggestions


def simplify_structure_after_drop(
    structure,
    drop_components,
    empty_parallel_policy="remove",
):
    """
    Remove selected components from an AutoEIS circuit structure and simplify it.

    Parameters
    ----------
    structure : str or list
        AutoEIS parsed circuit structure, e.g.
        ['s', 'R7', ['p', ['s', 'C1', 'R2'], ['s', 'R3', ['p', 'R5', 'P6']]]]

    drop_components : list[str] or set[str]
        Components to remove, e.g. ['R2', 'R5'].

    empty_parallel_policy : {'remove', 'short'}
        What to do when an entire branch inside a parallel block disappears.

        'remove':
            Remove the empty branch.

        'short':
            Treat the empty branch as a short circuit, so the whole parallel
            block collapses.

    Returns
    -------
    simplified_structure : str or list
        Simplified AutoEIS-compatible structure.
    """

    drop_components = set(drop_components)

    # Leaf node: a component such as 'R2', 'C1', 'P6'
    if isinstance(structure, str):
        if structure in drop_components:
            return _EMPTY
        return structure

    if not isinstance(structure, list):
        raise TypeError(f"Unexpected structure type: {type(structure)}")

    op = structure[0]
    children = structure[1:]

    if op not in {"s", "p"}:
        raise ValueError(f"Unknown operator {op!r}. Expected 's' or 'p'.")

    simplified_children = []

    for child in children:
        new_child = simplify_structure_after_drop(
            child,
            drop_components,
            empty_parallel_policy=empty_parallel_policy,
        )

        if new_child is _EMPTY:
            if op == "p" and empty_parallel_policy == "short":
                return _EMPTY

            # For series, removing a component means that segment disappears.
            # For parallel with policy='remove', the branch is removed.
            continue

        # Flatten nested series: ['s', 'R1', ['s', 'R2', 'R3']]
        # becomes ['s', 'R1', 'R2', 'R3']
        if op == "s" and isinstance(new_child, list) and new_child[0] == "s":
            simplified_children.extend(new_child[1:])

        # Flatten nested parallel: ['p', 'R1', ['p', 'R2', 'R3']]
        # becomes ['p', 'R1', 'R2', 'R3']
        elif op == "p" and isinstance(new_child, list) and new_child[0] == "p":
            simplified_children.extend(new_child[1:])

        else:
            simplified_children.append(new_child)

    # Nothing remains
    if len(simplified_children) == 0:
        return _EMPTY

    # Degenerate series/parallel with only one child should collapse
    if len(simplified_children) == 1:
        return simplified_children[0]

    return [op, *simplified_children]


def drop_components_from_circuit(
    circuit,
    drop_components,
    empty_parallel_policy="remove",
):
    """
    Drop components from an AutoEIS circuit string.

    Parameters
    ----------
    circuit : str
        Circuit string, e.g. 'R7-[C1-R2,R3-[R5,P6]]'.

    drop_components : list[str]
        Components to remove, e.g. ['R2', 'R5'].

    empty_parallel_policy : {'remove', 'short'}
        Rule for handling empty branches in parallel blocks.

    Returns
    -------
    simplified_circuit : str
        Simplified circuit string with specified components removed.
    """

    structure = ae.parser._parse_to_structure(circuit)

    simplified_structure = simplify_structure_after_drop(
        structure,
        drop_components,
        empty_parallel_policy=empty_parallel_policy,
    )

    if simplified_structure is _EMPTY:
        raise ValueError("All components were removed, so the resulting circuit is empty.")

    simplified_circuit = ae.parser._stringify_structure(simplified_structure)

    return simplified_circuit


def simplify_unidentifiable_components(
    circuit: str,
    freq: np.ndarray,
    Z: np.ndarray,
    params: dict,
    Rct_to_Z_scale=0.01,
    identifiability_thresh=1e-4,
    participation_thresh=0.2,
    empty_parallel_policy="remove",
    refit_ecm=True,
    fit_kwargs=None,
    verbose=False,
):
    """
    Suggest unidentifiable parameters/components to drop and return the simplified circuit.

    This function combines the analysis of the FIM to identify unidentifiable parameters
    and the simplification of the circuit structure by dropping those parameters.

    Parameters
    ----------
    circuit : str
        The original circuit string.
    freq : array-like
        The frequencies at which the impedance is measured.
    Z : array-like
        The measured impedance values at the given frequencies.
    params : dict
        The dictionary of circuit parameters and their values.
    Rct_to_Z_scale : float, optional
        The scale factor to shift the ohmic resistor value for FIM analysis.
    identifiability_thresh : float, optional
        The threshold for identifying unidentifiable parameters based on FIM eigenvalues.
    participation_thresh : float, optional
        The threshold for identifying unidentifiable parameters based on FIM eigenvectors.
    empty_parallel_policy : {'remove', 'short'}, optional
        Rule for handling empty branches in parallel blocks when simplifying the circuit.
    refit_ecm : bool, optional
        Whether to refit the simplified circuit to the data and return the new parameters.
    fit_kwargs : dict, optional
        Keyword arguments forwarded to ae.utils.fit_circuit_parameters.
    verbose : bool, optional
        Whether to print detailed information about the analysis and simplification.

    Returns
    -------
    list
        A list of possible simplified circuits. If `refit_ecm` is True, each element is a
        tuple of (simplified_circuit, simplified_params). Otherwise, each element is just the
        simplified_circuit string.
    """

    suggestions = suggest_unidentifiable_params(
        circuit,
        freq,
        Z,
        params,
        Rct_to_Z_scale=Rct_to_Z_scale,
        identifiability_thresh=identifiability_thresh,
        participation_thresh=participation_thresh,
        verbose=verbose,
    )

    if not suggestions:
        if verbose:
            print("No unidentifiable parameters suggested to drop.")
        if refit_ecm:
            return [(circuit, params)]
        else:
            return [circuit]

    simplified_circuits = []
    for drop_components in suggestions:
        simplified_circuit = drop_components_from_circuit(
            circuit,
            drop_components,
            empty_parallel_policy=empty_parallel_policy,
        )

        if verbose:
            print(f"Simplified circuit after dropping {drop_components}: {simplified_circuit}")

        if refit_ecm:
            # Optionally, we can fit the simplified circuit to the data and get the new
            # parameters.
            fit_kwargs = {} if fit_kwargs is None else fit_kwargs
            p0 = np.array(
                [params[name] for name in ae.parser.get_parameter_labels(simplified_circuit)]
            )
            simplified_params = ae.utils.fit_circuit_parameters(
                simplified_circuit, freq, Z, p0, **fit_kwargs
            )
            # Compute metrics
            if verbose:
                print("Fitted parameters:")
                pprint(simplified_params, sort_dicts=False)
                chi2 = np.mean(
                    EISObjective(simplified_circuit, freq, Z, method="chi-squared")(
                        np.array([val for val in simplified_params.values()])
                    )
                )
                print(f"Chi-squared error: {chi2:.4e}")
                print()
            # Only append if the simplified circuit is not already in the list
            if simplified_circuit not in [sc[0] for sc in simplified_circuits]:
                simplified_circuits.append((simplified_circuit, simplified_params))
        else:
            if simplified_circuit not in simplified_circuits:
                simplified_circuits.append(simplified_circuit)

    return simplified_circuits
