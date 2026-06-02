import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
import warnings
import itertools # generate all possible combinations of elements in lists
import re
from tqdm import tqdm
import random
import pickle
import os


random.seed(42)
np.random.seed(42)

# Suppress the specific warning from impedance.py
warnings.filterwarnings("ignore", category=UserWarning, 
                        message="Simulating circuit based on initial parameters")


def simulate_ecm(params, frequencies=frequencies):
    initial_guess = [float(value) for value in params]
    try:
        model = CustomCircuit(circuit=circuit, initial_guess=initial_guess)
        return model.predict(frequencies)
    except Exception:
        return None

def normalize_curve(Z):
    real = Z.real
    imag = Z.imag
    real_min, real_max = real.min(), real.max()
    imag_min, imag_max = imag.min(), imag.max()

    if real_max > real_min:
        real_norm = (real - real_min) / (real_max - real_min)
    else:
        real_norm = np.zeros_like(real)

    if imag_max > imag_min:
        imag_norm = (imag - imag_min) / (imag_max - imag_min)
    else:
        imag_norm = np.zeros_like(imag)

    return real_norm, imag_norm

def stack_curves(curves):
    if not curves:
        return np.empty((0, 0, 2), dtype=np.float32)

    n_freq = len(curves[0][0])
    curve_points = np.empty((len(curves), n_freq, 2), dtype=np.float32)
    for idx, (re_norm, im_norm) in enumerate(curves):
        if len(re_norm) != n_freq or len(im_norm) != n_freq:
            raise ValueError('All curves must use the same frequency grid')
        curve_points[idx, :, 0] = re_norm
        curve_points[idx, :, 1] = im_norm

    return curve_points


def distances_to_curve(curve_points, reference_curve):
    diff = curve_points - reference_curve[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).mean(axis=1)


def curve_distance(curve_a, curve_b):
    points_a = np.vstack(curve_a).T
    points_b = np.vstack(curve_b).T
    return np.mean(np.linalg.norm(points_a - points_b, axis=1))

def sample_candidates(n_candidates=400, seed=42):
    rng = np.random.default_rng(seed)
    candidates = []

    r0_bounds = (1e0, 1e5)
    r_bounds = (1e0, 1e6)
    cpe_bounds = (1e-8, 1e-3)
    alpha_bounds = (0.5, 1.0)

    for _ in range(n_candidates):
        r0 = 10 ** rng.uniform(np.log10(r0_bounds[0]), np.log10(r0_bounds[1]))
        r1 = 10 ** rng.uniform(np.log10(r_bounds[0]), np.log10(r_bounds[1]))
        r2 = 10 ** rng.uniform(np.log10(r_bounds[0]), np.log10(r_bounds[1]))

        cpe1 = 10 ** rng.uniform(np.log10(cpe_bounds[0]), np.log10(cpe_bounds[1]))
        cpe2 = 10 ** rng.uniform(np.log10(cpe_bounds[0]), np.log10(cpe_bounds[1]))

        alpha1 = rng.uniform(alpha_bounds[0], alpha_bounds[1])
        alpha2 = rng.uniform(alpha_bounds[0], alpha_bounds[1])

        candidates.append([r0, cpe1, alpha1, r1, cpe2, alpha2, r2])

    return candidates

def evaluate_candidates(candidates):
    param_list = []
    curves = []

    for params in tqdm(candidates, desc='Simulating candidates'):
        Z = simulate_ecm(params)
        if Z is None:
            continue
        curves.append(normalize_curve(Z))
        param_list.append(params)

    return param_list, curves


def high_frequency_minus_im(curve, high_frequency_index=0):
    return -float(curve[1][high_frequency_index])


def filter_high_frequency_curves(
    param_list,
    curves,
    max_high_frequency_minus_im=-0.8,
    high_frequency_index=0,
):
    high_frequency_values = np.asarray([
        high_frequency_minus_im(curve, high_frequency_index=high_frequency_index)
        for curve in curves
    ])
    keep_mask = high_frequency_values <= max_high_frequency_minus_im
    kept_idx = np.flatnonzero(keep_mask)
    removed_idx = np.flatnonzero(~keep_mask)

    filtered_params = [param_list[i] for i in kept_idx]
    filtered_curves = [curves[i] for i in kept_idx]

    filter_info = {
        'max_high_frequency_minus_im': max_high_frequency_minus_im,
        'high_frequency_index': high_frequency_index,
        'raw_valid_candidate_count': len(param_list),
        'filtered_candidate_count': len(filtered_params),
        'removed_candidate_count': len(removed_idx),
        'kept_original_idx': kept_idx.tolist(),
        'removed_original_idx': removed_idx.tolist(),
        'high_frequency_minus_im': high_frequency_values.tolist(),
    }

    if not filtered_params:
        raise ValueError(
            'High-frequency filter removed every candidate. '
            'Lower max_high_frequency_minus_im or broaden the candidate sampling range.'
        )

    return filtered_params, filtered_curves, filter_info

def choose_initial_curve(curve_points, start='farthest_from_center'):
    if isinstance(start, (int, np.integer)):
        if start < 0 or start >= len(curve_points):
            raise IndexError('start index is outside the candidate range')
        return int(start)

    if start == 'farthest_from_center':
        center_curve = curve_points.mean(axis=0)
        return int(np.argmax(distances_to_curve(curve_points, center_curve)))

    if start == 'random':
        return int(np.random.default_rng(0).integers(len(curve_points)))

    raise ValueError("start must be an index, 'farthest_from_center', or 'random'")


def greedy_select(
    param_list,
    curves,
    k=30,
    start='farthest_from_center',
    min_distance=None,
    return_scores=False,
):
    if not param_list or k <= 0:
        if return_scores:
            return [], [], []
        return [], []

    curve_points = stack_curves(curves)
    n_select = min(k, len(param_list))

    first_idx = choose_initial_curve(curve_points, start=start)
    selected_idx = [first_idx]
    selection_scores = [np.inf]

    min_dist_to_selected = distances_to_curve(curve_points, curve_points[first_idx])
    min_dist_to_selected[first_idx] = -np.inf

    while len(selected_idx) < n_select:
        best_idx = int(np.argmax(min_dist_to_selected))
        best_score = float(min_dist_to_selected[best_idx])

        if min_distance is not None and best_score < min_distance:
            break

        selected_idx.append(best_idx)
        selection_scores.append(best_score)

        new_dist = distances_to_curve(curve_points, curve_points[best_idx])
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_dist)
        min_dist_to_selected[selected_idx] = -np.inf

    selected_params = [param_list[i] for i in selected_idx]
    if return_scores:
        return selected_params, selected_idx, selection_scores
    return selected_params, selected_idx

def plot_selected(curves, selected_idx, outpath):
    plt.figure(figsize=(8, 6))
    for idx, (re_norm, im_norm) in enumerate(curves):
        if idx in selected_idx:
            plt.plot(re_norm, -im_norm, linewidth=2.5, alpha=0.95)
        # else:
        #     plt.plot(re_norm, -im_norm, linewidth=0.8, color='gray', alpha=0.25)

    plt.xlabel('Normalized Re(Z)')
    plt.ylabel('Normalized -Im(Z)')
    plt.title('Normalized Nyquist curves: diverse parameter sets')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.show()


def transform_parameters_for_pca(param_list):
    params = np.asarray(param_list, dtype=float)
    feature_names = np.array([
        'log10(R0)', 'log10(CPE1)', 'alpha1',
        'log10(R1)', 'log10(CPE2)', 'alpha2', 'log10(R2)',
    ])
    log_columns = [0, 1, 3, 4, 6]

    transformed = params.copy()
    transformed[:, log_columns] = np.log10(transformed[:, log_columns])
    return transformed, feature_names


def plot_parameter_pca_2d(param_list, selected_idx, outpath):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if len(param_list) < 2:
        print('Skipping PCA plot: need at least two parameter sets')
        return None

    X, feature_names = transform_parameters_for_pca(param_list)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    selected_idx = np.asarray(selected_idx, dtype=int)
    explained = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(7.5, 6))
    plt.scatter(
        scores[:, 0], scores[:, 1],
        s=24, color='0.78', alpha=0.55, label='All valid candidates',
    )
    plt.scatter(
        scores[selected_idx, 0], scores[selected_idx, 1],
        s=44, color='tab:red', edgecolor='black', linewidth=0.35,
        alpha=0.9, label='Selected curves',
    )

    pc1_top = feature_names[np.argsort(np.abs(pca.components_[0]))[::-1][:3]]
    pc2_top = feature_names[np.argsort(np.abs(pca.components_[1]))[::-1][:3]]
    loading_text = (
        'PC1 top loadings: ' + ', '.join(pc1_top) + '\n'
        'PC2 top loadings: ' + ', '.join(pc2_top)
    )
    plt.gca().text(
        0.02, 0.98, loading_text,
        transform=plt.gca().transAxes,
        va='top', ha='left', fontsize=9,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '0.85', 'alpha': 0.9},
    )

    plt.xlabel(f'PC1 ({explained[0]:.1f}% variance)')
    plt.ylabel(f'PC2 ({explained[1]:.1f}% variance)')
    plt.title('2D PCA of element parameters')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.show()

    return {
        'scores': scores,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'feature_names': feature_names.tolist(),
    }


def plot_curve_pca_2d(curves, selected_idx, outpath):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if len(curves) < 2:
        print('Skipping curve PCA plot: need at least two curves')
        return None

    curve_points = stack_curves(curves)
    X = curve_points.reshape(len(curves), -1)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    selected_idx = np.asarray(selected_idx, dtype=int)
    explained = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(7.5, 6))
    plt.scatter(
        scores[:, 0], scores[:, 1],
        s=24, color='0.78', alpha=0.55, label='All valid candidates',
    )
    plt.scatter(
        scores[selected_idx, 0], scores[selected_idx, 1],
        s=44, color='tab:red', edgecolor='black', linewidth=0.35,
        alpha=0.9, label='Selected curves',
    )

    plt.xlabel(f'PC1 ({explained[0]:.1f}% variance)')
    plt.ylabel(f'PC2 ({explained[1]:.1f}% variance)')
    plt.title('2D PCA of normalized Nyquist curves')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.show()

    return {
        'scores': scores,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
    }