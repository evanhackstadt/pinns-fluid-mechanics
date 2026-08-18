# data.py

"""
2D Stenosis Geometry-Conditioned PINN
    Functions to assemble multi-geometry dataset for PINN training.
    DeepXDE model can't take multiple geometries, so create PointSet

Evan Hackstadt
Rugonyi Lab
"""

import numpy as np
import torch
import deepxde as dde



# ———————————————— LABELED DATASET ————————————————

# --- Sample Labeled Data Points (Single Geometry) ---
def _sample_labeled_points_single(fem_data, n, cfg, 
                                  components = [0, 1, 2],
                                  coordinates: list[tuple] = None):
    """
    Build a single nested sequence of n labeled points for one geometry.
    Returns subset of fem_data, shape (n, 2 + len(components)), ordered by selection priority. 
    Each prefix of length n is the labeled set for that n.
    """
    if n <= 0:
        raise ValueError(f"n = {n} (must be > 0)")

    M = len(fem_data)
    u, v, p = fem_data[:, 2], fem_data[:, 3], fem_data[:, 4]
    
    if coordinates:
        
        all_idx = []
        for (x, y) in coordinates:
            # get closest fem_data value to requested coordinate
            distances = np.abs(fem_data[:, 0] - x) + np.abs(fem_data[:, 1] - y)
            idx = (distances).argmin(axis=0)
            all_idx.append(idx)
            
    else:
        # Approximate pointwise gradient magnitudes for requested components
        # using finite differences on the flat (unstructured) masked array. 
        # This is a rough proxy.
        du = dv = dp = 0
        if 0 in components:
            du = np.abs(np.gradient(u)) + np.abs(np.gradient(np.gradient(u)))
        if 1 in components:
            dv = np.abs(np.gradient(v)) + np.abs(np.gradient(np.gradient(v)))
        if 2 in components:
            dp = np.abs(np.gradient(p))

        # Clip outliers before normalizing to avoid one extreme point dominating
        raw_scores = np.clip(du + dv + dp, 0, np.percentile(du + dv + dp, 95))
        scores = raw_scores / raw_scores.sum()

        generator = np.random.default_rng(seed=cfg.seed)
        n_scored  = int(n * (1 - cfg.uniform_frac))
        n_uniform = n - n_scored

        # Sample scored indices first
        idx_scored = generator.choice(M, size=n_scored, replace=False, p=scores)
        
        # Sample uniform indices from remaining (excluding scored)
        remaining_indices = np.setdiff1d(np.arange(M), idx_scored)
        idx_uniform = generator.choice(remaining_indices, size=n_uniform, replace=False)

        # Combine: scored first, then uniform (guaranteed exactly n points)
        all_idx = np.concatenate([idx_scored, idx_uniform])
        
    # select and copy the labeled points
    labeled_pts = fem_data[all_idx].copy()

    # create a NaN column for excluded components
    nan_col = np.full(labeled_pts.shape[0], np.nan)

    # components: 0->u (col 2), 1->v (col 3), 2->p (col 4)
    for c in [0, 1, 2]:
        if c not in components:
            labeled_pts[:, c + 2] = nan_col
    
    return labeled_pts


# --- Entry Point ---
def build_labeled_data_dict(fem_data_dict, n, sensors, cfg, 
                            components = [0, 1, 2],
                            coordinates: list[tuple] = None):
    """
    Concatenate labeled datasets (n points per geometry) across all geometries in the fem data.
    Blind to train/test split; fem_data_dict should only contain the desired set of geometries.
    Args:
        fem_data_dict: dictionary mapping geo_tag --> fem_data array of shape (N, 5) = [x,y,u,v,p]
        n: the number of labeled points to sample per geometry
        cfg: custom config object
        components: list of components to sample (u=0, v=1, p=2)
        coordinates: optionally specify exact coordinates from which to create labeled points.
    Returns:
        labeled_data_dict: dict mapping (a,b) --> {
            "query": array of (n, 2) = [x,y],
            "targets": array of (n, 3) = [u,v,p]
        }
    """
    
    labeled_data = {}
    
    for (a, b), fem_data in fem_data_dict.items():
        
        labeled_pts = _sample_labeled_points_single(fem_data, n, cfg, components, coordinates)
        labeled_data[(a, b)] = {
            "query":   fem_data[:, 0:2],
            "targets": fem_data[:, 2:5],
        }
    
    return labeled_data