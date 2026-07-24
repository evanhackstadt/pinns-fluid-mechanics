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


# ———————————————— DOMAIN DATASET ————————————————

# --- Sample Domain Points - Single Geometry ---
def sample_domain_points(cfg, a, b):
    """
    Helper function that constructs a given geometry, samples interior and boundary points,
    concatenates the given (a,b) to the data, and returns arrays.
    Returns:
        interior_data: array of shape (n_interior, 4) = [x,y,a,b]
        boundary_data: array of shape (n_boundary, 4) = [x,y,a,b]
    """
    
    # Construct the geometry: base channel rectangle - obstructing ellipse
    channel = dde.geometry.Rectangle([-cfg.L/2, 0], [cfg.L/2, cfg.H_max])
    obstruction = dde.geometry.Ellipse([cfg.x_c, cfg.y_c], a, b)    # NOTE: DeepXDE requires a>b, workaround not implemented
    geometry = dde.geometry.CSGDifference(channel, obstruction)
    
    # sample coordinate points on domain
    interior_xy = geometry.random_points(cfg.n_interior)
    boundary_xy = geometry.random_boundary_points(cfg.n_boundary)
    
    # add (a,b) to every (x,y) point since network takes (x,y,a,b) as inputs
    interior_ab = np.full((interior_xy.shape[0], 2), [a, b])
    boundary_ab = np.full((boundary_xy.shape[0], 2), [a, b])
    
    interior_data = np.concatenate([interior_xy, interior_ab], axis=1)
    boundary_data = np.concatenate([boundary_xy, boundary_ab], axis=1)
    
    return interior_data, boundary_data


# --- Create Manual Domain Dataset - All Train Geometries ---
def build_interior_dataset(cfg):
    """
    Concatenate domain points across all training geometries.
    Returns:
        all_interior_data: array of shape (n_interior * n_train_geometries, 4) = [x,y,a,b]. 
        all_boundary_data: array of shape (n_boundary * n_train_geometries, 4) = [x,y,a,b]. 
    """
    interior_list = []
    boundary_list = []
    
    for (a, b) in cfg.train_geometries:
        interior_data, boundary_data = sample_domain_points(cfg, a, b)
        interior_list.append(interior_data)
        boundary_list.append(boundary_data)
    
    all_interior_data = np.concatenate(interior_list, axis=0)
    all_boundary_data = np.concatenate(boundary_list, axis=0)
    
    return all_interior_data, all_boundary_data



# ———————————————— LABELED DATASET ————————————————


# --- Sample Labeled Data Points ---
def sample_labeled_points(fem_data, n_max, cfg):
    """
    Build a single nested sequence of up to n_max labeled points for one geometry.
    Returns subset of fem_data, shape (n_max, 5), ordered by selection priority. 
    Each prefix of length n is the labeled set for that n.
    """
    if n_max <= 0:
        assert f"n = {n_max} (must be > 0)"

    M = len(fem_data)
    u, v, p = fem_data[:, 2], fem_data[:, 3], fem_data[:, 4]

    # Approximate pointwise gradient magnitudes using finite differences on
    # the flat (unstructured) masked array. This is a rough proxy.
    du = np.abs(np.gradient(u)) + np.abs(np.gradient(np.gradient(u)))
    dv = np.abs(np.gradient(v)) + np.abs(np.gradient(np.gradient(v)))
    dp = np.abs(np.gradient(p))

    # Clip outliers before normalizing to avoid one extreme point dominating
    raw_scores = np.clip(du + dv + dp, 0, np.percentile(du + dv + dp, 95))
    scores = raw_scores / raw_scores.sum()

    generator = np.random.default_rng(seed=cfg.seed)
    n_scored  = int(n_max * (1 - cfg.uniform_frac))
    n_uniform = n_max - n_scored

    # Sample scored indices first
    idx_scored = generator.choice(M, size=n_scored, replace=False, p=scores)
    
    # Sample uniform indices from remaining (excluding scored)
    remaining_indices = np.setdiff1d(np.arange(M), idx_scored)
    idx_uniform = generator.choice(remaining_indices, size=n_uniform, replace=False)

    # Combine: scored first, then uniform (guaranteed exactly n_max points)
    all_idx = np.concatenate([idx_scored, idx_uniform])

    return fem_data[all_idx]


def build_labeled_dataset(fem_data_dict, n_max, cfg):
    """
    Concatenate labeled dataset across all geometries in the fem data.
    Blind to train/test split; fem_data_dict should only contain the desired set of geometries.
    Args:
        cfg: custom config object
        fem_data_dict: dictionary mapping geo_tag --> fem_data array of shape (N, 5) = [x,y,u,v,p]
    Returns:
        all_labeled_pts: array of shape (n_labeled_train * n_geometries, 7) = [x,y,a,b, u,v,p]
    """
    
    labeled_list = []
    
    for (a, b), fem_data in fem_data_dict.items():
        
        labeled_pts = sample_labeled_points(fem_data, n_max, cfg)
        
        # Insert (a, b) values after (x, y)
        ab_cols = np.full((labeled_pts.shape[0], 2), [a, b])
        labeled_data = np.concatenate([labeled_pts[:, :2], ab_cols, labeled_pts[:, 2:]], axis=1)
        
        labeled_list.append(labeled_data)
    
    all_labeled_pts = np.concatenate(labeled_list, axis=0)
    
    return all_labeled_pts