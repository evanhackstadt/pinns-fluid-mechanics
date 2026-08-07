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


# ———————————————— DATA HELPER FUNCTIONS ————————————————

def array_mask_ab(array, a, b, a_idx = 2, b_idx = 3, inverse: bool = False):
    mask = np.isclose(array[:, a_idx], a) & np.isclose(array[:, b_idx], b)
    if inverse:
        return array[~mask]
    else:
        return array[mask]


# ———————————————— DOMAIN DATASET ————————————————

# --- Sample Domain Points - Single Geometry ---
def _sample_domain_points_single(cfg, a, b):
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
def build_domain_dataset(cfg, geometries: list[tuple] = None):
    """
    Concatenate domain points across provided geometries 
    (or all training geometries by default).
    Returns:
        all_interior_data: array of shape (n_interior * n_train_geometries, 4) = [x,y,a,b]. 
        all_boundary_data: array of shape (n_boundary * n_train_geometries, 4) = [x,y,a,b]. 
    """
    selected_geos = geometries if geometries is not None else cfg.train_geometries
    
    interior_list = []
    boundary_list = []
    
    for (a, b) in selected_geos:
        interior_data, boundary_data = _sample_domain_points_single(cfg, a, b)
        interior_list.append(interior_data)
        boundary_list.append(boundary_data)
    
    all_interior_data = np.concatenate(interior_list, axis=0)
    all_boundary_data = np.concatenate(boundary_list, axis=0)
    
    return all_interior_data, all_boundary_data



# ———————————————— LABELED DATASET ————————————————


# --- Sample Labeled Data Points ---
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


def build_labeled_dataset(fem_data_dict, n, cfg, 
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
        all_labeled_pts: array of shape (n_labeled * n_geometries, 7) = [x,y,a,b, u,v,p]
    """
    
    labeled_list = []
    
    for (a, b), fem_data in fem_data_dict.items():
        
        labeled_pts = _sample_labeled_points_single(fem_data, n, cfg, components, coordinates)
        
        # Insert (a, b) values after (x, y)
        ab_cols = np.full((labeled_pts.shape[0], 2), [a, b])
        labeled_data = np.concatenate([labeled_pts[:, :2], ab_cols, labeled_pts[:, 2:]], axis=1)
        
        labeled_list.append(labeled_data)
    
    all_labeled_pts = np.concatenate(labeled_list, axis=0)
    
    return all_labeled_pts