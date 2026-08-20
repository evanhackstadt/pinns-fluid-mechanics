# data.py

"""
2D Stenosis Geometry-Conditioned PINN
    Functions to assemble labeled data dict for Supervised DeepONet

Evan Hackstadt
Rugonyi Lab
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
import torch
import deepxde as dde



# ———————————————— GEOMETRY DATASET ————————————————

def sample_ellipse_geometries(cfg):
    """
    Sample or load (a, b) pairs via Latin Hypercube Sampling within the triangle a >= b. 
    Oversamples and filters; repeats until n_total valid samples found.
    Returns (train_geos, test_geos) where each is a unique list of tuples (a, b)
    """
    # check if vals manually specified in config ==> override sampling
    override_train = hasattr(cfg, 'train_geometries') and cfg.train_geometries is not None
    override_test  = hasattr(cfg, 'test_geometries') and cfg.test_geometries is not None

    if override_train and override_test:
        print("Ellipse geometries manually specified in config file. Caching and returning.")
    
    else:
        # need to sample either train, test, or all geometries
        def _sample_geometries_lhs(cfg, target_n, exclude=[]):
            sampler = LatinHypercube(d=2, seed=cfg.seed)
            samples = []
            while len(samples) < target_n:
                raw = sampler.random(n=target_n * 3)   # oversample to account for rejection
                scaled = scale(raw, [cfg.a_range[0], cfg.b_range[0]], 
                                    [cfg.a_range[1], cfg.b_range[1]])
                valid = scaled[scaled[:, 0] >= scaled[:, 1]]   # enforce a >= b
                valid_tups = [(round(a, 3), round(b, 3)) for a, b in valid if (a, b) not in exclude]    # cast to tuple and enforce train/test exclude  
                samples.extend(valid_tups)
            
            pairs = [(round(float(a), 3), round(float(b), 3)) for (a, b) in samples[:target_n]]     # cast to floats, round again for safety
            return pairs

        if override_train and not override_test:
            print(f"Sampling {cfg.n_test_geometries} testing ellipse geometries; training geometries manually specified.")
            train_geos = cfg.train_geometries
            test_geos = _sample_geometries_lhs(cfg, cfg.n_test_geometries, exclude=train_geos)
        elif override_test and not override_train:
            print(f"Sampling {cfg.n_train_geometries} training ellipse geometries; testing geometries manually specified.")
            test_geos = cfg.test_geometries
            train_geos = _sample_geometries_lhs(cfg, cfg.n_train_geometries, exclude=test_geos)
        else:
            n_total = cfg.n_train_geometries + cfg.n_test_geometries
            print(f"Sampling {n_total} total ellipse geometries --> hold out {cfg.n_test_geometries} test geometries.")
            all_geos = _sample_geometries_lhs(cfg, n_total)
            # train-test split: hold out equally-spaced geos for testing
            test_idx = np.round(np.linspace(0, n_total - 1, cfg.n_test_geometries)).astype(int)
            train_idx = np.setdiff1d(np.arange(n_total), test_idx)    # remaining indices
            test_geos = [all_geos[i] for i in test_idx]
            train_geos = [all_geos[i] for i in train_idx]

    # Sort by area as a measure of severity
    area = lambda ab: ab[0] * ab[1]
    train_geos.sort(key=area)
    test_geos.sort(key=area)

    return train_geos, test_geos



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