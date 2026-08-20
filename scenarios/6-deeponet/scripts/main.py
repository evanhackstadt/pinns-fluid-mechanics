# main.py

"""
2D Stenosis Physics-Informed Deep Operator Network
    Main script and CLI entry point; calls other scripts

Evan Hackstadt
Rugonyi Lab
"""


import sys
import argparse
from pathlib import Path
import glob
import json, pickle

import numpy as np
import torch
import deepxde as dde

from config import StenosisConfig
from geometry import create_stenosis_mesh, ellipse_bottom, ellipse_mask
from fem import read_mesh, solve_stenosis, fem_predict
from data import sample_ellipse_geometries, build_labeled_data_dict
from deeponet import *
from analysis import *


# ———————————————— HELPER FUNCTIONS ————————————————

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re",        type=float, help="Reynolds number")
    parser.add_argument("--n-adam",    type=int,   help="Adam iterations")
    parser.add_argument("--n-lbfgs",   type=int,   help="L-BFGS max iterations")
    parser.add_argument("--mesh-size", type=float, help="FEM mesh element size")
    
    parser.add_argument("--force-mesh", action="store_true",
                        help="Force fresh mesh generation instead of depending on cached file.")
    parser.add_argument("--skip-fem", action="store_true",
                        help="Skip the live FEM solve and instead load from cached file.")
    parser.add_argument("--force-resample", action="store_true",
                        help="Force resampling of all branch, trunk, and labeled data. Will require a fresh DeepONet training as well.")
    parser.add_argument("--force-deeponet", action="store_true",
                        help="Force fresh DeepONet training instead of depending on saved models.")
    
    return parser.parse_args()



# ———————————————— SUB-FUNCTIONS ————————————————
# perform lazy execution of functions from other scripts & save results


# --- 1. Sample Geometries ---
def load_or_cache_geometries(cfg: StenosisConfig, force_resample: bool):
    """
    Sample or load (a, b) pairs, handle train/test split, and writes & returns config object.
    Args:
        cfg: custom config object
        force_resample: whether or not to force resampmling of geometries.
    Returns:
        cfg: new config object holding the sampled train and test geometries
    """

    train_file = cfg.data_dir / f"geometries_train.pkl"
    test_file  = cfg.data_dir / f"geometries_test.pkl"
    
    if not force_resample and train_file.exists() and test_file.exists():
        print(f"Loading geometries from files {train_file.name}, and {test_file.name}")
        with open(train_file, "rb") as fp:
            train_geos = pickle.load(fp)
        with open(test_file, "rb") as fp:
            test_geos = pickle.load(fp)
    
    else:
        train_geos, test_geos = sample_ellipse_geometries(cfg)
        print(f"Saving geometries to files {train_file.name}, and {test_file.name}")
        # Update cache - pickle for load/store tuples, csv for visual inspection
        with open(train_file, "wb") as fp:
            pickle.dump(train_geos, fp)
            np.savetxt(cfg.data_dir / "geometries_train.csv", train_geos, delimiter=",", fmt='%.3g')
        with open(test_file, "wb") as fp:
            pickle.dump(test_geos, fp)
            np.savetxt(cfg.data_dir / "geometries_test.csv", test_geos, delimiter=",", fmt='%.3g')
    
    # Update config object
    cfg.train_geometries = train_geos
    cfg.test_geometries = test_geos
    print(f"Updated config with {len(train_geos)} training geometries and {len(test_geos)} testing geometries")
    
    return cfg
        
    

# --- 2. Generate Meshes ---
def load_or_cache_mesh(cfg: StenosisConfig, a: float, b: float, force_mesh: bool):
    """
    Generates mesh for a given geometry, unless file already exists.
    Args:
        cfg: custom config object
        a: ellipse semimajor
        b: ellipse semiminor
        force_mesh: whether or not to force re-generation regardless of existing files.
    Returns:
        msh_file: Path to the existing or generated mesh file for the geometry.
    """
    
    tag = cfg.geo_tag(a, b)
    msh_file = cfg.meshes_dir / f"stenosis_{tag}.msh"
    
    # Create mesh if file does not already exist, or forced
    if force_mesh or not msh_file.exists():
        print(f"Generating mesh to be saved to: {msh_file}")
        create_stenosis_mesh(cfg, a, b, msh_file)
    else:
        print(f"Skipping mesh since existing mesh file found: {msh_file.name}")
    
    return msh_file


# --- 3. FEM Solve ---
def load_or_cache_fem(cfg: StenosisConfig, a: float, b: float, 
                      msh_file: Path, skip_fem: bool):
    """
    Solves the system using Finite Element Method for a given geometry, unless solution file already exists.
    Args:
        cfg: custom config object
        a: ellipse semimajor
        b: ellipse semiminor
        msh_file: generated mesh for this geometry
        skip_fem: whether or not to skip the live fem solve.
    Returns:
        fem_data: ground-truth array of shape (nx*ny - ellipse pts, 5) = [x,y,u,v,p]
        msh: mesh loaded from file
        u_sol: if fem not skipped, return FEniCS u solution object
        p_sol: if fem not skipped, return FEniCS p solution object
    """
        
    tag = cfg.geo_tag(a, b)
    fem_file = cfg.fem_dir / f"solution_{tag}.npy"
    
    if skip_fem and fem_file.exists():
        print(f"Skipping FEM since existing FEM file found: {fem_file.name}")
        fem_data = np.load(fem_file, allow_pickle=True).astype(np.float32)
        
        # still process mesh
        msh, _ = read_mesh(msh_file)
        
        return fem_data, None, None, msh
    
    elif not skip_fem:
        
        print(f"Running FEM, solution to be saved to: {fem_file.name}")
        msh, u_sol, p_sol = solve_stenosis(cfg, msh_file)
        
        # Save on a dense evaluation grid for later comparison
        xs = np.linspace(-cfg.L/2, cfg.L/2, cfg.query_nx)
        ys = np.linspace(0, cfg.H_max, cfg.query_ny)
        XX, YY = np.meshgrid(xs, ys)
        flat = np.column_stack([XX.ravel(), YY.ravel()])
        
        # Mask out points inside the ellipse obstruction
        mask = ellipse_mask(flat[:, 0], flat[:, 1], cfg, a, b)
        query = flat[mask]
        
        # Get FEM vals for the uniform query
        fem_data = fem_predict(u_sol, p_sol, msh, query, cfg)    # fem_data also holds query pts
        np.save(fem_file, fem_data)
        print(f"FEM solution saved to {fem_file}")
        
        return fem_data, u_sol, p_sol, msh
        
    else:
        raise ValueError("FEM file not found or incompatible args.")
    


# --- 4. Branch Inputs ---
def load_or_cache_sensors(cfg: StenosisConfig, force: bool = False):
    """
    Load sensor grid from disk or compute and save it.
    The sensor grid must be identical across all training/inference runs.
    Saving it prevents accidental mismatch if SENSOR_NX/NY constants change.
    """
    sensor_path = cfg.data_dir / "deeponet_sensors.csv"
    
    if not force and sensor_path.exists():
        print(f"Loading sensor grid from {sensor_path.name}")
        return np.loadtxt(sensor_path, delimiter=",").astype(np.float32)

    else:
        print(f"Computing sensor grid, saving to {sensor_path.name}")
        sensors = make_sensor_grid(cfg)
        np.savetxt(sensor_path, sensors, delimiter=",")
        return sensors


# --- 5. Labeled Data ---
def load_or_cache_labeled_data(fem_data_dict: dict, sensors: np.ndarray, 
                               cfg: StenosisConfig, force: bool = False):
    """
    Load labeled data dict from disk or compute and save it.
    Combines sensor SDF values with fem_data_dict.
    Returns a dict mapping (a,b) --> {
        "query": (N_labeled, 2),
        "targets": (N_labeled, 3)
    }
    """
    labeled_path = cfg.data_dir / "labeled_data_dict.pkl"
    
    if not force and labeled_path.exists():
        print(f"Loading labeled data from {labeled_path.name}")
        with open(labeled_path, "rb") as fp:
            labeled_data_dict = pickle.load(fp)

    else:
        print(f"Building labeled data dict, saving to {labeled_path.name}")
        labeled_data_dict = build_labeled_data_dict(fem_data_dict, 
                                                    cfg.n_labeled_train, 
                                                    sensors, 
                                                    cfg)
    
    with open(labeled_path, "wb") as fp:
        pickle.dump(labeled_data_dict, fp)
    
    return labeled_data_dict
    

# --- 6. Train DeepONet ---
def load_or_train_deeponet(cfg: StenosisConfig,
                           sensors: np.ndarray,
                           function_space: StenosisGeometrySpace,
                           labeled_data_dict: dict,
                           force_deeponet: bool):
    """
    Performs main DeepONet training on all training geometries
    Args:
        cfg: custom config object
        pde_data: DeepXDE PDE data object handling trunk points (geometry, PDE loss, channel BCs)
        sensors: array of fixed sensor points (n_sensors, 2) = [x,y]
        function_space: the StenosisGeometrySpace object used for building the model
        force_deeponet: whether or not to force retraining from scratch regardless of saved models
    Returns:
        trained_model: DeepXDE model object with trained weights / parameters
    """
    model_prefix = cfg.deeponet_dir / "model"
    
    assert list(labeled_data_dict.keys()) == cfg.train_geometries, \
        "fem_data_dict key order must match cfg.train_geometries"
    
    model = build_deeponet_model(cfg, sensors, function_space, labeled_data_dict)
    
    # proxy for train completion: saved model and training log
    existing_models = glob.glob(f"{model_prefix}*.pt")
    log_file = cfg.deeponet_dir / "training_log.json"
    rerun = False if len(existing_models) > 0 and log_file.exists() else True
    
    # train / restore
    if force_deeponet or rerun:
        print(f"Training DeepONet, solution to be saved to: {model_prefix.name}")
        cfg.clear_dir(cfg.deeponet_dir)     # clear any old models
        trained_model = train_deeponet(model, model_prefix, cfg)
    else:
        print(f"Skipping PINN since existing model(s) found: {existing_models}")
        trained_model = restore_deeponet(model, model_prefix, cfg)
    
    return trained_model



# --- 7-8. DeepONet Predict ---
def predict_and_save_errors(fem_data_dict, model, sensors, function_space, label, cfg: StenosisConfig):
    """
    Evaluate the model on each provided geometry, compare to ground truth, and save errors.
    Args:
        fem_data_dict: dictionary mapping (a, b) --> array of shape (N, 5) = [x,y,u,v,p] 
        model: a DeepXDE model object to use for prediction (can be baseline, finetuned, etc.)
        sensors: array of fixed sensor points (n_sensors, 2) = [x, y]
        function_space: the StenosisGeometrySpace object used for building the model
        label: either "train" or "test" to distinguish error file
        cfg: custom config object
    Returns:
        deeponet_data_dict: dictionary mapping geometries --> deeponet_data array of shape (N, 5) = [x,y,u,v,p], 
        summary_path: Path to summary_{tag}.json
    """
    all_errors = {}
    deeponet_data_dict = {}
    
    for (a, b), fem_data in fem_data_dict.items():
        print(f"Predicting on geometry ({a}, {b})")
        output_dir = cfg.geo_dir(a, b)
        tag = cfg.geo_tag(a, b)
        
        # predict
        query_pts = fem_data[:, 0:2]
        deeponet_data = deeponet_predict(model, sensors, cfg, a, b, query_pts, function_space)
        errors = compute_errors(deeponet_data, fem_data)
        
        # log
        save_errors(errors, output_dir, a, b)
        deeponet_data_dict[(a, b)] = deeponet_data
        all_errors[tag] = errors
    
    # Save errors
    summary_path = cfg.summary_dir / f"error_summary_{label}.json"
    with summary_path.open("w") as f:
        json.dump(all_errors, f, indent=2)
        
    return deeponet_data_dict, summary_path
    


# --- 9. Visualization ---
def visualization(deeponet_data_dict_train, deeponet_data_dict_test, 
                  fem_data_dict_train, fem_data_dict_test, 
                  train_error_summary_path, test_error_summary_path,
                  cfg: StenosisConfig):
    """
    Perform a variety of analyses and visualizations to evaluate the trained PINN
    Args:
        deeponet_data_dict_train: dictionary mapping training geo_tags -->  deeponet_data array of shape (N, 5) = [x,y,u,v,p] 
        deeponet_data_dict_test:  dictionary mapping testing geo_tags  -->  deeponet_data array of shape (N, 5) = [x,y,u,v,p] 
        fem_data_dict_train:  dictionary mapping training geo_tags -->  fem_data array of shape (N, 5) = [x,y,u,v,p] 
        fem_data_dict_test:   dictionary mapping testing geo_tags  -->  fem_data array of shape (N, 5) = [x,y,u,v,p] 
        train_error_summary_path: Path to the summary_train.json error log
        test_error_summary_path:  Path to the summary_test.json error log
        cfg: custom config object
    """
    
    '''
    # TRAIN geometries - per-geometry analyses
    for (a, b) in deeponet_data_dict_train.keys():
        print(f"Visualizing training geometry ({a}, {b})")
        
        output_dir = cfg.geo_dir(a, b)
        tag = cfg.geo_tag(a, b)
        deeponet_data = deeponet_data_dict_train[(a, b)]
        fem_data = fem_data_dict_train[(a, b)]
        
        plot_domain(cfg, a, b, output_dir)
        plot_output_heatmaps(deeponet_data, fem_data, cfg, tag, output_dir, a, b, separate_plots=False)
        plot_error_heatmaps(deeponet_data, fem_data, cfg, tag, output_dir, a, b, separate_plots=False)
        plot_velocity_quiver(deeponet_data, fem_data, cfg, tag, output_dir, a, b)
    
    
    # TEST geometries - per-geometry, per-strategy analyses
    for (a, b) in deeponet_data_dict_test.keys():
        print(f"Visualizing testing geometry ({a}, {b})")
        
        output_dir = cfg.geo_dir(a, b)
        tag = cfg.geo_tag(a, b)
        
        plot_domain(cfg, a, b, output_dir)
        
        output_dir = cfg.geo_dir(a, b)
        deeponet_data = deeponet_data_dict_test[(a, b)]
        fem_data = fem_data_dict_test[(a, b)]
        
        plot_output_heatmaps(deeponet_data, fem_data, cfg, tag, 
                             output_dir, a, b, separate_plots=False)
        plot_error_heatmaps(deeponet_data, fem_data, cfg, tag, 
                            output_dir, a, b, separate_plots=False)
        plot_velocity_quiver(deeponet_data, fem_data, cfg, tag,
                             output_dir, a, b)
    '''
    print(f"Per-Geometry analysis and visualization complete.")
    
    
    # GLOBAL analyses
    
    # visualize all geometries overlayed
    plot_all_domains(cfg, cfg.results_dir)
    
    # plot main training loss curves
    loss_data = np.loadtxt(cfg.deeponet_dir / "loss.dat", delimiter=" ", comments="#")
    plot_loss_curves(loss_data, cfg.deeponet_dir,
                     loss_term_labels=["PDE (continuity)", "PDE (x-momentum)", "PDE (y-momentum)",
                                       "BC (obstacle no-slip)",
                                       "BC (inlet u)", "BC (inlet v)", 
                                       "BC (wall u)", "BC (wall v)",
                                       "BC (outlet p)",
                                       "Supervised u", "Supervised v", "Supervised p"])
    
    # plot mega-grid of all testing outputs
    output_dir = cfg.results_dir / "test" / "ALL"
    plot_output_heatmaps_multi(deeponet_data_dict_test, fem_data_dict_test, cfg, "test", output_dir)
    plot_error_heatmaps_multi(deeponet_data_dict_test, fem_data_dict_test, cfg, "test", output_dir)
    plot_velocity_quiver_multi(deeponet_data_dict_test, fem_data_dict_test, cfg, "test", output_dir)

    # compare train & test errors separately
    output_dir = cfg.summary_dir / "train"
    plot_error_comparison(train_error_summary_path, output_dir, cfg, 
                          parameter="ab_area", fixed_strat=None, 
                          strategies=None, lineplot=True)
    plot_error_comparison(train_error_summary_path, output_dir, cfg, 
                          parameter="a", fixed_strat=None, 
                          strategies=None, lineplot=True)
    plot_error_comparison(train_error_summary_path, output_dir, cfg, 
                          parameter="b", fixed_strat=None, 
                          strategies=None, lineplot=True)
    
    output_dir = cfg.summary_dir / "test"
    plot_error_comparison(test_error_summary_path, output_dir, cfg, 
                          parameter="ab", fixed_strat=None, strategies=None)
    
    # compare train + test errors side-by-side
    with open(train_error_summary_path, "r", encoding="utf-8") as f:
        train_errors = json.load(f)
    with open(test_error_summary_path, "r", encoding="utf-8") as f:
        test_errors = json.load(f)
    all_errors = train_errors | test_errors
    
    all_error_summary_path = train_error_summary_path.parent / "error_summary_all.json"
    with open(all_error_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2)
    
    output_dir = cfg.summary_dir / "all"
    plot_error_comparison(all_error_summary_path, output_dir, cfg, parameter="ab_area",
                          fixed_strat=None, strategies=None)

    # log config
    config_dict = cfg.config_as_dict()
    config_path = cfg.results_dir / "config_log.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Global analysis and visualization complete.")
    
    

    

# ———————————————— MAIN ————————————————
# top-level overview of pipeline & handle args

def main():
    
    # Stage 0: Args and Print
    args = parse_args()
    cfg  = StenosisConfig()

    # Override config with CLI args
    if args.re:
        cfg.Re = args.re
    if args.n_adam:
        cfg.n_adam = args.n_adam
    if args.n_lbfgs:
        cfg.n_lbfgs = args.n_lbfgs
    if args.mesh_size:
        cfg.mesh_size = args.mesh_size
    if args.force_resample:
        args.force_deeponet = True  # dependency
    
    
    # Stage 1: Generate ellipse geometries
    cfg = load_or_cache_geometries(cfg, args.force_resample)
    cfg.make_all_dirs()
    
    # Validate geometry split
    for tup in cfg.train_geometries:
        if tup in cfg.test_geometries:
            raise ValueError(f"Duplicate geometries found in train and test geometries.\nTrain={cfg.train_geometries}\nTest={cfg.test_geometries}")

    # Print to terminal
    print(f"\n{'='*50}\nEXECUTION PLAN\n{'='*50}")
    print(f"\n1. Generate meshes for each geometry (forced = {args.force_mesh})")
    print(f"2. Solve FEM ground truth for each geometry (forced = {args.skip_fem})")
    print(f"3. Assemble branch data (SDF at sensor points, train and test geometries) (forced = {args.force_resample})")
    print(f"4. Assemble trunk data (extended trunk data, train geometries) (forced = {args.force_resample})")
    print(f"5. Build and Train DeepONet on {len(cfg.train_geometries)} training geometries (forced = {args.force_deeponet})")
    print(f"6. Validate DeepONet on training geometries")
    print(f"8. Test DeepONet on {len(cfg.test_geometries)} testing geometries")
    print(f"9. Perform analysis and visualization")
    print(f"\nTrain geometries={cfg.train_geometries}")
    print(f"\nTest geometries{cfg.test_geometries}")
    
    # Stages 2-3: Generate meshes and ground-truth FEM for train + test geometries
    fem_data_dict_train = {}
    fem_data_dict_test  = {}
    fem_sol_objects_train = {'u': {}, 'p': {}, 'msh': {}}
    fem_sol_objects_test  = {'u': {}, 'p': {}, 'msh': {}}
    
    for (a, b) in cfg.train_geometries:
        msh_file = load_or_cache_mesh(cfg, a, b, args.force_mesh)
        fem_data, msh, u_sol, p_sol = load_or_cache_fem(cfg, a, b, msh_file, args.skip_fem)
        
        fem_data_dict_train[(a, b)] = fem_data
        fem_sol_objects_train['u'][(a, b)] = u_sol
        fem_sol_objects_train['p'][(a, b)] = p_sol
        fem_sol_objects_train['msh'][(a, b)] = msh
    
    for (a, b) in cfg.test_geometries:
        msh_file = load_or_cache_mesh(cfg, a, b, args.force_mesh)
        fem_data, msh, u_sol, p_sol = load_or_cache_fem(cfg, a, b, msh_file, args.skip_fem)
        
        fem_data_dict_test[(a, b)] = fem_data
        fem_sol_objects_test['u'][(a, b)] = u_sol
        fem_sol_objects_test['p'][(a, b)] = p_sol
        fem_sol_objects_test['msh'][(a, b)] = msh

    
    # Stage 4: Branch Inputs (establish geometry spaces and sensors)
    function_space = StenosisGeometrySpace(cfg, cfg.train_geometries)
    function_space_test = StenosisGeometrySpace(cfg, cfg.test_geometries)
    sensors = load_or_cache_sensors(cfg)
    
    
    # Stage 5: Labeled Data
    labeled_data_dict = load_or_cache_labeled_data(fem_data_dict_train, 
                                                   sensors, 
                                                   cfg, 
                                                   args.force_resample)
    
    
    # Stage 6: Build and Train DeepONet
    trained_model = load_or_train_deeponet(cfg, sensors, function_space, 
                                           labeled_data_dict, args.force_deeponet)
    
    
    # Stage 7: Evaluate on Training Geometries
    deeponet_data_dict_train, summary_train = predict_and_save_errors(fem_data_dict_train, 
                                                                      trained_model,
                                                                      sensors, 
                                                                      function_space,
                                                                      "train", 
                                                                      cfg)
    # Stage 8: Evaluate on Testing Geometries
    deeponet_data_dict_test, summary_test = predict_and_save_errors(fem_data_dict_test, 
                                                                    trained_model, 
                                                                    sensors, 
                                                                    function_space_test,
                                                                    "test", 
                                                                    cfg)
    
    
    # Stage 9: Analysis and Visualization
    visualization(deeponet_data_dict_train, deeponet_data_dict_test,
                    fem_data_dict_train, fem_data_dict_test,
                    summary_train, summary_test, cfg)
    
    print(f"\n{'='*50}\nPIPELINE COMPLETE\n{'='*50}")
    


if __name__ == "__main__":
    main()