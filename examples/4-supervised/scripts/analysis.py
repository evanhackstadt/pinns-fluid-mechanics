# analysis.py

"""
2D Parameterized Stenosis
    Analysis and visualization functions

Evan Hackstadt
Rugonyi Lab
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
import seaborn as sns

from geometry import ellipse_mask


CMAP_VAR   = "rainbow"
CMAP_ERR   = "hot_r"
COLOR_PINN = "#185FA5"   # blue  – PINN prediction
COLOR_TRUE = "#E8593C"   # coral – analytical ground truth
FIG_DPI    = 200


# ———————————— PER-RUN ANALYSIS FUNCTIONS ————————————

# --- Compute Errors ---
def compute_errors(pinn_data, fem_data):
    """
    Computes L2, L_inf (max absolute error), and MSE between prediction and ground-truth data.
    Args:
        pinn_data: array of shape (N, 5) with columns = [x, y, u_pinn, v_pinn, p_pinn]
        fem_data:  array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
    Returns:
        errors: dict containing L2, L_inf, and MSE for u, v, p, and total
    """
    
    errors = {}
    
    for i, variable in enumerate(['u', 'v', 'p']):
        # ensure coordinate alignment
        pinn_xy = pinn_data[:, 0:2]
        fem_xy  =  fem_data[:, 0:2]
        if not np.array_equal(pinn_xy, fem_xy):
            assert "Error: (x,y) coordinate mismatch between PINN and FEM data."
        
        # extract data
        pred = pinn_data[:, i+2:i+3]
        true = fem_data[:, i+2:i+3]
        # compute errors
        diff   = pred - true
        l2_rel = np.linalg.norm(diff) / np.linalg.norm(true)
        l_inf  = np.max(np.abs(diff))
        mse    = np.square(np.subtract(true, pred)).mean()
        # store
        errors[variable] = {
            "L2": l2_rel,
            "L_inf": l_inf,
            "MSE": mse
        }
    
    # store total error across variables
    for metric in ["L2", "L_inf", "MSE"]:
        total = errors["u"][metric] + errors["v"][metric] + errors["p"][metric]
        errors["total"] = {}
        errors["total"][metric] = total
    
    return errors


# --- Save Errors as File ---
def save_errors(errors, output_dir, a, b, n):
    """
    Write the calculated errors to a json file.
    Args:
        errors (dict): output of compute_errors() containing L2, L_inf, and MSE
        output_dir: path to the folder to save the file
        tag: string noting a, b, and Re parameters of the run
    """
    
    output_dir = Path(output_dir)
    error_path = output_dir / "errors.json"
    errors["parameters"] = {"a": a, "b": b, "n": n}
    
    with error_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)



# --- Loss Curves ---
def plot_loss_curves(loss_data, output_dir):
    """
    Plot PDE loss and BC loss, as well as individual loss terms, for both train and test, over training iterations.
    Args:
        loss_data: array of shape (iters, 2*n_loss_terms + 1) with columns [iteration, **loss_train_terms, **loss_test_terms]
        output_dir: path to the relevant plots folder to save plot
    """
    # loss terms in order:
        #   PDE_continuity, PDE_x_momentum, PDE_y_momentum, 
        #   BC_inlet_u, BC_inlet_v, BC_wall_u, BC_wall_v, BC_outlet_p,
        #   BC_data_observed_u, BC_data_observed_v, BC_data_observed_p
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_terms = int((loss_data.shape[1] - 1) / 2)
    print(f"Extracting {n_terms} loss terms")
    
    if loss_data.shape[1] % 2 != 1:
        raise ValueError("Problem parsing loss_data array. Expects an odd number of columns = steps + 2*n_loss_terms.")
    
    steps      = loss_data[:, 0]
    loss_train = loss_data[:, 1:n_terms+1]
    loss_test  = loss_data[:, n_terms+1:]
    
    labels = {"train": loss_train, "test": loss_test}
    
    for label, loss in labels.items():
        
        pde_cont = loss[:, 0]
        pde_x_m  = loss[:, 1]
        pde_y_m  = loss[:, 2]
        bc_i_u   = loss[:, 3]
        bc_i_v   = loss[:, 4]
        bc_w_u   = loss[:, 5]
        bc_w_v   = loss[:, 6]
        bc_o_p   = loss[:, 7]
        if n_terms == 11:
            bc_obs_u = loss[:, 8]
            bc_obs_v = loss[:, 9]
            bc_obs_p = loss[:, 10]

        fig, ax = plt.subplots(figsize=(10, 7), dpi=FIG_DPI)

        ax.semilogy(steps, pde_cont, color="#00008b", lw=1.5, label="PDE (continuity)")
        ax.semilogy(steps, pde_x_m,  color="#1e90ff", lw=1.5, label="PDE (x-momentum)")
        ax.semilogy(steps, pde_y_m,  color="#00ffff", lw=1.5, label="PDE (y-momentum)")
        ax.semilogy(steps, bc_i_u,   color="#ff0000", lw=1.5, label="BC (inlet $u$)")
        ax.semilogy(steps, bc_i_v,   color="#e1bb12", lw=1.5, label="BC (inlet $v$)")
        ax.semilogy(steps, bc_w_u,   color="#00ff00", lw=1.5, label="BC (wall $u$)")
        ax.semilogy(steps, bc_w_v,   color="#008000", lw=1.5, label="BC (wall $v$)")
        ax.semilogy(steps, bc_o_p,   color="#2f4f4f", lw=1.5, label="BC (outlet $p$)")
        if n_terms == 11:
            ax.semilogy(steps, bc_obs_u, color="#f4a460", lw=1.5, label="BC (observed $u$)")
            ax.semilogy(steps, bc_obs_v, color="#ff69b4", lw=1.5, label="BC (observed $v$)")
            ax.semilogy(steps, bc_obs_p, color="#ff00ff", lw=1.5, label="BC (observed $p$)")
    
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(f"Loss history - {label}")
        ax.legend(framealpha=0.9, prop={'size': 10})
        ax.grid(True, which="both", lw=0.35, alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    
        plt.tight_layout()
        fname = output_dir / f"loss_curves_terms_{label}.png"
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        
        # Also save summarized PDE and BC curves
        pde_total = pde_cont + pde_x_m + pde_y_m
        bc_total = bc_i_u + bc_i_v + bc_w_u + bc_w_v + bc_o_p
        if n_terms == 11:
            bc_total += bc_obs_u + bc_obs_v + bc_obs_p

        fig, ax = plt.subplots(figsize=(7, 4), dpi=FIG_DPI)
        
        ax.semilogy(steps, pde_total, color=COLOR_PINN, lw=1.5, label="PDE Loss")
        ax.semilogy(steps, bc_total,  color=COLOR_TRUE, lw=1.5, label="BC Loss")
    
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(f"Loss history - {label}")
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", lw=0.35, alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    
        plt.tight_layout()
        fname = output_dir / f"loss_curves_summed_{label}.png"
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        


# --- Plot Domain ---
def plot_domain(cfg, a, b, output_dir, labeled_pts=None):
    """
    Visualizes the spatial domain and sampled labeled points, if provided.
    Args:
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        output_dir: path to the relevant plots folder to save plot
        labeled_pts: array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
    """
    # create box
    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)
    plt.xlim(-cfg.L/2, cfg.L/2)
    plt.ylim(0, cfg.H_max)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    # add the ellipse patch
    ellipse = Ellipse(xy=(cfg.x_c, cfg.y_c), 
                      width=a*2, 
                      height=b*2, 
                      angle=cfg.angle,
                      color='black')
    ax.add_patch(ellipse)
    
    # add points used for supervised learning
    if labeled_pts is not None:
        plt.scatter(labeled_pts[:, 0], labeled_pts[:, 1], s=30, c='green')
        plt.title(f"Domain with n={labeled_pts.shape[0]} measurements (green)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save plot
    plt.savefig(output_dir / "geometry.png", dpi=FIG_DPI)
    plt.close()



# ———————————— HEATMAP FUNCTIONS ————————————

# --- Helper: Format Data ---
def _prepare_grid_data(x_query, y_query, values, cfg, a, b):
    # Reconstruct the original uniform grid axes
    xs = np.linspace(-cfg.L/2, cfg.L/2, cfg.nx)
    ys = np.linspace(0, cfg.H_max, cfg.ny)
    XX, YY = np.meshgrid(xs, ys)
    flat_x, flat_y = XX.ravel(), YY.ravel()

    # Compute the mask
    outside = ellipse_mask(flat_x, flat_y, cfg, a, b)
    Z_flat = np.full(len(flat_x), np.nan)
    # Fill in values at valid points (outside ellipse)
    Z_flat[outside] = values
    # Mesh
    ZZ = Z_flat.reshape(cfg.ny, cfg.nx)

    return XX, YY, ZZ


# --- Helper: Plot One Heatmap ---
def plot_heatmap_single(axis, X, Y, values, cmap, cfg, a, b,
                        cbar_math_format=False, cbar_cap=False,
                        cbar_label=None, title=None):
    
    vmax = np.percentile(values, 95) if cbar_cap else np.max(values)
    
    X_grid, Y_grid, Z_grid = _prepare_grid_data(X, Y, values, cfg, a, b)
    pcm = axis.pcolormesh(X_grid, Y_grid, Z_grid, cmap=cmap, 
                          shading="auto", vmax=vmax)
    cbar = plt.colorbar(pcm, ax=axis, label=cbar_label)
    
    # format
    plt.xlim(-cfg.L/2, cfg.L/2)
    axis.set_ylim(0, cfg.H_max)
    
    if cbar_math_format:
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

    # draw obstruction boundary as a dashed ellipse outline, if provided
    if a and b:
        ellipse = Ellipse(
            xy=(cfg.x_c, cfg.y_c),
            width=a * 2,
            height=b * 2,
            angle=np.degrees(cfg.angle),
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            linewidth=1.25,
            zorder=10,
        )
        axis.add_patch(ellipse)

    # decorate
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")
    axis.set_title(title)
    axis.set_aspect("equal")
    
    return axis


# --- Heatmaps of Model Outputs ---
def plot_output_heatmaps(pinn_data, fem_data, cfg, tag, output_dir,
                         a=None, b=None, separate_plots=False):
    """
    Create a multiplot figure showing PINN and FEM heatmaps of each output over the domain.
    Args:
        pinn_data: array of shape (N, 5) with columns = [x, y, u_pinn, v_pinn, p_pinn]
        fem_data:  array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
        query: array of (x,y) inputs parallel to outputs, shape (N, 2)
        cfg: custom config class object
        tag: string noting a, b, and Re of the run
        output_dir: path to the relevant plots folder to save file
        a (optional): ellipse a; providing a and b will plot the ellipse wall on the heatmap
        b (optional): ellipse b; providing a and b will plot the ellipse wall on the heatmap
        separate_plots (optional): if set to True, also save heatmaps for each model/variable as separate files
    """
    
    # ensure coordinate alignment
    pinn_xy = pinn_data[:, 0:2]
    fem_xy  =  fem_data[:, 0:2]
    if not np.array_equal(pinn_xy, fem_xy):
        assert "Error: (x,y) coordinate mismatch between PINN and FEM data."
    
    X, Y = pinn_data[:, 0], pinn_data[:, 1]
    models = {"PINN": pinn_data, "FEM": fem_data}
    variables = ["u", "v", "p"]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 6), dpi=FIG_DPI, constrained_layout=True)

    # plot each (model, variable) pair on its own axis
    for i, (model, data) in enumerate(models.items()):
        for j, var in enumerate(variables):
            
            # cbar_cap = True if model == "FEM" else False
            cbar_cap = False
            
            values = data[:, j+2]
            axes[i, j] = plot_heatmap_single(axes[i, j], X, Y, values, CMAP_VAR, cfg, a, b,
                                             cbar_cap=cbar_cap,
                                             cbar_label=f"${var}$",
                                             title=f"{model} ${var}(x, y)$")
            
            # if saving plots separately
            if separate_plots:
                fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
                plot_heatmap_single(ax_sep, X, Y, values, CMAP_VAR, cfg, a, b,
                                    cbar_label=f"${var}$",
                                    title=f"{model} ${var}(x, y)$")
                fname = output_dir / f"output_{model}_{var}_{tag}.png"
                fig_sep.savefig(fname, dpi=FIG_DPI)
                plt.close(fig_sep)
        
    # save multiplot
    fname = output_dir / f"outputs_{tag}.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)


# --- Heatmaps of Errors ---
def plot_error_heatmaps(pinn_data, fem_data, cfg, tag, output_dir,
                        a=None, b=None, separate_plots=False):
    """
    Create a multiplot figure showing an error heatmap for each output.
    Args:
        pinn_data: array of shape (N, 5) with columns = [x, y, u_pinn, v_pinn, p_pinn]
        fem_data:  array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
        query: array of (x,y) inputs parallel to outputs, shape (N, 2)
        cfg: custom config class object
        tag: string noting a, b, and Re of the run
        output_dir: path to the relevant plots folder to save file
        a (optional): ellipse a; providing a and b will plot the ellipse wall on the heatmap
        b (optional): ellipse b; providing a and b will plot the ellipse wall on the heatmap
        separate_plots (optional): if set to True, also save heatmaps for each variable as separate files
    """
    
    # ensure coordinate alignment
    pinn_xy = pinn_data[:, 0:2]
    fem_xy  =  fem_data[:, 0:2]
    if not np.array_equal(pinn_xy, fem_xy):
        assert "Error: (x,y) coordinate mismatch between PINN and FEM data."
    
    X, Y = pinn_data[:, 0], pinn_data[:, 1]
    variables = {
        "u": [0, 0],    # 2x2 multiplot indices
        "v": [1, 0],
        "p": [0, 1]
    }
    total_err = np.zeros(shape=(pinn_data.shape[0],), dtype=float)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=FIG_DPI, constrained_layout=True)

    # plot error for each variable and total
    for i, (var, idxs) in enumerate(variables.items()):
        a_i, a_j = idxs[0], idxs[1]
        
        # get vals and plot
        err = np.abs(pinn_data[:, i+2] - fem_data[:, i+2])
        total_err += err
        axes[a_i, a_j] = plot_heatmap_single(axes[a_i, a_j], X, Y, err, CMAP_ERR, cfg, a, b,
                                             cbar_math_format=True,
                                             cbar_label="|error|",
                                             title=f"Absolute error of ${var}(x,y)$")
        
        # save standalone plots separately if requested
        if separate_plots:
            fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
            plot_heatmap_single(ax_sep, X, Y, err, CMAP_ERR, cfg, a, b,
                                cbar_math_format=True,
                                cbar_label="|error|",
                                title=f"Absolute error of ${var}(x,y)$")
            fname = output_dir / f"error_{var}_{tag}.png"
            fig_sep.savefig(fname, dpi=FIG_DPI)
            plt.close(fig_sep)
    
    
    # plot total error
    axes[1, 1] = plot_heatmap_single(axes[1, 1], X, Y, total_err, CMAP_ERR, cfg, a, b,
                                     cbar_math_format=True,
                                     cbar_label="|error|",
                                     title="Total absolute error across variables")
    
    # if separate plots, need to save total standalone
    if separate_plots:
        fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
        plot_heatmap_single(ax_sep, X, Y, total_err, CMAP_ERR, cfg, a, b,
                            cbar_math_format=True,
                            cbar_label="|error|",
                            title="Total absolute error across variables")
        fname = output_dir / f"error_total_{tag}.png"
        fig_sep.savefig(fname, dpi=FIG_DPI)
        plt.close(fig_sep)
        
    # save multiplot
    fname = output_dir / f"errors_{tag}.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)





# ———————————— ACROSS-RUNS ANALYSIS ————————————

def compare_runs(summary_path, output_dir, parameter, 
                   fixed_ab: list = None, fixed_n = None):
    """
    Create point plots of error across different values of n, for variables u,v,p and total MSE.
    Args:
        summary_path: path to summary.json containing errors across runs
        output_dir: path to folder to save plots
        axis: string specifying the parameter of interest, choices = ["n", "a", "b", "ab"]
        fixed_ab: specified list of [a,b] to use across n; discards other geometries. If None, takes average errors across all (a,b). Requires variable="n".
        fixed_n:  specified value of n to use across (a,b); discards other n. If None, takes average errors across all n. Requires variable!="n".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CALL param={parameter}, fixed_ab={fixed_ab}, fixed_n={fixed_n}")
    
    # Parse args
    parameter_choices = ["n", "a", "b", "ab"]
    if parameter not in parameter_choices:
        raise ValueError(f"Received parameter = {parameter}, but must be one of {parameter_choices}")
    if fixed_ab and len(fixed_ab) != 2:
        raise ValueError(f"fixed_ab must be a list of length 2; received {fixed_ab}.")
    if parameter == "n":
        if fixed_n is not None:
            raise ValueError(f"fixed_n is not compatible with parameter={parameter}.")
    if parameter != "n":
        if fixed_ab is not None:
            raise ValueError(f"fixed_ab is not compatible with parameter={parameter}.")
    
    # Load data
    summary_path = Path(summary_path)
    with summary_path.open() as f:
        errors = json.load(f)    # [run_][u/v/p/total/parameters][attribute]
    
    METRICS = ["L2", "L_inf", "MSE"]
    VARS = ["u", "v", "p"]
    
    selected = {}
    
    # Select runs based on fixed_var, otherwise average across all cases of each parameter value
    for data in errors.values():
        # data = {"u"={...}, ..., "total"={}, "parameters"={}}
        a = data["parameters"]["a"]
        b = data["parameters"]["b"]
        n = data["parameters"]["n"]
        param_keys = {"n": n, "a": a, "b": b, "ab": f"({a}, {b})"}
        param_key = param_keys[parameter]
        print(f"param_key: {parameter}={param_key}")
        
        if fixed_ab is not None:
            if a == fixed_ab[0] and b == fixed_ab[1]:
                print(f"selected a,b={a},{b}")
                data.pop("parameters")
                selected[param_key] = data
                continue
            else:
                continue
        
        elif fixed_n is not None:
            if n == fixed_n:
                print(f"selected n={n}")
                data.pop("parameters")
                selected[param_key] = data
                continue
            else:
                continue
        
        else:
            for data in errors.values():
                # Initialize selected[param_key] if it doesn't exist
                if param_key not in selected:
                    selected[param_key] = {var: {met: [] for met in METRICS} for var in VARS}
                    selected[param_key]["total"] = {"MSE": []}
                
                # Collect errors across (a,b) in lists
                for var in VARS:
                    for met in METRICS:
                        selected[param_key][var][met].append(data[var][met])
                selected[param_key]["total"]["MSE"].append(data["total"]["MSE"])
            
            # Now average collected values
            for param_key in selected:
                for var in VARS:
                    for met in METRICS:
                        selected[param_key][var][met] = np.mean(selected[param_key][var][met])
                selected[param_key]["total"]["MSE"] = np.mean(selected[param_key]["total"]["MSE"])
    
    # Prepare plot data
    PARAMETER_LABELS = {
        "n": "number of labeled training points",
        "a": "ellipse width (a)",
        "b": "ellipse height (b)",
        "ab": "ellipse geometry (a, b)"
    }
    
    total_mses = pd.DataFrame(columns=[parameter, 'mse'])     # store for separate plot
    
    for metric in METRICS:
        
        plot_data = pd.DataFrame(columns=[parameter, 'variable', 'error'])
        
        for param_key, data in selected.items():
            # keys = parameter values; vals = {'u': {}, ...}            
            total_mse = data["total"]["MSE"]
            total_mses.loc[len(total_mses)] = [param_key, total_mse]
            for var in VARS:
                error = data[var][metric]
                plot_data.loc[len(plot_data)] = [param_key, var, error]
        
        # Plot metric
        ax = sns.pointplot(plot_data, x=parameter, y='error', hue='variable')
        plt.xlabel(PARAMETER_LABELS[parameter])
        plt.ylabel(metric)
        
        title = f"{metric} error across {parameter}"
        if fixed_n:
            title += f", (where n={fixed_n})"
        elif fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
        else:
            averaged_across = "ab" if parameter == "n" else "n"
            title += f", averaged across {averaged_across}"
            
        plt.title(title)
        plt.tight_layout()
        
        fname = output_dir / f"errors_by_{parameter}_{metric}.png"
        ax.figure.savefig(fname, dpi=FIG_DPI)
        plt.close(ax.figure)
    
    # Also plot total MSE
    ax = sns.pointplot(total_mses, x=parameter, y='mse')
    plt.xlabel(PARAMETER_LABELS[parameter])
    plt.ylabel("Total MSE")
    plt.title(f"Total MSE of all outputs, across {parameter}")
    plt.tight_layout()
    
    fname = output_dir / f"errors_by_{parameter}_MSE_total.png"
    ax.figure.savefig(fname, dpi=FIG_DPI)
    plt.close(ax.figure)