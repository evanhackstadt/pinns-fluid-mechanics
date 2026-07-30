# analysis.py

"""
2D Stenosis Geometry-Conditioned PINN
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
from data import array_mask_ab


PALETTE_DEEP = sns.color_palette("deep").as_hex()
CMAP_VAR   = "rainbow"
CMAP_ERR   = "flare"
COLOR_PINN = PALETTE_DEEP[0]
COLOR_U    = PALETTE_DEEP[1]
COLOR_TRUE = PALETTE_DEEP[2]
COLOR_V    = PALETTE_DEEP[3]
COLOR_P    = PALETTE_DEEP[4]
COLOR_AGGREGATE = PALETTE_DEEP[5]
COLOR_PDE   = PALETTE_DEEP[6]
COLOR_BC    = PALETTE_DEEP[7]
COLOR_VARIABLE_MAP = {"u": COLOR_U, "v": COLOR_V, "p": COLOR_P}
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
    
    VARS = ['u', 'v', 'p']
    errors = {}
    
    for i, variable in enumerate(VARS):
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
        # store
        errors[variable] = {
            "L2": l2_rel,
            "L_inf": l_inf,
        }
    
    # store aggregated error across variables (mean L2, max L_inf)
    mean_L2 = np.mean([errors[var]["L2"] for var in VARS])
    max_L_inf = np.max([errors[var]["L_inf"] for var in VARS])
    errors["aggregate"] = {"mean_L2": mean_L2, 
                           "max_L_inf": max_L_inf}
    
    return errors


# --- Save Errors as File ---
def save_errors(errors, output_dir, a, b):
    """
    Write the calculated errors to a json file.
    Args:
        errors (dict): output of compute_errors() containing L2, L_inf, and MSE
        output_dir: path to the folder to save the file
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        n: number of labeled points used for supervised training
    """
    
    output_dir = Path(output_dir)
    error_path = output_dir / "errors.json"
    errors["parameters"] = {"a": a, "b": b}
    
    with error_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)



# --- Loss Curves ---
def plot_loss_curves(loss_data, output_dir, 
                     loss_term_labels = ["PDE (continuity)", "PDE (x-momentum)", "PDE (y-momentum)",
                                         "BC (inlet u)", "BC (inlet v)", "BC (wall u)", "BC (wall v)",
                                         "BC (obstacle u)", "BC (obstacle v)", "BC (outlet p)",
                                         "BC (observed u)", "BC (observed v)", "BC (observed p)"]):
    """
    Plot PDE loss and BC loss, as well as individual loss terms, for training loss over iterations.
    Args:
        loss_data: array of shape (iters, 2*n_loss_terms + 1) with columns [iteration, **loss_train_terms, **loss_test_terms]
        output_dir: Path to the directory in which to save the plot
        loss_term_labels: optional list of labels corresponding to loss terms. Recommended for cases besides general training.
                          If not provided, hardcoded terms = PDE cont, PDE xm/ym, BC in u/v, BC wall u/v, BC obst u/v, BC out p, BC obs u/v/p.
    """
    # all loss terms in order:
        #   PDE_continuity, PDE_x_momentum, PDE_y_momentum, 
        #   BC_inlet_u, BC_inlet_v, 
        #   BC_wall_u, BC_wall_v, 
        #   BC_obstacle_u, BC_obstacle_v, 
        #   BC_outlet_p,
        #   BC_data_observed_u, BC_data_observed_v, BC_data_observed_p

    output_dir = Path(output_dir)
    
    # Parse loss data
    n_terms = int((loss_data.shape[1] - 1) / 2)
    if loss_data.shape[1] % 2 != 1:
        raise ValueError("Problem parsing loss_data array. Expected an odd number of columns = [steps] + [2*n_loss_terms].")
    if n_terms != len(loss_term_labels):
        raise ValueError(f"Found {n_terms} in loss data but received {len(loss_term_labels)} labels.")
    
    steps = loss_data[:, 0]
    loss  = loss_data[:, 1:n_terms+1]    # train loss only
    
    loss_terms = {}
    for i, label in enumerate(loss_term_labels):
        loss_terms[label] = loss[:, i]

    # Plot
    
    # use visually-distinct colors: https://mokole.com/palette.html
    LOSS_COLORS = ["#2f4f4f", "#8b4513", "#228b22",
                   "#4b0082", "#ff0000", "#ffff00",
                   "#00ff00", "#00ffff", "#0000ff",
                   "#ff00ff", "#1e90ff", "#eee8aa", 
                   "#ff69b4"]
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=FIG_DPI)
    pde_terms = []
    bc_terms = []

    for i, (label, data) in enumerate(loss_terms.items()):
        ax.semilogy(steps, data, color=LOSS_COLORS[i], lw=1.5, label=label)
        if "pde" in label.lower():
            pde_terms.append(data)
        elif "bc" in label.lower():
            bc_terms.append(data)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Training Loss History")
    ax.legend(framealpha=0.9, prop={'size': 10})
    ax.grid(True, which="both", lw=0.35, alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))

    plt.tight_layout()
    fname = output_dir / f"loss_curves_terms.png"
    plt.savefig(fname, dpi=FIG_DPI)
    plt.close()
    
    # Also save summarized PDE and BC curves
    fig, ax = plt.subplots(figsize=(7, 4), dpi=FIG_DPI)
    
    pde_total = np.sum(pde_terms, axis=0)
    bc_total  = np.sum(bc_terms,  axis=0)
    ax.semilogy(steps, pde_total, color=COLOR_PINN, lw=1.5, label="PDE Loss")
    ax.semilogy(steps, bc_total,  color=COLOR_TRUE, lw=1.5, label="BC Loss")

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Training Loss History")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", lw=0.35, alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))

    plt.tight_layout()
    fname = output_dir / f"loss_curves_summed.png"
    plt.savefig(fname, dpi=FIG_DPI)
    plt.close()
        


# --- Plot Domain ---
def plot_domain(cfg, a, b, output_dir, labeled_pts = None):
    """
    Visualizes the spatial domain and sampled labeled points, if provided.
    Args:
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        output_dir: path to the relevant plots folder to save plot
        labeled_pts: array of shape (N, ≥4) with columns = [x,y,a,b,...]
    """
    # create box
    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    plt.xlim(-cfg.L/2, cfg.L/2)
    plt.ylim(0, cfg.H_max)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    # add the ellipse patch
    ellipse = Ellipse(xy=(cfg.x_c, cfg.y_c), 
                      width=a*2, 
                      height=b*2, 
                      color='black')
    ax.add_patch(ellipse)
    
    # add points used for supervised learning
    if labeled_pts is not None:
        pts_this_ab  = array_mask_ab(labeled_pts, a, b, 2, 3)
        pts_other_ab = array_mask_ab(labeled_pts, a, b, 2, 3, inverse=True)
        plt.scatter(pts_this_ab[:, 0], pts_this_ab[:, 1], 
                    s=25, c=COLOR_PINN, label='This geometry')
        plt.scatter(pts_other_ab[:, 0], pts_other_ab[:, 1], 
                    s=25, c='grey', alpha=0.3, label='Other geometries')
        plt.legend()
        plt.title(f"Domain with n={pts_this_ab.shape[0]} of {labeled_pts.shape[0]} total measurements.")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.geo_tag(a, b)

    # save plot
    plt.savefig(output_dir / f"domain_{tag}.png", dpi=FIG_DPI)
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
def _plot_heatmap_single(axis, X, Y, values, cmap, cfg, a, b,
                         vmin=None, vmax=None, cbar_math_format=False, 
                         cbar_label=None, title=None):
    
    X_grid, Y_grid, Z_grid = _prepare_grid_data(X, Y, values, cfg, a, b)
    pcm = axis.pcolormesh(X_grid, Y_grid, Z_grid, cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          shading="auto")
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


def _validate_heatmap_alignment(fem_data, pinn_data_list):
    if not isinstance(pinn_data_list, (list, tuple)):
        pinn_data_list = [pinn_data_list]
    fem_xy = fem_data[:, 0:2]
    for idx, pinn_data in enumerate(pinn_data_list):
        if not np.array_equal(pinn_data[:, 0:2], fem_xy):
            raise ValueError(
                f"Heatmap alignment error: PINN dataset at index {idx} has mismatched (x,y) coordinates compared to FEM data."
            )
    return pinn_data_list


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
        tag: string to label the file with (e.g. a, b, other params)
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
    models = {"FEM": fem_data, "PINN": pinn_data}
    variables = ["u", "v", "p"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 6), dpi=FIG_DPI, constrained_layout=True)

    # plot each (model, variable) pair on its own axis
    for j, var in enumerate(variables):
        
        # min/max across models (PINN/FEM), for the given variable
        model_mins = []
        model_maxes = []
        for i, (model, data) in enumerate(models.items()):
            values = data[:, j+2]
            model_mins.append(np.min(values))
            model_maxes.append(np.max(values))
        
        vmin = np.min(model_mins)
        vmax = np.max(model_maxes)
        
        for i, (model, data) in enumerate(models.items()):
            
            values = data[:, j+2]
            axes[i, j] = _plot_heatmap_single(axes[i, j], X, Y, values, CMAP_VAR, cfg, a, b,
                                              vmin=vmin, vmax=vmax,
                                              cbar_label=f"${var}$",
                                              title=f"{model} ${var}(x, y)$")
            
            # if saving plots separately
            if separate_plots:
                fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
                _plot_heatmap_single(ax_sep, X, Y, values, CMAP_VAR, cfg, a, b,
                                     vmin=vmin, vmax=vmax,
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
    Create a multiplot figure showing an error heatmap for each variable, as well as mean L2 across variables.
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
    mean_err = np.zeros(shape=(pinn_data.shape[0],), dtype=float)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=FIG_DPI, constrained_layout=True)

    # plot error for each variable and total
    for i, (var, idxs) in enumerate(variables.items()):
        a_i, a_j = idxs[0], idxs[1]
        
        # get vals and plot
        err = np.abs(pinn_data[:, i+2] - fem_data[:, i+2])
        mean_err += err
        axes[a_i, a_j] = _plot_heatmap_single(axes[a_i, a_j], X, Y, err, CMAP_ERR, cfg, a, b,
                                              cbar_math_format=True,
                                              cbar_label="|error|",
                                              title=f"Absolute error of ${var}(x,y)$")
        
        # save standalone plots separately if requested
        if separate_plots:
            fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
            _plot_heatmap_single(ax_sep, X, Y, err, CMAP_ERR, cfg, a, b,
                                 cbar_math_format=True,
                                 cbar_label="|error|",
                                 title=f"Absolute error of ${var}(x,y)$")
            fname = output_dir / f"error_{var}_{tag}.png"
            fig_sep.savefig(fname, dpi=FIG_DPI)
            plt.close(fig_sep)
    
    
    # plot mean error
    mean_err = mean_err / len(variables.keys())
    axes[1, 1] = _plot_heatmap_single(axes[1, 1], X, Y, mean_err, CMAP_ERR, cfg, a, b,
                                      cbar_math_format=True,
                                      cbar_label="|error|",
                                      title="Mean absolute error across variables")
    
    # if separate plots, need to save total standalone
    if separate_plots:
        fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
        _plot_heatmap_single(ax_sep, X, Y, mean_err, CMAP_ERR, cfg, a, b,
                             cbar_math_format=True,
                             cbar_label="|error|",
                             title="Mean absolute error across variables")
        fname = output_dir / f"error_mean_{tag}.png"
        fig_sep.savefig(fname, dpi=FIG_DPI)
        plt.close(fig_sep)
        
    # save multiplot
    fname = output_dir / f"errors_{tag}.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)


# --- Velocity Vector Field ---
# Generated by Claude Sonnet 4.6
def plot_velocity_quiver(pinn_data, fem_data, cfg, tag, output_dir,
                         a=None, b=None, nx_q=40, ny_q=20):
    """
    Quiver plot of the velocity vector field (u, v) for both PINN and FEM.
    Arrows are colored by magnitude ||v|| = sqrt(u² + v²).

    Args:
        pinn_data:  array of shape (N, 5) = [x, y, u_pinn, v_pinn, p_pinn]
        fem_data:   array of shape (N, 5) = [x, y, u_fem,  v_fem,  p_fem]
        cfg:        StenosisConfig object
        tag:        string label for filename
        output_dir: Path to save plots
        a, b:       ellipse semi-axes; if provided, draws ellipse outline
        nx_q, ny_q: quiver grid resolution (coarser than heatmap is intentional —
                    dense arrows are unreadable)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a coarse uniform grid for the quiver arrows
    xs_q = np.linspace(-cfg.L / 2, cfg.L / 2, nx_q)
    ys_q = np.linspace(0, cfg.H_max, ny_q)
    XX_q, YY_q = np.meshgrid(xs_q, ys_q)
    flat_q = np.column_stack([XX_q.ravel(), YY_q.ravel()])

    # Mask points inside the ellipse obstruction
    if a is not None and b is not None:
        outside_q = ellipse_mask(flat_q[:, 0], flat_q[:, 1], cfg, a, b)
        flat_q = flat_q[outside_q]
        X_q = flat_q[:, 0]
        Y_q = flat_q[:, 1]
    else:
        X_q = XX_q.ravel()
        Y_q = YY_q.ravel()

    # Interpolate model outputs onto the coarse quiver grid.
    # Source data lives on the dense heatmap grid — use nearest-point lookup
    # via a KD-tree to avoid re-running the model.
    from scipy.spatial import cKDTree

    src_xy = pinn_data[:, 0:2]
    tree = cKDTree(src_xy)
    _, idx = tree.query(np.column_stack([X_q, Y_q]))

    def _extract(data, indices):
        U = data[indices, 2]   # u-velocity col
        V = data[indices, 3]   # v-velocity col
        M = np.sqrt(U**2 + V**2)
        return U, V, M

    models = {
        "PINN": pinn_data,
        "FEM":  fem_data,
    }

    # Shared colormap range across both panels (use FEM as reference)
    _, _, M_fem_full = _extract(fem_data, idx)
    vmax = np.percentile(M_fem_full, 98)   # clip extreme values near walls

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=FIG_DPI, constrained_layout=True)

    for ax, (label, data) in zip(axes, models.items()):
        U_q, V_q, M_q = _extract(data, idx)

        # Normalize arrow length so direction is legible at all speeds.
        # Scale factor ~0.8 avoids overlap at the chosen grid density.
        norm_mag = np.where(M_q > 0, M_q, 1e-12)
        U_n = U_q / norm_mag
        V_n = V_q / norm_mag

        qv = ax.quiver(
            X_q, Y_q,
            U_n, V_n,
            M_q,                        # color by magnitude
            cmap=CMAP_VAR,
            clim=(0, vmax),
            scale=nx_q * 1.5,           # tune: larger = shorter arrows
            scale_units="width",
            width=0.003,
            headwidth=4,
            headlength=5,
        )
        plt.colorbar(qv, ax=ax, label=r"$\|\mathbf{v}\|$")

        # Ellipse outline
        if a is not None and b is not None:
            ellipse_patch = Ellipse(
                xy=(cfg.x_c, cfg.y_c),
                width=a * 2, height=b * 2,
                edgecolor="black", facecolor="black",
                linestyle="-", linewidth=1.0, zorder=5,
            )
            ax.add_patch(ellipse_patch)

        ax.set_xlim(-cfg.L / 2, cfg.L / 2)
        ax.set_ylim(0, cfg.H_max)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(f"{label} velocity field $\\mathbf{{v}}(x,y)$")

    fname = output_dir / f"quiver_{tag}.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)



# ——————————— SEMI-GLOBAL ANALYSIS ———————————

def plot_output_heatmaps_multi(pinn_data_list, fem_data, cfg, tag, output_dir,
                               a=None, b=None, strategy_labels=None,
                               fem_row_label="FEM", separate_plots=False):
    """
    Create a stacked comparison heatmap figure with one FEM row and one row per PINN strategy.
    Args:
        pinn_data_list: list of arrays, each shape (N, 5) = [x, y, u, v, p]
        fem_data: array of shape (N, 5) = [x, y, u, v, p]
        cfg: custom config class object
        tag: string to label the file with
        output_dir: path to the relevant plots folder
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        strategy_labels: optional list of labels corresponding to each PINN dataset
        fem_row_label: label for the FEM row
        separate_plots: if True, saves each row to a separate file as well
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pinn_data_list = _validate_heatmap_alignment(fem_data, pinn_data_list)
    n_models = len(pinn_data_list)
    if strategy_labels is None:
        strategy_labels = [f"PINN {i+1}" for i in range(n_models)]
    if len(strategy_labels) != n_models:
        raise ValueError("strategy_labels must match the number of pinn_data arrays.")

    X, Y = fem_data[:, 0], fem_data[:, 1]
    variables = ["u", "v", "p"]
    model_names = [fem_row_label] + strategy_labels

    # determine consistent color limits for each variable across all rows
    value_ranges = {}
    for j, var in enumerate(variables):
        values = [fem_data[:, j + 2]] + [pinn[:, j + 2] for pinn in pinn_data_list]
        value_ranges[var] = (np.min([np.min(v) for v in values]),
                             np.max([np.max(v) for v in values]))

    n_rows = 1 + n_models
    fig, axes = plt.subplots(n_rows, len(variables), figsize=(6 * len(variables), 3 * n_rows),
                             dpi=FIG_DPI, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row in range(n_rows):
        row_label = model_names[row]
        data = fem_data if row == 0 else pinn_data_list[row - 1]
        for col, var in enumerate(variables):
            vmin, vmax = value_ranges[var]
            title = f"{row_label} ${var}(x, y)$"
            _plot_heatmap_single(axes[row, col], X, Y, data[:, col + 2], CMAP_VAR, cfg, a, b,
                                 vmin=vmin, vmax=vmax,
                                 cbar_label=f"${var}$",
                                 title=title)
        axes[row, 0].set_ylabel(row_label, rotation=0, labelpad=55, va="center", fontsize=12)

    fname = output_dir / f"outputs_{tag}_comparison.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)

    if separate_plots:
        for row in range(n_rows):
            fig_sep, axes_sep = plt.subplots(1, len(variables), figsize=(6 * len(variables), 3),
                                             dpi=FIG_DPI, constrained_layout=True)
            axes_sep = np.atleast_1d(axes_sep)
            data = fem_data if row == 0 else pinn_data_list[row - 1]
            row_label = model_names[row]
            for col, var in enumerate(variables):
                vmin, vmax = value_ranges[var]
                title = f"{row_label} ${var}(x, y)$"
                _plot_heatmap_single(axes_sep[col], X, Y, data[:, col + 2], CMAP_VAR, cfg, a, b,
                                     vmin=vmin, vmax=vmax,
                                     cbar_label=f"${var}$",
                                     title=title)
            fname = output_dir / f"outputs_{tag}_{row_label.replace(' ', '_')}.png"
            fig_sep.savefig(fname, dpi=FIG_DPI)
            plt.close(fig_sep)


def plot_error_heatmaps_multi(pinn_data_list, fem_data, cfg, tag, output_dir,
                              a=None, b=None, strategy_labels=None,
                              separate_plots=False):
    """
    Create stacked rows of error heatmaps, one row per PINN strategy.
    Args:
        pinn_data_list: list of arrays, each shape (N, 5) = [x, y, u, v, p]
        fem_data: array of shape (N, 5) = [x, y, u, v, p]
        cfg: custom config class object
        tag: string to label the file with
        output_dir: path to the relevant plots folder
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        strategy_labels: optional list of labels corresponding to each PINN dataset
        separate_plots: if True, saves each strategy row separately as well
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pinn_data_list = _validate_heatmap_alignment(fem_data, pinn_data_list)
    n_models = len(pinn_data_list)
    if strategy_labels is None:
        strategy_labels = [f"PINN {i+1}" for i in range(n_models)]
    if len(strategy_labels) != n_models:
        raise ValueError("strategy_labels must match the number of pinn_data arrays.")

    X, Y = fem_data[:, 0], fem_data[:, 1]
    variables = ["u", "v", "p", "mean"]
    col_titles = ["|error_u|", "|error_v|", "|error_p|", "mean |error|"]

    error_maps = []
    value_ranges = {var: 0.0 for var in variables}
    for pinn_data in pinn_data_list:
        err_u = np.abs(pinn_data[:, 2] - fem_data[:, 2])
        err_v = np.abs(pinn_data[:, 3] - fem_data[:, 3])
        err_p = np.abs(pinn_data[:, 4] - fem_data[:, 4])
        mean_err = (err_u + err_v + err_p) / 3.0
        error_maps.append({"u": err_u, "v": err_v, "p": err_p, "mean": mean_err})

        value_ranges["u"] = max(value_ranges["u"], np.max(err_u))
        value_ranges["v"] = max(value_ranges["v"], np.max(err_v))
        value_ranges["p"] = max(value_ranges["p"], np.max(err_p))
        value_ranges["mean"] = max(value_ranges["mean"], np.max(mean_err))

    n_rows = n_models
    fig, axes = plt.subplots(n_rows, len(variables), figsize=(6 * len(variables), 3 * n_rows),
                             dpi=FIG_DPI, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, label in enumerate(strategy_labels):
        for col, var in enumerate(variables):
            values = error_maps[row][var]
            title = f"{label} {col_titles[col]}"
            _plot_heatmap_single(axes[row, col], X, Y, values, CMAP_ERR, cfg, a, b,
                                 vmin=0.0, vmax=value_ranges[var],
                                 cbar_math_format=True,
                                 cbar_label="|error|",
                                 title=title)
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=55, va="center", fontsize=12)

    fname = output_dir / f"errors_{tag}_comparison.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)

    if separate_plots:
        for row, label in enumerate(strategy_labels):
            fig_sep, axes_sep = plt.subplots(1, len(variables), figsize=(6 * len(variables), 3),
                                             dpi=FIG_DPI, constrained_layout=True)
            axes_sep = np.atleast_1d(axes_sep)
            for col, var in enumerate(variables):
                values = error_maps[row][var]
                title = f"{label} {col_titles[col]}"
                _plot_heatmap_single(axes_sep[col], X, Y, values, CMAP_ERR, cfg, a, b,
                                     vmin=0.0, vmax=value_ranges[var],
                                     cbar_math_format=True,
                                     cbar_label="|error|",
                                     title=title)
            fname = output_dir / f"errors_{tag}_{label.replace(' ', '_')}.png"
            fig_sep.savefig(fname, dpi=FIG_DPI)
            plt.close(fig_sep)




# ———————————— GLOBAL ANALYSIS ————————————

# --- Helper: Parse Error JSON ---
def _extract_error_summary(summary_path,
                           variables: list = ["u", "v", "p"],
                           metrics: list = ["L2", "L_inf", "MSE"],
                           aggregate_metrics: list = ["mean_L2", "max_L_inf"],
                           strategies: list | None = None):
    """
    Extract the entire summary_train.json into a tidy DataFrame.
    Each row represents one variable/metric or aggregate-metric value for a single run.
    Returns a DataFrame with columns [ab, a, b, n, variable, metric, value].
    """
    summary_path = Path(summary_path)
    with summary_path.open() as f:
        errors = json.load(f)
    
    def extract_geometry(geometry_data: dict, strategy=None):
        rows = []
        a = float(geometry_data["parameters"]["a"])
        b = float(geometry_data["parameters"]["b"])
        for var in variables:
            for met in metrics:
                rows.append({
                    "ab": f'({a}, {b})',
                    "a": a,
                    "b": b,
                    "strategy": strategy,
                    "variable": var,
                    "metric": met,
                    "value": float(geometry_data[var][met]),
                })
        for aggmetric in aggregate_metrics:
            rows.append({
                "ab": f'({a}, {b})',
                "a": a,
                "b": b,
                "strategy": strategy,
                "variable": "aggregate",
                "metric": aggmetric,
                "value": float(geometry_data["aggregate"][aggmetric]),
            })
        return rows

    rows = []
    for entry in errors.values():
        if strategies:  
            # need to descend one level deeper through strategy keys
            for strat, data in entry.items():
                rows.extend(extract_geometry(data, strat))
        else:
            rows.extend(extract_geometry(entry))

    if len(rows) == 0:
        raise ValueError("No error data was found in the summary file.")

    df = pd.DataFrame(rows)
    
    if strategies:
        return df.sort_values(by=["a", "b", "strategy", "variable", "metric"], ignore_index=True)
    else:
        return df.sort_values(by=["a", "b", "variable", "metric"], ignore_index=True)


# --- Error Comparison Point Plots ---
def plot_error_comparison(summary_path, output_dir, cfg, parameter,
                          fixed_ab: list = None, fixed_strat: str = None,
                          strategies: list = None):
    """
    Compare error across all runs, with a specified parameter as the axis. The free parameter can be fixed or averaged.
    Args:
        summary_path: path to summary.json containing errors across runs
        output_dir: path to folder to save plots
        parameter: string specifying the parameter of interest, choices = ["ab", "a", "b", "strategy"]
        fixed_ab:  specified list of [a,b] to use across n; discards other geometries. If None, takes average errors across all (a,b). Requires variable="n".
        fixed_strat: specified name of a prediction strategy to use across (a,b); discards other strategies. If None, takes average errors across all strategies. Requires variable!="".
        strategies: include if comparing test set errors keyed by strategies
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse args
    parameter_choices = ["ab", "a", "b", "strategy"]
    if parameter not in parameter_choices:
        raise ValueError(f"Received parameter = {parameter}, but must be one of {parameter_choices}")
    if fixed_ab is not None and len(fixed_ab) != 2:
        raise ValueError(f"fixed_ab must be a list of length 2; received {fixed_ab}.")
    if parameter == "strategy":
        if fixed_strat is not None:
            raise ValueError(f"fixed_strat is not compatible with parameter={parameter}.")
    if parameter != "strategy":
        if fixed_ab is not None:
            raise ValueError(f"fixed_ab is not compatible with parameter={parameter}.")
    
    # Error Summary Dict Keys
    VARS = ["u", "v", "p"]
    METRICS = ["L2", "L_inf"]
    AGGREGATE_METRICS = ["mean_L2", "max_L_inf"]
    
    PARAMETER_LABELS = {
        "a": "ellipse width (a)",
        "b": "ellipse height (b)",
        "ab": "ellipse geometry (a, b)",
        "strategy": "PINN prediction strategy",
    }

    error_df = _extract_error_summary(summary_path, VARS, METRICS, 
                                      AGGREGATE_METRICS, strategies=strategies)

    # filter to specified value of free parameter, if applicable
    if fixed_ab is not None:
        error_df = error_df[(error_df["a"].astype(float) == float(fixed_ab[0])) & (error_df["b"].astype(float) == float(fixed_ab[1]))]
    if fixed_strat is not None:
        error_df = error_df[error_df["strategy"].astype(str) == fixed_strat]

    if error_df.empty:
        raise ValueError("No matching runs were found for the requested summary selection.")

    # index by parameter of interest (axis)
    if parameter == "strategy":
        error_df["parameter_value"] = error_df["strategy"]
        averaging = fixed_ab is None
    else:
        error_df["parameter_value"] = error_df["a"]
        if parameter == "b":
            error_df["parameter_value"] = error_df["b"]
        elif parameter == "ab":
            error_df["parameter_value"] = error_df.apply(lambda row: f"({row['a']}, {row['b']})", axis=1)
        averaging = fixed_strat is None

    if averaging:
        error_df = error_df.groupby(["parameter_value", "variable", "metric"], observed=True, as_index=False)["value"].mean()

    parameter_order = error_df["parameter_value"].drop_duplicates().tolist()
    
    # Plot variable-level metrics
    var_plot_data = error_df[error_df["variable"].isin(VARS) 
                             & error_df["metric"].isin(METRICS)]
    for metric in METRICS:
        plot_data = var_plot_data[var_plot_data["metric"] == metric]
        ax = sns.pointplot(
            data=plot_data,
            x="parameter_value",
            y="value",
            hue="variable",
            hue_order=VARS,
            order=parameter_order,
            palette=COLOR_VARIABLE_MAP,
            markers=["o", "s", "^"],
            linestyles=["-", "--", ":"],
            dodge=True,
        )
        ax.tick_params("x", rotation=45, rotation_mode="xtick")
        plt.xlabel(PARAMETER_LABELS[parameter])
        plt.ylabel(metric)
        
        # label based on args
        title = f"{metric} error across {parameter}"
        fname = f"errors_by_{parameter}_{metric}"
        if fixed_strat:
            title += f", (, {fixed_strat} prediction)"
            fname += f"_{fixed_strat}"
        elif fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
            fname += f"_a{fixed_ab[0]}_b{fixed_ab[1]}"
        else:
            averaged_across = "ab" if parameter == "strategy" else "strategy"
            title += f", averaged across {averaged_across}"
        fname += ".png"
            
        plt.title(title)
        plt.tight_layout()
        
        savepath = output_dir / fname
        ax.figure.savefig(savepath, dpi=FIG_DPI)
        plt.close(ax.figure)
    
    # Plot aggregate metrics
    agg_plot_data = error_df[(error_df["variable"] == "aggregate") 
                             & error_df["metric"].isin(AGGREGATE_METRICS)]
    for aggmetric in AGGREGATE_METRICS:
        plot_data = agg_plot_data[agg_plot_data["metric"] == aggmetric]
        ax = sns.pointplot(
            data=plot_data,
            x="parameter_value",
            y="value",
            order=parameter_order,
            color=COLOR_AGGREGATE
        )
        ax.tick_params("x", rotation=45, rotation_mode="xtick")
        plt.xlabel(PARAMETER_LABELS[parameter])
        plt.ylabel(aggmetric)
        
        # label based on args
        title = f"{aggmetric} error of all outputs, across {parameter}"
        fname = f"errors_by_{parameter}_{aggmetric}"
        if fixed_strat:
            title += f", (, {fixed_strat} prediction)"
            fname += f"_{fixed_strat}"
        elif fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
            fname += f"_{aggmetric}_a{fixed_ab[0]}_b{fixed_ab[1]}"
        else:
            averaged_across = "ab" if parameter == "strategy" else "strategy"
            title += f", averaged across {averaged_across}"
        fname += ".png"
            
        plt.title(title)
        plt.tight_layout()
        
        savepath = output_dir / fname
        ax.figure.savefig(savepath, dpi=FIG_DPI)
        plt.close(ax.figure)
    

# --- Error Comparison Heatmaps ---
def plot_error_comparison_2d(summary_path, output_dir, cfg, index_parameter="strategy", col_parameter="ab"):
    """
    Create 2D grid heatmaps of errors for all combinations of two parameters.
    Args:
        summary_path: path to summary.json containing errors across runs
        output_dir: path to folder to save plots
        parameter_1: string specifying the parameter to plot on the x-axis, choices = ["strategy", "a", "b", "ab"]
        parameter_2: string specifying the parameter of plot on the y-axis, choices = ["strategy", "a", "b", "ab"]
    """
    
    VARS = ["u", "v", "p"]
    METRICS = ["L2", "L_inf"]
    AGGREGATE_METRICS = ["mean_L2", "max_L_inf"]
    
    PARAMETER_LABELS = {
        "strategy": "prediction strategy",
        "a": "ellipse width (a)",
        "b": "ellipse height (b)",
        "ab": "ellipse geometry (a, b)"
    }
    
    # cols are a, b, n, variable, metric, value
    error_df = _extract_error_summary(summary_path, VARS, METRICS, 
                                      AGGREGATE_METRICS, strategies=True)
    
    # Create plot for each metric for each variable
    for var in VARS:
        for metric in METRICS:
            selected = error_df[(error_df["variable"] == var) 
                                & (error_df["metric"] == metric)]
            plot_data = selected.pivot(index=index_parameter, columns=col_parameter, values="value")
            
            ax = sns.heatmap(
                plot_data,
                cmap=CMAP_ERR,
                annot=True,
                linewidth=1.0
            )
            
            plt.xlabel(PARAMETER_LABELS[col_parameter])
            plt.ylabel(PARAMETER_LABELS[index_parameter])
            
            # label based on args
            title = f"{metric} error for ${var}(x,y)$ across runs"
            fname = f"errors_heatmap_{var}_{metric}.png"
                
            plt.title(title)
            plt.tight_layout()
            
            savepath = output_dir / fname
            ax.figure.savefig(savepath, dpi=FIG_DPI)
            plt.close(ax.figure)
    
    # Create a plot for each aggregate metric
    for aggmetric in AGGREGATE_METRICS:
            selected = error_df[(error_df["variable"] == "aggregate") 
                                & (error_df["metric"] == aggmetric)]
            plot_data = selected.pivot(index=index_parameter, columns=col_parameter, values="value")
            
            ax = sns.heatmap(
                plot_data,
                cmap=CMAP_ERR,
                annot=True,
                linewidth=0.5
            )
            
            plt.xlabel(PARAMETER_LABELS[col_parameter])
            plt.ylabel(PARAMETER_LABELS[index_parameter])
            
            # label based on args
            title = f"{aggmetric} error across runs and outputs"
            fname = f"errors_heatmap_aggregate_{aggmetric}.png"
                
            plt.title(title)
            plt.tight_layout()
            
            savepath = output_dir / fname
            ax.figure.savefig(savepath, dpi=FIG_DPI)
            plt.close(ax.figure)
    