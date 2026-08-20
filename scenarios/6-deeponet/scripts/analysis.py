# analysis.py

"""
2D Stenosis Physics-Informed Deep Operator Network
    Analysis and visualization functions

    Note that some of these functions contain legagcy parameters or naming conventions
    since they were copied in from previous scenarios.

Evan Hackstadt
Rugonyi Lab
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from geometry import ellipse_mask


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
    Computes L2 and L_inf (max absolute error) between prediction and ground-truth data.
    Args:
        pinn_data: array of shape (N, 5) with columns = [x, y, u_pinn, v_pinn, p_pinn]
        fem_data:  array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
    Returns:
        errors: dict containing L2 and L_inf for u, v, p; and mean L2 and max L_inf
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
            "L2": float(l2_rel),
            "L_inf": float(l_inf),
        }
    
    # store aggregated error across variables (mean L2, max L_inf)
    mean_L2 = np.mean([errors[var]["L2"] for var in VARS])
    max_L_inf = np.max([errors[var]["L_inf"] for var in VARS])
    errors["aggregate"] = {"mean_L2": float(mean_L2), 
                           "max_L_inf": float(max_L_inf)}
    
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
        

# --- Data Helper Function ---
def _array_mask_ab(array, a, b, a_idx = 2, b_idx = 3, inverse: bool = False):
    mask = np.isclose(array[:, a_idx], a) & np.isclose(array[:, b_idx], b)
    if inverse:
        return array[~mask]
    else:
        return array[mask]

# --- Plot Domain ---
def plot_domain(cfg, a, b, output_dir, labeled_pts = None, 
                plot_other_geos_labeled = False):
    """
    Visualizes the spatial domain and sampled labeled points, if provided.
    Args:
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        output_dir: path to the relevant plots folder to save plot
        labeled_pts: array of shape (N, ≥4) with columns = [x,y,a,b,...]
        plot_other_geos_labeled: whether or not to also plot the labeled points from other geometries
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
        components = ["u", "v", "p"]
        lb_cmp = ', '.join([components[i] for i in cfg.test_observation_components])
        
        pts_this_ab  = _array_mask_ab(labeled_pts, a, b, 2, 3)
        plt.scatter(pts_this_ab[:, 0], pts_this_ab[:, 1], 
                    s=25, c=COLOR_PINN, label='This geometry')
        if plot_other_geos_labeled:
            pts_other_ab = _array_mask_ab(labeled_pts, a, b, 2, 3, inverse=True)
            plt.scatter(pts_other_ab[:, 0], pts_other_ab[:, 1], 
                        s=25, c='grey', alpha=0.3, label='Other geometries')
            plt.legend()
            plt.title(f"Domain with n={pts_this_ab.shape[0]} of {labeled_pts.shape[0]} total measurements of component(s) ({lb_cmp}).")
        else:
            plt.title(f"Domain with n={pts_this_ab.shape[0]} measurements of component(s) ({lb_cmp}).")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.geo_tag(a, b)

    # save plot
    plt.savefig(output_dir / f"domain_{tag}.png", dpi=FIG_DPI)
    plt.close()


def plot_all_domains(cfg, output_dir, show_train=True, show_test=True):
    """
    Plot the channel domain and draw the ellipse outlines for every train/test geometry.
    Args:
        cfg: custom config class object
        output_dir: path to the plots folder
        show_train: whether to draw train geometries
        show_test: whether to draw test geometries
    """
    if not (show_train or show_test):
        raise ValueError("At least one of show_train or show_test must be True.")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    ax.set_xlim(-cfg.L/2, cfg.L/2)
    ax.set_ylim(0, cfg.H_max)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Train and test ellipse geometries")

    if show_train:
        for a, b in cfg.train_geometries:
            ellipse = Ellipse(
                xy=(cfg.x_c, cfg.y_c),
                width=a * 2,
                height=b * 2,
                edgecolor="C0",
                facecolor="none",
                linestyle="-",
                linewidth=1.5,
                alpha=0.85,
                zorder=10,
            )
            ax.add_patch(ellipse)

    if show_test:
        for a, b in cfg.test_geometries:
            ellipse = Ellipse(
                xy=(cfg.x_c, cfg.y_c),
                width=a * 2,
                height=b * 2,
                edgecolor="C1",
                facecolor="none",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                zorder=10,
            )
            ax.add_patch(ellipse)

    handles = []
    if show_train:
        handles.append(Line2D([0], [0], color="C0", lw=2, linestyle="-", label="train geometries"))
    if show_test:
        handles.append(Line2D([0], [0], color="C1", lw=2, linestyle="--", label="test geometries"))
    ax.legend(handles=handles, framealpha=0.9)
    ax.set_aspect("equal")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / "all_domains.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)


# ———————————— HEATMAP FUNCTIONS ————————————

# --- Helper: Format Data ---
def _prepare_grid_data(x_query, y_query, values, cfg, a, b):
    # Reconstruct the original uniform grid axes
    xs = np.linspace(-cfg.L/2, cfg.L/2, cfg.query_nx)
    ys = np.linspace(0, cfg.H_max, cfg.query_ny)
    XX, YY = np.meshgrid(xs, ys)
    Z_flat = np.full(XX.size, np.nan)

    x_query = np.asarray(x_query)
    y_query = np.asarray(y_query)
    values = np.asarray(values)
    if not (x_query.ndim == y_query.ndim == values.ndim == 1):
        raise ValueError("Heatmap query coordinates and values must be one-dimensional.")
    if not (len(x_query) == len(y_query) == len(values)):
        raise ValueError("Heatmap query coordinates and values must have the same length.")

    # Use the supplied query coordinates as the source of truth. Recomputing
    # the ellipse mask can disagree at boundary points due to float rounding.
    x_idx = np.rint((x_query - xs[0]) / (xs[1] - xs[0])).astype(int)
    y_idx = np.rint((y_query - ys[0]) / (ys[1] - ys[0])).astype(int)
    valid_grid_points = (
        (x_idx >= 0) & (x_idx < len(xs)) &
        (y_idx >= 0) & (y_idx < len(ys))
    )
    if np.any(valid_grid_points):
        valid_grid_points &= (
            np.isclose(xs[np.clip(x_idx, 0, len(xs) - 1)], x_query, rtol=1e-6, atol=1e-7) &
            np.isclose(ys[np.clip(y_idx, 0, len(ys) - 1)], y_query, rtol=1e-6, atol=1e-7)
        )
    if not np.all(valid_grid_points):
        raise ValueError("Heatmap query coordinates do not lie on the configured uniform grid.")

    flat_indices = y_idx * len(xs) + x_idx
    if len(np.unique(flat_indices)) != len(flat_indices):
        raise ValueError("Heatmap query coordinates contain duplicate grid points.")
    Z_flat[flat_indices] = values
    # Mesh
    ZZ = Z_flat.reshape(cfg.query_ny, cfg.query_nx)

    return XX, YY, ZZ


# --- Helper: Plot One Heatmap ---
def _plot_heatmap_single(axis, X, Y, values, cmap, cfg, a, b,
                         vmin=None, vmax=None, cbar_math_format=False, 
                         cbar_label=None, cbar_shrink=1.0, title=None):
    
    X_grid, Y_grid, Z_grid = _prepare_grid_data(X, Y, values, cfg, a, b)
    pcm = axis.pcolormesh(X_grid, Y_grid, Z_grid, cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          shading="auto")
    
    cbar = plt.colorbar(pcm, ax=axis, label=cbar_label, shrink=cbar_shrink)
    
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
        "FEM":  fem_data,
        "PINN": pinn_data,
    }

    # Shared colormap range across both panels (use FEM as reference)
    _, _, M_fem_full = _extract(fem_data, idx)
    vmax = np.percentile(M_fem_full, 98)   # clip extreme values near walls

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=FIG_DPI, constrained_layout=True)

    for ax, (label, data) in zip(axes, models.items()):
        U_q, V_q, M_q = _extract(data, idx)

        # Scale arrow length slightly with magnitude while preserving direction.
        # This keeps low-magnitude arrows shorter and high-magnitude arrows longer,
        # without fully normalizing all arrows to equal length.
        norm_mag = np.where(M_q > 0, M_q, 1e-12)
        M_norm = M_q / np.max(M_q) if np.max(M_q) > 0 else M_q
        length_mod = 0.4 + 1.1 * np.sqrt(M_norm)
        U_s = U_q / norm_mag * length_mod
        V_s = V_q / norm_mag * length_mod

        qv = ax.quiver(
            X_q, Y_q,
            U_s, V_s,
            M_q,                        # color by magnitude
            cmap=CMAP_VAR,
            clim=(0, vmax),
            scale=nx_q * 1.25,          # tune: larger = shorter arrows
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



# ———————————— GLOBAL ANALYSIS HEATMAPS ————————————

# Stack FEM + DeepONet heatmaps across geometries
def plot_output_heatmaps_multi(deeponet_data_dict, fem_data_dict, cfg, tag, output_dir):
    """
    Create a stacked FEM/DeepONet comparison for multiple geometries.
    Args:
        deeponet_data_dict: dict mapping (a, b) to arrays with columns [x, y, u, v, p]
        fem_data_dict: dict mapping (a, b) to arrays with columns [x, y, u, v, p]
        cfg: custom config class object
        tag: string to label the file with
        output_dir: path to the relevant plots folder
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not deeponet_data_dict or not fem_data_dict:
        raise ValueError("deeponet_data_dict and fem_data_dict must not be empty.")
    if set(deeponet_data_dict) != set(fem_data_dict):
        raise ValueError("deeponet_data_dict and fem_data_dict must contain the same geometry keys.")

    configured_geometries = getattr(cfg, "test_geometries", None)
    geometries = [geo for geo in configured_geometries or () if geo in fem_data_dict]
    geometries.extend(geo for geo in fem_data_dict if geo not in geometries)

    for geometry in geometries:
        _validate_heatmap_alignment(fem_data_dict[geometry], deeponet_data_dict[geometry])

    variables = ["u", "v", "p"]

    n_rows = len(geometries)
    n_cols = 2 * len(variables)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             dpi=FIG_DPI, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for geometry_index, (a, b) in enumerate(geometries):
        row_data = [("FEM", fem_data_dict[(a, b)]),
                    ("DeepONet", deeponet_data_dict[(a, b)])]
        for variable_index, variable in enumerate(variables):
            pair_values = np.concatenate([
                fem_data_dict[(a, b)][:, variable_index + 2],
                deeponet_data_dict[(a, b)][:, variable_index + 2],
            ])
            vmin = np.min(pair_values)
            vmax = np.max(pair_values)
            if vmin == vmax:
                vmax = vmin + np.finfo(float).eps

            for model_offset, (model_name, data) in enumerate(row_data):
                col = 2 * variable_index + model_offset
                _plot_heatmap_single(
                    axes[geometry_index, col], data[:, 0], data[:, 1],
                    data[:, variable_index + 2],
                    CMAP_VAR, cfg, a, b, vmin=vmin, vmax=vmax,
                    cbar_label=f"${variable}$",
                    cbar_shrink=0.6,
                    title=f"{model_name} ${variable}(x, y)$",
                )

        axes[geometry_index, 0].set_ylabel(
            f"$({a:g}, {b:g})$", rotation=0, labelpad=65,
            va="center", fontsize=12,
        )

    fname = output_dir / f"outputs_{tag}_comparison.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)


# Stack error heatmaps across geometries
def plot_error_heatmaps_multi(deeponet_data_dict, fem_data_dict, cfg, tag, output_dir):
    """
    Create one row of error heatmaps for each geometry.
    Args:
        deeponet_data_dict: dict mapping (a, b) to arrays with columns [x, y, u, v, p]
        fem_data_dict: dict mapping (a, b) to arrays with columns [x, y, u, v, p]
        cfg: custom config class object
        tag: string to label the file with
        output_dir: path to the relevant plots folder
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not deeponet_data_dict or not fem_data_dict:
        raise ValueError("deeponet_data_dict and fem_data_dict must not be empty.")
    if set(deeponet_data_dict) != set(fem_data_dict):
        raise ValueError("deeponet_data_dict and fem_data_dict must contain the same geometry keys.")

    configured_geometries = getattr(cfg, "test_geometries", None)
    geometries = [geo for geo in configured_geometries or () if geo in fem_data_dict]
    geometries.extend(geo for geo in fem_data_dict if geo not in geometries)

    variables = ["u", "v", "p", "mean"]
    col_titles = ["|error_u|", "|error_v|", "|error_p|", "mean |error|"]
    error_maps = []

    for geometry in geometries:
        fem_data = fem_data_dict[geometry]
        deeponet_data = deeponet_data_dict[geometry]
        _validate_heatmap_alignment(fem_data, deeponet_data)

        errors = {
            "u": np.abs(deeponet_data[:, 2] - fem_data[:, 2]),
            "v": np.abs(deeponet_data[:, 3] - fem_data[:, 3]),
            "p": np.abs(deeponet_data[:, 4] - fem_data[:, 4]),
        }
        errors["mean"] = sum(errors[variable] for variable in ("u", "v", "p")) / 3.0
        error_maps.append(errors)

    n_rows = len(geometries)
    fig, axes = plt.subplots(n_rows, len(variables),
                             figsize=(6 * len(variables), 3 * n_rows),
                             dpi=FIG_DPI, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, ((a, b), errors) in enumerate(zip(geometries, error_maps)):
        for col, (variable, col_title) in enumerate(zip(variables, col_titles)):
            vmax = np.max(errors[variable])
            if vmax == 0:
                vmax = np.finfo(float).eps
            _plot_heatmap_single(
                axes[row, col], fem_data_dict[(a, b)][:, 0], fem_data_dict[(a, b)][:, 1],
                errors[variable], CMAP_ERR, cfg, a, b,
                vmin=0.0, vmax=vmax,
                cbar_math_format=True,
                cbar_label="|error|",
                cbar_shrink=0.8,
                title=f"DeepONet {col_title}, geometry $(a, b)=({a:g}, {b:g})$",
            )
        axes[row, 0].set_ylabel(f"$(a, b)=({a:g}, {b:g})$", rotation=0,
                                labelpad=55, va="center", fontsize=12)

    fname = output_dir / f"errors_{tag}_comparison.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)


# Stack FEM + DeepONet velocity quivers across geometries
def plot_velocity_quiver_multi(deeponet_data_dict, fem_data_dict, cfg, tag, output_dir,
                                nx_q=24, ny_q=12):
    """
    Create one FEM/DeepONet quiver row for each geometry.
    Args:
        deeponet_data_dict: dict mapping (a, b) to arrays with columns [x, y, u, v, p]
        fem_data_dict: dict mapping (a, b) to arrays with columns [x, y, u, v, p]
        cfg: custom config class object
        tag: string to label the file with
        output_dir: path to the relevant plots folder
        nx_q, ny_q: quiver grid resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not deeponet_data_dict or not fem_data_dict:
        raise ValueError("deeponet_data_dict and fem_data_dict must not be empty.")
    if set(deeponet_data_dict) != set(fem_data_dict):
        raise ValueError("deeponet_data_dict and fem_data_dict must contain the same geometry keys.")

    configured_geometries = getattr(cfg, "test_geometries", None)
    geometries = [geo for geo in configured_geometries or () if geo in fem_data_dict]
    geometries.extend(geo for geo in fem_data_dict if geo not in geometries)

    from scipy.spatial import cKDTree

    xs_q = np.linspace(-cfg.L / 2, cfg.L / 2, nx_q)
    ys_q = np.linspace(0, cfg.H_max, ny_q)
    XX_q, YY_q = np.meshgrid(xs_q, ys_q)
    flat_grid = np.column_stack([XX_q.ravel(), YY_q.ravel()])

    def _extract_velocity(data, indices):
        U = data[indices, 2]
        V = data[indices, 3]
        M = np.sqrt(U**2 + V**2)
        return U, V, M

    quiver_data = []
    for a, b in geometries:
        fem_data = fem_data_dict[(a, b)]
        deeponet_data = deeponet_data_dict[(a, b)]
        _validate_heatmap_alignment(fem_data, deeponet_data)

        outside_grid = ellipse_mask(flat_grid[:, 0], flat_grid[:, 1], cfg, a, b)
        query_points = flat_grid[outside_grid]
        tree = cKDTree(fem_data[:, 0:2])
        _, indices = tree.query(query_points)

        model_data = []
        for data in (fem_data, deeponet_data):
            velocity_data = _extract_velocity(data, indices)
            model_data.append(velocity_data)
        pair_vmax = max(np.max(model_data[0][2]), np.max(model_data[1][2]))
        if pair_vmax == 0:
            pair_vmax = np.finfo(float).eps
        quiver_data.append((a, b, query_points, model_data, pair_vmax))

    n_rows = len(geometries)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows),
                                dpi=FIG_DPI, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, (a, b, query_points, model_data, vmax) in enumerate(quiver_data):
        for col, (model_name, (U_q, V_q, M_q)) in enumerate(
                zip(("FEM", "DeepONet"), model_data)):
            norm_mag = np.where(M_q > 0, M_q, 1e-12)
            M_norm = M_q / np.max(M_q) if np.max(M_q) > 0 else M_q
            length_mod = 0.5 + 1.1 * np.sqrt(M_norm)
            U_s = U_q / norm_mag * length_mod
            V_s = V_q / norm_mag * length_mod

            qv = axes[row, col].quiver(
                query_points[:, 0], query_points[:, 1],
                U_s, V_s, M_q,
                cmap=CMAP_VAR,
                clim=(0, vmax),
                scale=nx_q * 1.25,
                scale_units="width",
                width=0.003,
                headwidth=4,
                headlength=5,
            )
            plt.colorbar(qv, ax=axes[row, col], label=r"$\|\mathbf{v}\|$")

            axes[row, col].add_patch(Ellipse(
                xy=(cfg.x_c, cfg.y_c),
                width=a * 2, height=b * 2,
                edgecolor="black", facecolor="none",
                linestyle="-", linewidth=1.0, zorder=5,
            ))
            axes[row, col].set_xlim(-cfg.L / 2, cfg.L / 2)
            axes[row, col].set_ylim(0, cfg.H_max)
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlabel("$x$")
            axes[row, col].set_ylabel("$y$")
            axes[row, col].set_title(
                f"{model_name}, geometry $(a, b)=({a:g}, {b:g})$ "
                "$\\mathbf{v}(x,y)$"
            )

    fname = output_dir / f"quiver_{tag}_comparison.png"
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)



# ———————————— GLOBAL ERROR ANALYSIS ————————————

# --- Helper: Parse Error JSON ---
def _extract_error_summary(summary_path,
                           variables: list = ["u", "v", "p"],
                           metrics: list = ["L2", "L_inf", "MSE"],
                           aggregate_metrics: list = ["mean_L2", "max_L_inf"],
                           strategies: bool = True):
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
        area = 3.14 * a * b / 2
        for var in variables:
            for met in metrics:
                rows.append({
                    "ab": f'({a}, {b})',
                    "ab_area": round(area, 3),
                    "a": a,
                    "b": b,
                    "strategy": strategy,
                    "variable": var,
                    "metric": met,
                    "value": float(geometry_data[var][met]),
                })
        for aggmetric in aggregate_metrics:
            rows.append({
                "ab": f'({a:.2g}, {b:.2g})',
                "ab_area": round(area, 3),
                "a": round(a, 2),
                "b": round(b, 2),
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

    return df.sort_values(by=["a", "b", "variable", "metric"], ignore_index=True)


# --- Error Comparison Point Plots ---
def plot_error_comparison(summary_path, output_dir, cfg, parameter,
                          fixed_ab: list = None, fixed_strat: str = None,
                          strategies: list = None, label: str = "", lineplot: bool = False):
    """
    Compare error across all runs, with a specified parameter as the axis. The free parameter can be fixed or averaged.
    Args:
        summary_path: path to summary.json containing errors across runs
        output_dir: path to folder to save plots
        parameter: string specifying the parameter of interest, choices = ["ab", "a", "b", "strategy"]
        fixed_ab:  specified list of [a,b] to use across n; discards other geometries. If None, takes average errors across all (a,b). Requires variable="n".
        fixed_strat: specified name of a prediction strategy to use across (a,b); discards other strategies. If None, takes average errors across all strategies. Requires variable!="".
        strategies: list of strategies in desired order, if dictionary has an additional key layer of strategies
        label: train/test label (optional)
        lineplot: whether to draw a lineplot (where x-axis parameter vals aren't individually labeled) instead of default pointplot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse args
    parameter_choices = ["a", "b", "ab", "ab_area", "strategy"]
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
        "ab": "ellipse (a, b)",
        "ab_area": "ellipse area",
        "strategy": "PINN prediction strategy",
    }

    error_df = _extract_error_summary(summary_path, VARS, METRICS, 
                                      AGGREGATE_METRICS, strategies=strategies)
    if strategies:
            error_df['strategy'] = pd.Categorical(error_df['strategy'], strategies)
    
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
        elif parameter == "ab_area":
            error_df["parameter_value"] = error_df["ab_area"]
        averaging = fixed_strat is None

    if averaging:
        error_df = error_df.groupby(["parameter_value", "variable", "metric"], observed=True, as_index=False)["value"].mean()
    
    parameter_order = error_df["parameter_value"].drop_duplicates().tolist()
    
    # Plot variable-level metrics
    var_plot_data = error_df[error_df["variable"].isin(VARS) 
                             & error_df["metric"].isin(METRICS)]
    for metric in METRICS:
        plot_data = var_plot_data[var_plot_data["metric"] == metric]

        if lineplot:
            ax = sns.lineplot(
                data=plot_data,
                x="parameter_value",
                y="value",
                style="variable",
                hue="variable",
                hue_order=VARS,
                palette=COLOR_VARIABLE_MAP,
            )
        else:
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

        ax.set_xlabel(PARAMETER_LABELS[parameter], labelpad=18)
        if parameter == "ab":
            ax.text(0.01, -0.32, "smallest", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10)
            ax.text(0.99, -0.32, "largest", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10)
        ax.set_ylabel(metric)
        
        # label based on args
        title = f"{metric} error of each {label} output, across {parameter}"
        fname = f"errors_by_{parameter}_{metric}"
        if fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
            fname += f"_a{fixed_ab[0]}_b{fixed_ab[1]}"
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

        if lineplot:
            ax = sns.lineplot(
                data=plot_data,
                x="parameter_value",
                y="value",
                color=COLOR_AGGREGATE
            )
        else:
            ax = sns.pointplot(
                data=plot_data,
                x="parameter_value",
                y="value",
                order=parameter_order,
                color=COLOR_AGGREGATE
            )
            ax.tick_params("x", rotation=45, rotation_mode="xtick")
        
        ax.set_xlabel(PARAMETER_LABELS[parameter], labelpad=18)
        if parameter == "ab":
            ax.text(0.01, -0.32, "smallest", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10)
            ax.text(0.99, -0.32, "largest", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10)
        ax.set_ylabel(aggmetric)
        
        # label based on args
        title = f"{aggmetric} error of all {label} outputs, across {parameter}"
        fname = f"errors_by_{parameter}_{aggmetric}"
        if fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
            fname += f"_{aggmetric}_a{fixed_ab[0]}_b{fixed_ab[1]}"
        fname += ".png"
            
        plt.title(title)
        plt.tight_layout()
        
        savepath = output_dir / fname
        ax.figure.savefig(savepath, dpi=FIG_DPI)
        plt.close(ax.figure)