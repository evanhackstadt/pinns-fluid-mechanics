# analysis.py

"""
2D Parameterized Stenosis
    Analysis and visualization functions

Evan Hackstadt
Rugonyi Lab
"""


import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse


CMAP_VAR   = "rainbow"
CMAP_ERR   = "hot_r"
COLOR_PINN = "#185FA5"   # blue  – PINN prediction
COLOR_TRUE = "#E8593C"   # coral – analytical ground truth
FIG_DPI    = 200


# ———————————— ANALYSIS FUNCTIONS ————————————

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
def save_errors(errors, output_dir, tag):
    """
    Write the calculated errors to a json file.
    Args:
        errors (dict): output of compute_errors() containing L2, L_inf, and MSE
        output_dir: path to the folder to save the file
        tag: string noting a, b, and Re parameters of the run
    """
    
    error_path = os.path.join(output_dir, f"errors.json")
    errors["parameters"] = tag
    
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)



# --- Loss Curves ---
def plot_loss_curves(loss_history, output_dir):
    """
    Plot PDE residual loss and BC loss separately over training iterations.
    Args:
        loss_history: DeepXDE LossHistory object, rows = iterations, cols = loss terms
        output_dir: path to the relevant plots folder to save plot
    """
    steps      = np.array(loss_history.steps)
    loss_train = np.array(loss_history.loss_train)   # (iters, n_terms)
    loss_test  = np.array(loss_history.loss_test)   # (iters, n_terms)
    
    labels = {"train": loss_train, "test": loss_test}
    
    for label, loss in labels.items():
            
        # loss terms in order:
        #   PDE_continuity, PDE_x_momentum, PDE_y_momentum, 
        #   BC_inlet_p, BC_outlet_p, BC_wall_u, BC_wall_v
        pde_cont = loss[:, 0]
        pde_x_m = loss[:, 1]
        pde_y_m = loss[:, 2]
        bc_i_p = loss[:, 3]
        bc_o_p = loss[:, 4]
        bc_w_u = loss[:, 5]
        bc_w_v = loss[:, 6]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)

        ax.semilogy(steps, pde_cont, color="#280CAC", lw=1.5, label="PDE (continuity)")
        ax.semilogy(steps, pde_x_m, color="#1C8ADE", lw=1.5, label="PDE (x-momentum)")
        ax.semilogy(steps, pde_y_m, color="#2AD8E1", lw=1.5, label="PDE (y-momentum)")
        ax.semilogy(steps, bc_i_p,  color="#DF18A6", lw=1.5, label="BC (inlet pressure)")
        ax.semilogy(steps, bc_o_p,  color="#D20B0B", lw=1.5, label="BC (outlet pressure)")
        ax.semilogy(steps, bc_w_u,  color="#E2750E", lw=1.5, label="BC (wall x-velocity)")
        ax.semilogy(steps, bc_w_v,  color="#F1CF0E", lw=1.5, label="BC (wall y-velocity)")
    
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(f"Loss history - {label}")
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", lw=0.35, alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    
        plt.tight_layout()
        fname = os.path.join(output_dir, f"loss_curves_terms_{label}.png")
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        
        # Also save summarized PDE and BC curves
        pde_total = pde_cont + pde_x_m + pde_y_m
        bc_total = bc_i_p + bc_o_p + bc_w_u + bc_w_v

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
        fname = os.path.join(output_dir, f"loss_curves_summed_{label}.png")
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        


# --- Plot Domain ---
def plot_domain(cfg, a, b, output_dir):
    """
    Visualizes the spatial domain (ellipse obstruction on the 2D channel)
    Args:
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        output_dir: path to the relevant plots folder to save plot
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
    
    # save plot
    plt.savefig(os.path.join(output_dir, f"geometry.png"), dpi=FIG_DPI)
    plt.close()


# ———————————— HEATMAP FUNCTIONS ————————————


# --- Helper: Format Data ---
def _prepare_grid_data(x, y, values):
    """Reshape flattened query coordinates into a regular 2D grid for pcolormesh."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    if x.shape != y.shape or x.shape != values.shape:
        raise ValueError(
            f"Coordinate/value shape mismatch: x={x.shape}, y={y.shape}, values={values.shape}"
        )

    x_vals = np.unique(x)
    y_vals = np.unique(y)
    if x_vals.size * y_vals.size != values.size:
        raise ValueError(
            "Expected a regular rectilinear grid of query points for heatmap plotting."
        )

    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = values.reshape(len(y_vals), len(x_vals))
    return X_grid, Y_grid, Z_grid


# --- Helper: Plot One Heatmap ---
def plot_heatmap_single(axis, X, Y, values, cmap, cfg, a, b,
                        cbar_math_format=False, cbar_label=None, title=None):
    
    X_grid, Y_grid, Z_grid = _prepare_grid_data(X, Y, values)
    pcm = axis.pcolormesh(X_grid, Y_grid, Z_grid, cmap=cmap, shading="auto")
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
            
            values = data[:, j+2]
            axes[i, j] = plot_heatmap_single(axes[i, j], X, Y, values, CMAP_VAR, cfg, a, b,
                                             cbar_label=f"${var}$",
                                             title=f"{model} ${var}(x, y)$")
            
            # if saving plots separately
            if separate_plots:
                fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
                plot_heatmap_single(ax_sep, X, Y, values, CMAP_VAR, cfg, a, b,
                                    cbar_label=f"${var}$",
                                    title=f"{model} ${var}(x, y)$")
                fname = os.path.join(output_dir, f"{tag}_{model}_{var}_output.png")
                fig_sep.savefig(fname, dpi=FIG_DPI)
                plt.close(fig_sep)
        
    # save multiplot
    fname = os.path.join(output_dir, f"{tag}_outputs.png")
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
            fname = os.path.join(output_dir, f"{tag}_{var}_error.png")
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
        fname = os.path.join(output_dir, f"{tag}_total_error.png")
        fig_sep.savefig(fname, dpi=FIG_DPI)
        plt.close(fig_sep)
        
    # save multiplot
    fname = os.path.join(output_dir, f"{tag}_errors.png")
    fig.savefig(fname, dpi=FIG_DPI)
    plt.close(fig)
    
    
    
    
'''
X_grid, Y_grid, Z_grid = _prepare_grid_data(X, Y, values)
            pcm = axes[i, j].pcolormesh(X_grid, Y_grid, Z_grid, cmap=CMAP_VAR, shading="auto")
            plt.colorbar(pcm, ax=axes[i, j], label=f"${var}$")

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
                axes[i, j].add_patch(ellipse)

            # decorate
            axes[i, j].set_xlabel("$x$")
            axes[i, j].set_ylabel("$y$")
            axes[i, j].set_title(f"{model} ${var}(x, y)$")
            axes[i, j].set_aspect("equal")
'''
    
    
'''
X_grid, Y_grid, Z_grid = _prepare_grid_data(X, Y, err)
        pcm = axes[a_i, a_j].pcolormesh(X_grid, Y_grid, Z_grid, cmap=CMAP_ERR, shading="auto")
        cbar = plt.colorbar(pcm, ax=axes[a_i, a_j], label="|error|")
        
        # format
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
            axes[a_i, a_j].add_patch(ellipse)

        # decorate
        axes[a_i, a_j].set_xlabel("$x$")
        axes[a_i, a_j].set_ylabel("$y$")
        axes[a_i, a_j].set_title(f"Absolute error of ${var}(x,y)$")
        axes[a_i, a_j].set_aspect("equal")
'''