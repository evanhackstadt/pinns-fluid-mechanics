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


PALETTE_DEEP = sns.color_palette("deep").as_hex()
CMAP_VAR   = "rainbow"
CMAP_ERR   = "flare"
COLOR_PINN = PALETTE_DEEP[0]   # blue  – PINN prediction
COLOR_U    = PALETTE_DEEP[1]
COLOR_TRUE = PALETTE_DEEP[2]   # warm accent – analytical ground truth
COLOR_V    = PALETTE_DEEP[3]
COLOR_P    = PALETTE_DEEP[4]
COLOR_AGGREGATE = PALETTE_DEEP[5]
COLOR_PDE   = PALETTE_DEEP[6]
COLOR_BC    = PALETTE_DEEP[7]
COLOR_VARIABLE_MAP = {"u": COLOR_U, "v": COLOR_V, "p": COLOR_P}
LOSS_COLORS = {
    "PDE_continuity": PALETTE_DEEP[0],
    "PDE_x_momentum": PALETTE_DEEP[1],
    "PDE_y_momentum": PALETTE_DEEP[2],
    "BC_inlet_u": PALETTE_DEEP[3],
    "BC_inlet_v": PALETTE_DEEP[4],
    "BC_wall_u": PALETTE_DEEP[5],
    "BC_wall_v": PALETTE_DEEP[6],
    "BC_outlet_p": PALETTE_DEEP[7],
    "BC_observed_u": PALETTE_DEEP[8],
    "BC_observed_v": PALETTE_DEEP[9],
    "BC_observed_p": "#8A2BE2",
}
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
def save_errors(errors, output_dir, a, b, n):
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

        # use visually-distinct colors: https://mokole.com/palette.html
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
        plt.scatter(labeled_pts[:, 0], labeled_pts[:, 1], s=25, c=COLOR_TRUE)
        plt.title(f"Domain with n={labeled_pts.shape[0]} measurements (green)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save plot
    plt.savefig(output_dir / "domain.png", dpi=FIG_DPI)
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
            axes[i, j] = _plot_heatmap_single(axes[i, j], X, Y, values, CMAP_VAR, cfg, a, b,
                                              cbar_cap=cbar_cap,
                                              cbar_label=f"${var}$",
                                              title=f"{model} ${var}(x, y)$")
            
            # if saving plots separately
            if separate_plots:
                fig_sep, ax_sep = plt.subplots(figsize=(6, 3), dpi=FIG_DPI, constrained_layout=True)
                _plot_heatmap_single(ax_sep, X, Y, values, CMAP_VAR, cfg, a, b,
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





# ———————————— ACROSS-RUNS ANALYSIS ————————————

# --- Helper: Parse Error JSON ---
def _extract_error_summary(summary_path,
                           variables: list = ["u", "v", "p"],
                           metrics: list = ["L2", "L_inf", "MSE"],
                           aggregate_metrics: list = ["mean_L2", "max_L_inf"]):
    """
    Extract the entire summary.json into a tidy DataFrame without filtering or averaging.
    Each row represents one variable/metric or aggregate-metric value for a single run.
    Returns a DataFrame with columns [ab, a, b, n, variable, metric, value].
    """
    summary_path = Path(summary_path)
    with summary_path.open() as f:
        errors = json.load(f)

    rows = []
    for data in errors.values():
        a = float(data["parameters"]["a"])
        b = float(data["parameters"]["b"])
        n = int(data["parameters"]["n"])

        for var in variables:
            for met in metrics:
                rows.append({
                    "ab": f'({a}, {b})',
                    "a": a,
                    "b": b,
                    "n": n,
                    "variable": var,
                    "metric": met,
                    "value": float(data[var][met]),
                })
        for aggmetric in aggregate_metrics:
            rows.append({
                "ab": f'({a}, {b})',
                "a": a,
                "b": b,
                "n": n,
                "variable": "aggregate",
                "metric": aggmetric,
                "value": float(data["aggregate"][aggmetric]),
            })

    if len(rows) == 0:
        raise ValueError("No error data was found in the summary file.")

    return pd.DataFrame(rows).sort_values(by=["a", "b", "n", "variable", "metric"], ignore_index=True)


# --- Error Comparison Point Plots ---
def plot_error_comparison(summary_path, output_dir, parameter, 
                          fixed_ab: list = None, fixed_n = None):
    """
    Compare error across all runs, with a specified parameter as the axis. The free parameter can be fixed or averaged.
    Args:
        summary_path: path to summary.json containing errors across runs
        output_dir: path to folder to save plots
        parameter: string specifying the parameter of interest, choices = ["n", "a", "b", "ab"]
        fixed_ab: specified list of [a,b] to use across n; discards other geometries. If None, takes average errors across all (a,b). Requires variable="n".
        fixed_n:  specified value of n to use across (a,b); discards other n. If None, takes average errors across all n. Requires variable!="n".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse args
    parameter_choices = ["n", "a", "b", "ab"]
    if parameter not in parameter_choices:
        raise ValueError(f"Received parameter = {parameter}, but must be one of {parameter_choices}")
    if fixed_ab is not None and len(fixed_ab) != 2:
        raise ValueError(f"fixed_ab must be a list of length 2; received {fixed_ab}.")
    if parameter == "n":
        if fixed_n is not None:
            raise ValueError(f"fixed_n is not compatible with parameter={parameter}.")
    if parameter != "n":
        if fixed_ab is not None:
            raise ValueError(f"fixed_ab is not compatible with parameter={parameter}.")
    
    # Error Summary Dict Keys
    VARS = ["u", "v", "p"]
    METRICS = ["L2", "L_inf"]
    AGGREGATE_METRICS = ["mean_L2", "max_L_inf"]
    
    PARAMETER_LABELS = {
        "n": "number of labeled training points",
        "a": "ellipse width (a)",
        "b": "ellipse height (b)",
        "ab": "ellipse geometry (a, b)"
    }

    error_df = _extract_error_summary(summary_path, VARS, METRICS, AGGREGATE_METRICS)

    # filter to specified value of free parameter, if applicable
    if fixed_ab is not None:
        error_df = error_df[(error_df["a"].astype(float) == float(fixed_ab[0])) & (error_df["b"].astype(float) == float(fixed_ab[1]))]
    if fixed_n is not None:
        error_df = error_df[error_df["n"].astype(int) == int(fixed_n)]

    if error_df.empty:
        raise ValueError("No matching runs were found for the requested summary selection.")

    # index by parameter of interest (axis)
    if parameter == "n":
        error_df["parameter_value"] = error_df["n"]
        averaging = fixed_ab is None
    else:
        error_df["parameter_value"] = error_df["a"]
        if parameter == "b":
            error_df["parameter_value"] = error_df["b"]
        elif parameter == "ab":
            error_df["parameter_value"] = error_df.apply(lambda row: f"({row['a']}, {row['b']})", axis=1)
        averaging = fixed_n is None

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
        plt.xlabel(PARAMETER_LABELS[parameter])
        plt.ylabel(metric)
        
        # label based on args
        title = f"{metric} error across {parameter}"
        fname = f"errors_by_{parameter}_{metric}"
        if fixed_n:
            title += f", (where n={fixed_n})"
            fname += f"_n{fixed_n}"
        elif fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
            fname += f"_a{fixed_ab[0]}_b{fixed_ab[1]}"
        else:
            averaged_across = "ab" if parameter == "n" else "n"
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
        plt.xlabel(PARAMETER_LABELS[parameter])
        plt.ylabel(aggmetric)
        
        # label based on args
        title = f"{aggmetric} error of all outputs, across {parameter}"
        fname = f"errors_by_{parameter}_{aggmetric}"
        if fixed_n:
            title += f", (where n={fixed_n})"
            fname += f"_n{fixed_n}"
        elif fixed_ab:
            title += f" (where a={fixed_ab[0]}, b={fixed_ab[1]})"
            fname += f"_{aggmetric}_a{fixed_ab[0]}_b{fixed_ab[1]}"
        else:
            averaged_across = "ab" if parameter == "n" else "n"
            title += f", averaged across {averaged_across}"
        fname += ".png"
            
        plt.title(title)
        plt.tight_layout()
        
        savepath = output_dir / fname
        ax.figure.savefig(savepath, dpi=FIG_DPI)
        plt.close(ax.figure)
    

# --- Error Comparison Heatmaps ---
def plot_error_comparison_2d(summary_path, output_dir, index_parameter="n", col_parameter="ab"):
    """
    Create 2D grid heatmaps of errors for all combinations of two parameters.
    Args:
        summary_path: path to summary.json containing errors across runs
        output_dir: path to folder to save plots
        parameter_1: string specifying the parameter to plot on the x-axis, choices = ["n", "a", "b", "ab"]
        parameter_2: string specifying the parameter of plot on the y-axis, choices = ["n", "a", "b", "ab"]
    """
    
    # parameter = parameter x
    # select with parameter y fixed at each unique value of parameter y
    # pivot into 2D array
    # seaborn heatmap — each cell is a run (unique combo of n and (a,b))
    # plot different metrics:
    #   L2 error for each variable (3 plots)
    #   Linf error for each variable (3 plots)
    
    VARS = ["u", "v", "p"]
    METRICS = ["L2", "L_inf"]
    AGGREGATE_METRICS = ["mean_L2", "max_L_inf"]
    
    PARAMETER_LABELS = {
        "n": "number of labeled training points",
        "a": "ellipse width (a)",
        "b": "ellipse height (b)",
        "ab": "ellipse geometry (a, b)"
    }
    
    # cols are a, b, n, variable, metric, value
    error_df = _extract_error_summary(summary_path, VARS, METRICS, AGGREGATE_METRICS)
    
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
    