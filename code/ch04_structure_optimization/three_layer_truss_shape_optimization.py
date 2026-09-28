#!/usr/bin/env python3
"""
Shape optimization of a 2D cantilever truss with three horizontal layers.

Idea:
- Build a three-layer truss with horizontal chords, vertical posts, and
  alternating diagonals in the lower and upper strips.
- Clamp the three right-end joints.
- Apply a downward force at the lower left-end joint.
- Solve the linear truss equilibrium K(V) u = f.
- Minimize compliance J(V) = f^T u by moving intermediate joints.
- Keep the left and right end columns fixed in shape.
"""

import os
from pathlib import Path

import numpy as np


def build_three_layer_truss(n_panels=8, width=4.0, height=0.8):
    layer_y = np.array([0.0, 0.5 * height, height], dtype=float)
    nodes = []
    for layer in range(3):
        for ix in range(n_panels + 1):
            nodes.append([width * ix / n_panels, layer_y[layer]])
    V = np.array(nodes, dtype=float)

    def node(layer, ix):
        return layer * (n_panels + 1) + ix

    bars = []
    groups = []

    for layer in range(3):
        group = ("bottom_chord", "middle_chord", "top_chord")[layer]
        for ix in range(n_panels):
            bars.append((node(layer, ix), node(layer, ix + 1)))
            groups.append(group)

    for ix in range(n_panels + 1):
        bars.append((node(0, ix), node(1, ix)))
        groups.append("lower_vertical")
        bars.append((node(1, ix), node(2, ix)))
        groups.append("upper_vertical")

    for ix in range(n_panels):
        if ix % 2 == 0:
            bars.append((node(0, ix), node(1, ix + 1)))
            groups.append("lower_diagonal")
            bars.append((node(1, ix), node(2, ix + 1)))
            groups.append("upper_diagonal")
        else:
            bars.append((node(1, ix), node(0, ix + 1)))
            groups.append("lower_diagonal")
            bars.append((node(2, ix), node(1, ix + 1)))
            groups.append("upper_diagonal")

    return V, bars, groups, node


def bar_areas(
    groups,
    outer_chord_area=1.35,
    middle_chord_area=1.1,
    vertical_area=0.7,
    diagonal_area=1.0,
):
    areas = np.ones(len(groups), dtype=float)
    for e, group in enumerate(groups):
        if group in ("bottom_chord", "top_chord"):
            areas[e] = outer_chord_area
        elif group == "middle_chord":
            areas[e] = middle_chord_area
        elif "vertical" in group:
            areas[e] = vertical_area
        elif "diagonal" in group:
            areas[e] = diagonal_area
    return areas


def assemble_truss_stiffness(V, bars, areas, young_modulus=60.0):
    n_nodes = len(V)
    K = np.zeros((2 * n_nodes, 2 * n_nodes))

    for e, (i, j) in enumerate(bars):
        sx = V[j, 0] - V[i, 0]
        sy = V[j, 1] - V[i, 1]
        L2 = sx * sx + sy * sy
        if L2 < 1e-20:
            continue

        invL = 1.0 / np.sqrt(L2)
        nx = sx * invL
        ny = sy * invL
        ke = young_modulus * areas[e] * invL

        kxx = ke * nx * nx
        kxy = ke * nx * ny
        kyy = ke * ny * ny

        ii = 2 * i
        jj = 2 * j

        K[ii, ii] += kxx
        K[ii, ii + 1] += kxy
        K[ii + 1, ii] += kxy
        K[ii + 1, ii + 1] += kyy

        K[jj, jj] += kxx
        K[jj, jj + 1] += kxy
        K[jj + 1, jj] += kxy
        K[jj + 1, jj + 1] += kyy

        K[ii, jj] -= kxx
        K[ii, jj + 1] -= kxy
        K[ii + 1, jj] -= kxy
        K[ii + 1, jj + 1] -= kyy

        K[jj, ii] -= kxx
        K[jj, ii + 1] -= kxy
        K[jj + 1, ii] -= kxy
        K[jj + 1, ii + 1] -= kyy

    return K


def fixed_dof_array(fixed_nodes):
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([2 * node, 2 * node + 1])
    return np.array(sorted(fixed_dofs), dtype=int)


def solve_equilibrium(V, bars, areas, fixed_nodes, f, young_modulus=60.0, reg=1e-9):
    ndof = 2 * len(V)
    K = assemble_truss_stiffness(V, bars, areas, young_modulus=young_modulus)

    fixed_dofs = fixed_dof_array(fixed_nodes)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    Kff = K[np.ix_(free_dofs, free_dofs)] + reg * np.eye(len(free_dofs))
    ff = f[free_dofs]

    uf = np.linalg.solve(Kff, ff)
    u = np.zeros(ndof)
    u[free_dofs] = uf

    return u.reshape(len(V), 2), float(f @ u)


def compliance(V, bars, areas, fixed_nodes, f, young_modulus=60.0):
    _, J = solve_equilibrium(V, bars, areas, fixed_nodes, f, young_modulus=young_modulus)
    return J


def finite_difference_shape_gradient(
    V,
    bars,
    areas,
    movable_nodes,
    fixed_nodes,
    f,
    young_modulus=60.0,
    h=1e-5,
    move_x=True,
):
    grad = np.zeros_like(V)
    base = compliance(V, bars, areas, fixed_nodes, f, young_modulus=young_modulus)

    directions = (0, 1) if move_x else (1,)
    for i in movable_nodes:
        for a in directions:
            Vp = V.copy()
            Vm = V.copy()
            Vp[i, a] += h
            Vm[i, a] -= h
            Jp = compliance(Vp, bars, areas, fixed_nodes, f, young_modulus=young_modulus)
            Jm = compliance(Vm, bars, areas, fixed_nodes, f, young_modulus=young_modulus)
            grad[i, a] = (Jp - Jm) / (2.0 * h)

    return grad, base


def analytical_shape_gradient(V, bars, areas, u, movable_nodes, young_modulus=60.0, move_x=True):
    """Compliance shape gradient for standard 2D truss bar elements."""
    grad = np.zeros_like(V)

    for e, (i, j) in enumerate(bars):
        s = V[j] - V[i]
        q = u[j] - u[i]
        L2 = float(s @ s)
        if L2 < 1e-20:
            continue

        L = np.sqrt(L2)
        b = float(s @ q)
        edge_grad = young_modulus * areas[e] * (2.0 * b * q / (L**3) - 3.0 * b * b * s / (L**5))

        grad[i] += edge_grad
        grad[j] -= edge_grad

    movable_mask = np.zeros(len(V), dtype=bool)
    movable_mask[movable_nodes] = True
    grad[~movable_mask] = 0.0
    if not move_x:
        grad[:, 0] = 0.0
    return grad


def build_movable_neighbors(n_nodes, bars, movable_nodes):
    neighbors = [[] for _ in range(n_nodes)]
    movable_set = set(movable_nodes)
    for i, j in bars:
        if j in movable_set:
            neighbors[i].append(j)
        if i in movable_set:
            neighbors[j].append(i)
    return neighbors


def smooth_shape_gradient(grad, movable_nodes, alpha=0.2, n_smooth=2, movable_neighbors=None):
    g = grad.copy()
    if movable_neighbors is None:
        return g

    for _ in range(n_smooth):
        new_g = g.copy()
        for i in movable_nodes:
            neigh = movable_neighbors[i]
            if len(neigh) == 0:
                continue
            new_g[i] = (1.0 - alpha) * g[i] + alpha * np.mean(g[neigh], axis=0)
        g = new_g

    return g


def minimum_bar_length(V, bars):
    if len(bars) == 0:
        return np.inf
    return float(min(np.linalg.norm(V[j] - V[i]) for i, j in bars))


def project_shape(V, V0, fixed_shape_nodes, x_margin=0.2, y_bounds=(-0.6, 2.4)):
    V_projected = V.copy()
    V_projected[:, 0] = np.clip(V_projected[:, 0], V0[:, 0] - x_margin, V0[:, 0] + x_margin)
    V_projected[:, 1] = np.clip(V_projected[:, 1], y_bounds[0], y_bounds[1])
    V_projected[fixed_shape_nodes] = V0[fixed_shape_nodes]
    return V_projected


def step_shape_descent(V, grad, movable_nodes, step_size=1.0, max_step=0.12):
    Vnew = V.copy()
    for i in movable_nodes:
        step = -step_size * grad[i]
        norm = np.linalg.norm(step)
        if norm > max_step:
            step = step * (max_step / norm)
        Vnew[i] += step
    return Vnew


def backtracking_shape_step(
    V,
    V0,
    bars,
    areas,
    grad,
    movable_nodes,
    fixed_shape_nodes,
    fixed_nodes,
    f,
    current_J,
    initial_step_size,
    young_modulus=60.0,
    max_step=0.12,
    shrink=0.5,
    max_backtracking=30,
    min_bar_length=0.08,
    x_margin=0.2,
    y_bounds=(-0.6, 2.4),
):
    trial_step = initial_step_size

    for _ in range(max_backtracking):
        V_candidate = step_shape_descent(V, grad, movable_nodes, step_size=trial_step, max_step=max_step)
        V_candidate = project_shape(V_candidate, V0, fixed_shape_nodes, x_margin=x_margin, y_bounds=y_bounds)

        if min_bar_length is not None and minimum_bar_length(V_candidate, bars) < min_bar_length:
            trial_step *= shrink
            continue

        candidate_J = compliance(V_candidate, bars, areas, fixed_nodes, f, young_modulus=young_modulus)
        if np.isfinite(candidate_J) and candidate_J < current_J:
            return V_candidate, trial_step, True

        trial_step *= shrink

    return V.copy(), 0.0, False


def plot_truss(ax, V, bars, groups, areas, u=None, scale=1.0, title="", fixed_nodes=None, load_node=None):
    colors = {
        "bottom_chord": "#2563eb",
        "middle_chord": "#059669",
        "top_chord": "#2563eb",
        "lower_vertical": "#64748b",
        "upper_vertical": "#64748b",
        "lower_diagonal": "#dc2626",
        "upper_diagonal": "#ea580c",
    }
    max_area = max(float(np.max(areas)), 1e-12)

    for e, (i, j) in enumerate(bars):
        group = groups[e]
        linewidth = 1.0 + 1.5 * areas[e] / max_area
        ax.plot(
            [V[i, 0], V[j, 0]],
            [V[i, 1], V[j, 1]],
            color=colors.get(group, "black"),
            linewidth=linewidth,
            alpha=0.85,
        )

    ax.scatter(V[:, 0], V[:, 1], s=15, color="black", zorder=4)

    if fixed_nodes is not None:
        F = V[fixed_nodes]
        ax.scatter(F[:, 0], F[:, 1], s=46, marker="s", color="#111827", zorder=5)

    if load_node is not None:
        ax.scatter([V[load_node, 0]], [V[load_node, 1]], s=55, marker="o", color="#f59e0b", zorder=6)

    if u is not None:
        Vd = V + scale * u
        for i, j in bars:
            ax.plot(
                [Vd[i, 0], Vd[j, 0]],
                [Vd[i, 1], Vd[j, 1]],
                color="black",
                linewidth=0.9,
                linestyle="--",
                alpha=0.65,
            )
        ax.scatter(Vd[:, 0], Vd[:, 1], s=9, color="black", alpha=0.75, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)


def run_three_layer_truss_shape_optimization(
    n_panels=8,
    width=4.0,
    height=0.8,
    n_iter=100,
    step_size=2.0,
    young_modulus=60.0,
    load_magnitude=-1.0,
    smooth=True,
    gradient_method="analytic",
    line_search=True,
    line_search_growth=1.04,
    line_search_shrink=0.5,
    max_backtracking=30,
    max_step=0.12,
    min_bar_length=0.08,
    x_margin_fraction=0.4,
    y_bounds=(-0.6, 2.4),
    move_x=True,
):
    V0, bars, groups, node = build_three_layer_truss(n_panels=n_panels, width=width, height=height)
    V = V0.copy()
    areas = bar_areas(groups)

    fixed_nodes = [node(layer, n_panels) for layer in range(3)]
    fixed_shape_nodes = [node(layer, ix) for layer in range(3) for ix in (0, n_panels)]
    load_node = node(0, 0)

    f = np.zeros(2 * len(V))
    f[2 * load_node + 1] = load_magnitude

    movable_nodes = [i for i in range(len(V)) if i not in fixed_shape_nodes]
    movable_neighbors = build_movable_neighbors(len(V), bars, movable_nodes)

    panel_width = width / n_panels
    x_margin = x_margin_fraction * panel_width

    history = []
    snapshots = {}
    step_history = []
    accepted_history = []
    min_bar_history = []
    current_step_size = step_size

    for it in range(n_iter + 1):
        u, J = solve_equilibrium(V, bars, areas, fixed_nodes, f, young_modulus=young_modulus)
        history.append(J)
        min_bar_history.append(minimum_bar_length(V, bars))

        if it in [0, n_iter // 3, 2 * n_iter // 3, n_iter]:
            snapshots[it] = (V.copy(), u.copy(), J)

        if it == n_iter:
            break

        if gradient_method == "finite_difference":
            grad, _ = finite_difference_shape_gradient(
                V,
                bars,
                areas,
                movable_nodes,
                fixed_nodes,
                f,
                young_modulus=young_modulus,
                move_x=move_x,
            )
        elif gradient_method == "analytic":
            grad = analytical_shape_gradient(
                V,
                bars,
                areas,
                u,
                movable_nodes,
                young_modulus=young_modulus,
                move_x=move_x,
            )
        else:
            raise ValueError(f"unknown gradient_method: {gradient_method}")

        if smooth:
            grad = smooth_shape_gradient(grad, movable_nodes, movable_neighbors=movable_neighbors)

        if line_search:
            V, accepted_step, accepted = backtracking_shape_step(
                V,
                V0,
                bars,
                areas,
                grad,
                movable_nodes,
                fixed_shape_nodes,
                fixed_nodes,
                f,
                J,
                current_step_size,
                young_modulus=young_modulus,
                max_step=max_step,
                shrink=line_search_shrink,
                max_backtracking=max_backtracking,
                min_bar_length=min_bar_length,
                x_margin=x_margin,
                y_bounds=y_bounds,
            )
            step_history.append(accepted_step)
            accepted_history.append(accepted)
            if accepted:
                current_step_size = accepted_step * line_search_growth
            else:
                current_step_size = current_step_size * line_search_shrink
        else:
            V = step_shape_descent(V, grad, movable_nodes, step_size=current_step_size, max_step=max_step)
            V = project_shape(V, V0, fixed_shape_nodes, x_margin=x_margin, y_bounds=y_bounds)
            step_history.append(current_step_size)
            accepted_history.append(True)

    stats = {
        "areas": areas,
        "groups": np.array(groups),
        "step_history": np.array(step_history, dtype=float),
        "accepted_history": np.array(accepted_history, dtype=bool),
        "min_bar_history": np.array(min_bar_history, dtype=float),
        "gradient_method": gradient_method,
        "line_search": line_search,
        "min_bar_length": min_bar_length,
        "x_margin": x_margin,
        "y_bounds": np.array(y_bounds, dtype=float),
    }

    return V0, V, bars, groups, fixed_nodes, fixed_shape_nodes, load_node, f, history, snapshots, stats


def main(show=False):
    results_dir = Path("three_layer_truss_shape_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = results_dir / "matplotlib_cache"
    matplotlib_cache.mkdir(exist_ok=True)
    font_cache = results_dir / "fontconfig_cache"
    font_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))

    import matplotlib.pyplot as plt

    (
        V0,
        Vopt,
        bars,
        groups,
        fixed_nodes,
        fixed_shape_nodes,
        load_node,
        f,
        history,
        snapshots,
        stats,
    ) = run_three_layer_truss_shape_optimization()
    history = np.array(history, dtype=float)

    print("Initial compliance:", history[0])
    print("Final compliance:  ", history[-1])
    print("Relative reduction:", (history[0] - history[-1]) / history[0])
    print("Final minimum bar length:", stats["min_bar_history"][-1])
    print("Results written to:", results_dir.resolve())

    step_to_next = np.full(len(history), np.nan)
    accepted_to_next = np.zeros(len(history))
    step_to_next[:-1] = stats["step_history"]
    accepted_to_next[:-1] = stats["accepted_history"].astype(float)
    csv_data = np.column_stack(
        [
            np.arange(len(history)),
            history,
            (history[0] - history) / history[0],
            stats["min_bar_history"],
            step_to_next,
            accepted_to_next,
        ]
    )
    np.savetxt(
        results_dir / "three_layer_truss_compliance_history.csv",
        csv_data,
        delimiter=",",
        header="iteration,compliance,relative_reduction,min_bar_length,step_to_next,accepted_to_next",
        comments="",
    )
    np.savez(
        results_dir / "three_layer_truss_shape_data.npz",
        V0=V0,
        Vopt=Vopt,
        bars=np.array(bars, dtype=int),
        groups=np.array(groups),
        areas=stats["areas"],
        fixed_nodes=np.array(fixed_nodes, dtype=int),
        fixed_shape_nodes=np.array(fixed_shape_nodes, dtype=int),
        load_node=np.array(load_node, dtype=int),
        f=f,
        history=history,
        step_history=stats["step_history"],
        accepted_history=stats["accepted_history"],
        min_bar_history=stats["min_bar_history"],
    )

    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o", markersize=3)
    plt.xlabel("shape iteration")
    plt.ylabel("compliance $J=f^T u$")
    plt.title("Three-layer truss compliance reduction")
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "three_layer_truss_compliance_history.png", dpi=200)

    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    axs = axs.ravel()
    for ax, (it, (V, u, J)) in zip(axs, snapshots.items()):
        plot_truss(
            ax,
            V,
            bars,
            groups,
            stats["areas"],
            u=u,
            scale=0.25,
            title=f"iteration {it}, J={J:.4f}",
            fixed_nodes=fixed_nodes,
            load_node=load_node,
        )
    plt.tight_layout()
    plt.savefig(results_dir / "three_layer_truss_shape_snapshots.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
