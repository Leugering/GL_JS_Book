#!/usr/bin/env python3
"""
Shape optimization of a 2D three-layer cantilever network of Timoshenko beams.

Idea:
- Use the same three-layer graph as the truss example.
- Replace each axial truss bar by a planar Timoshenko frame element.
- Each node has three degrees of freedom: horizontal displacement, vertical
  displacement, and cross-section rotation.
- Clamp the three right-end joints.
- Apply a downward force at the lower left-end joint.
- Minimize compliance J(V) = f^T u by moving intermediate joints.
"""

import os
from pathlib import Path

import numpy as np


def build_three_layer_network(n_panels=8, width=4.0, height=0.8):
    layer_y = np.array([0.0, 0.5 * height, height], dtype=float)
    nodes = []
    for layer in range(3):
        for ix in range(n_panels + 1):
            nodes.append([width * ix / n_panels, layer_y[layer]])
    V = np.array(nodes, dtype=float)

    def node(layer, ix):
        return layer * (n_panels + 1) + ix

    beams = []
    groups = []

    for layer in range(3):
        group = ("bottom_chord", "middle_chord", "top_chord")[layer]
        for ix in range(n_panels):
            beams.append((node(layer, ix), node(layer, ix + 1)))
            groups.append(group)

    for ix in range(n_panels + 1):
        beams.append((node(0, ix), node(1, ix)))
        groups.append("lower_vertical")
        beams.append((node(1, ix), node(2, ix)))
        groups.append("upper_vertical")

    for ix in range(n_panels):
        if ix % 2 == 0:
            beams.append((node(0, ix), node(1, ix + 1)))
            groups.append("lower_diagonal")
            beams.append((node(1, ix), node(2, ix + 1)))
            groups.append("upper_diagonal")
        else:
            beams.append((node(1, ix), node(0, ix + 1)))
            groups.append("lower_diagonal")
            beams.append((node(2, ix), node(1, ix + 1)))
            groups.append("upper_diagonal")

    return V, beams, groups, node


def section_properties(
    groups,
    outer_area=1.35,
    middle_area=1.10,
    vertical_area=0.70,
    diagonal_area=1.00,
    outer_inertia=0.018,
    middle_inertia=0.014,
    vertical_inertia=0.007,
    diagonal_inertia=0.010,
):
    areas = np.ones(len(groups), dtype=float)
    inertias = np.ones(len(groups), dtype=float)

    for e, group in enumerate(groups):
        if group in ("bottom_chord", "top_chord"):
            areas[e] = outer_area
            inertias[e] = outer_inertia
        elif group == "middle_chord":
            areas[e] = middle_area
            inertias[e] = middle_inertia
        elif "vertical" in group:
            areas[e] = vertical_area
            inertias[e] = vertical_inertia
        elif "diagonal" in group:
            areas[e] = diagonal_area
            inertias[e] = diagonal_inertia

    return areas, inertias


def local_timoshenko_stiffness(length, area, inertia, young_modulus, shear_modulus, shear_factor):
    L = length
    EA = young_modulus * area
    EI = young_modulus * inertia
    kGA = shear_factor * shear_modulus * area

    phi = 12.0 * EI / (kGA * L * L)
    psi = 1.0 / (1.0 + phi)

    axial = EA / L
    bending = 12.0 * EI * psi / (L**3)
    shear_rotation = 6.0 * EI * psi / (L**2)
    rot_diag = (4.0 + phi) * EI * psi / L
    rot_offdiag = (2.0 - phi) * EI * psi / L

    k = np.zeros((6, 6))

    k[0, 0] = axial
    k[0, 3] = -axial
    k[3, 0] = -axial
    k[3, 3] = axial

    k[1, 1] = bending
    k[1, 2] = shear_rotation
    k[1, 4] = -bending
    k[1, 5] = shear_rotation

    k[2, 1] = shear_rotation
    k[2, 2] = rot_diag
    k[2, 4] = -shear_rotation
    k[2, 5] = rot_offdiag

    k[4, 1] = -bending
    k[4, 2] = -shear_rotation
    k[4, 4] = bending
    k[4, 5] = -shear_rotation

    k[5, 1] = shear_rotation
    k[5, 2] = rot_offdiag
    k[5, 4] = -shear_rotation
    k[5, 5] = rot_diag

    return k


def transformation_matrix(c, s):
    T = np.zeros((6, 6))
    T[0, 0] = c
    T[0, 1] = s
    T[1, 0] = -s
    T[1, 1] = c
    T[2, 2] = 1.0
    T[3, 3] = c
    T[3, 4] = s
    T[4, 3] = -s
    T[4, 4] = c
    T[5, 5] = 1.0
    return T


def assemble_timoshenko_stiffness(
    V,
    beams,
    areas,
    inertias,
    young_modulus=60.0,
    shear_modulus=25.0,
    shear_factor=5.0 / 6.0,
):
    n_nodes = len(V)
    K = np.zeros((3 * n_nodes, 3 * n_nodes))

    for e, (i, j) in enumerate(beams):
        dx = V[j, 0] - V[i, 0]
        dy = V[j, 1] - V[i, 1]
        L2 = dx * dx + dy * dy
        if L2 < 1e-20:
            continue

        L = np.sqrt(L2)
        c = dx / L
        s = dy / L
        k_local = local_timoshenko_stiffness(
            L,
            areas[e],
            inertias[e],
            young_modulus,
            shear_modulus,
            shear_factor,
        )
        T = transformation_matrix(c, s)
        k_global = T.T @ k_local @ T

        dofs = np.array(
            [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2],
            dtype=int,
        )
        K[np.ix_(dofs, dofs)] += k_global

    return K


def fixed_dof_array(fixed_nodes):
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([3 * node, 3 * node + 1, 3 * node + 2])
    return np.array(sorted(fixed_dofs), dtype=int)


def solve_equilibrium(
    V,
    beams,
    areas,
    inertias,
    fixed_nodes,
    f,
    young_modulus=60.0,
    shear_modulus=25.0,
    shear_factor=5.0 / 6.0,
    reg=1e-9,
):
    ndof = 3 * len(V)
    K = assemble_timoshenko_stiffness(
        V,
        beams,
        areas,
        inertias,
        young_modulus=young_modulus,
        shear_modulus=shear_modulus,
        shear_factor=shear_factor,
    )

    fixed_dofs = fixed_dof_array(fixed_nodes)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    Kff = K[np.ix_(free_dofs, free_dofs)] + reg * np.eye(len(free_dofs))
    ff = f[free_dofs]

    uf = np.linalg.solve(Kff, ff)
    u = np.zeros(ndof)
    u[free_dofs] = uf

    return u.reshape(len(V), 3), float(f @ u)


def compliance(V, beams, areas, inertias, fixed_nodes, f, young_modulus=60.0, shear_modulus=25.0):
    _, J = solve_equilibrium(
        V,
        beams,
        areas,
        inertias,
        fixed_nodes,
        f,
        young_modulus=young_modulus,
        shear_modulus=shear_modulus,
    )
    return J


def finite_difference_shape_gradient(
    V,
    beams,
    areas,
    inertias,
    movable_nodes,
    fixed_nodes,
    f,
    young_modulus=60.0,
    shear_modulus=25.0,
    h=1e-5,
    move_x=True,
):
    grad = np.zeros_like(V)
    base = compliance(V, beams, areas, inertias, fixed_nodes, f, young_modulus, shear_modulus)

    directions = (0, 1) if move_x else (1,)
    for i in movable_nodes:
        for a in directions:
            Vp = V.copy()
            Vm = V.copy()
            Vp[i, a] += h
            Vm[i, a] -= h
            Jp = compliance(Vp, beams, areas, inertias, fixed_nodes, f, young_modulus, shear_modulus)
            Jm = compliance(Vm, beams, areas, inertias, fixed_nodes, f, young_modulus, shear_modulus)
            grad[i, a] = (Jp - Jm) / (2.0 * h)

    return grad, base


def build_movable_neighbors(n_nodes, beams, movable_nodes):
    neighbors = [[] for _ in range(n_nodes)]
    movable_set = set(movable_nodes)
    for i, j in beams:
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


def minimum_beam_length(V, beams):
    if len(beams) == 0:
        return np.inf
    return float(min(np.linalg.norm(V[j] - V[i]) for i, j in beams))


def project_shape(V, V0, fixed_shape_nodes, x_margin=0.2, y_bounds=(-0.6, 2.4)):
    V_projected = V.copy()
    V_projected[:, 0] = np.clip(V_projected[:, 0], V0[:, 0] - x_margin, V0[:, 0] + x_margin)
    V_projected[:, 1] = np.clip(V_projected[:, 1], y_bounds[0], y_bounds[1])
    V_projected[fixed_shape_nodes] = V0[fixed_shape_nodes]
    return V_projected


def step_shape_descent(V, grad, movable_nodes, step_size=1.0, max_step=0.10):
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
    beams,
    areas,
    inertias,
    grad,
    movable_nodes,
    fixed_shape_nodes,
    fixed_nodes,
    f,
    current_J,
    initial_step_size,
    young_modulus=60.0,
    shear_modulus=25.0,
    max_step=0.10,
    shrink=0.5,
    max_backtracking=24,
    min_beam_length=0.08,
    x_margin=0.2,
    y_bounds=(-0.6, 2.4),
):
    trial_step = initial_step_size

    for _ in range(max_backtracking):
        V_candidate = step_shape_descent(V, grad, movable_nodes, step_size=trial_step, max_step=max_step)
        V_candidate = project_shape(V_candidate, V0, fixed_shape_nodes, x_margin=x_margin, y_bounds=y_bounds)

        if min_beam_length is not None and minimum_beam_length(V_candidate, beams) < min_beam_length:
            trial_step *= shrink
            continue

        candidate_J = compliance(
            V_candidate,
            beams,
            areas,
            inertias,
            fixed_nodes,
            f,
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
        )
        if np.isfinite(candidate_J) and candidate_J < current_J:
            return V_candidate, trial_step, True

        trial_step *= shrink

    return V.copy(), 0.0, False


def plot_beam_network(ax, V, beams, groups, areas, u=None, scale=1.0, title="", fixed_nodes=None, load_node=None):
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

    for e, (i, j) in enumerate(beams):
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
        Vd = V + scale * u[:, :2]
        for i, j in beams:
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


def run_three_layer_timoshenko_shape_optimization(
    n_panels=8,
    width=4.0,
    height=0.8,
    n_iter=45,
    step_size=0.7,
    young_modulus=60.0,
    shear_modulus=25.0,
    load_magnitude=-1.0,
    smooth=True,
    line_search=True,
    line_search_growth=1.03,
    line_search_shrink=0.5,
    max_backtracking=24,
    max_step=0.10,
    min_beam_length=0.08,
    x_margin_fraction=0.4,
    y_bounds=(-0.6, 2.4),
    move_x=True,
):
    V0, beams, groups, node = build_three_layer_network(n_panels=n_panels, width=width, height=height)
    V = V0.copy()
    areas, inertias = section_properties(groups)

    fixed_nodes = [node(layer, n_panels) for layer in range(3)]
    fixed_shape_nodes = [node(layer, ix) for layer in range(3) for ix in (0, n_panels)]
    load_node = node(0, 0)

    f = np.zeros(3 * len(V))
    f[3 * load_node + 1] = load_magnitude

    movable_nodes = [i for i in range(len(V)) if i not in fixed_shape_nodes]
    movable_neighbors = build_movable_neighbors(len(V), beams, movable_nodes)

    panel_width = width / n_panels
    x_margin = x_margin_fraction * panel_width

    history = []
    snapshots = {}
    step_history = []
    accepted_history = []
    min_beam_history = []
    current_step_size = step_size

    for it in range(n_iter + 1):
        u, J = solve_equilibrium(
            V,
            beams,
            areas,
            inertias,
            fixed_nodes,
            f,
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
        )
        history.append(J)
        min_beam_history.append(minimum_beam_length(V, beams))

        if it in [0, n_iter // 3, 2 * n_iter // 3, n_iter]:
            snapshots[it] = (V.copy(), u.copy(), J)

        if it == n_iter:
            break

        grad, _ = finite_difference_shape_gradient(
            V,
            beams,
            areas,
            inertias,
            movable_nodes,
            fixed_nodes,
            f,
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
            move_x=move_x,
        )

        if smooth:
            grad = smooth_shape_gradient(grad, movable_nodes, movable_neighbors=movable_neighbors)

        if line_search:
            V, accepted_step, accepted = backtracking_shape_step(
                V,
                V0,
                beams,
                areas,
                inertias,
                grad,
                movable_nodes,
                fixed_shape_nodes,
                fixed_nodes,
                f,
                J,
                current_step_size,
                young_modulus=young_modulus,
                shear_modulus=shear_modulus,
                max_step=max_step,
                shrink=line_search_shrink,
                max_backtracking=max_backtracking,
                min_beam_length=min_beam_length,
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
        "inertias": inertias,
        "groups": np.array(groups),
        "step_history": np.array(step_history, dtype=float),
        "accepted_history": np.array(accepted_history, dtype=bool),
        "min_beam_history": np.array(min_beam_history, dtype=float),
        "x_margin": x_margin,
        "y_bounds": np.array(y_bounds, dtype=float),
        "young_modulus": young_modulus,
        "shear_modulus": shear_modulus,
    }

    return V0, V, beams, groups, fixed_nodes, fixed_shape_nodes, load_node, f, history, snapshots, stats


def main(show=False):
    results_dir = Path("three_layer_timoshenko_beam_shape_results")
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
        beams,
        groups,
        fixed_nodes,
        fixed_shape_nodes,
        load_node,
        f,
        history,
        snapshots,
        stats,
    ) = run_three_layer_timoshenko_shape_optimization()
    history = np.array(history, dtype=float)

    print("Initial compliance:", history[0])
    print("Final compliance:  ", history[-1])
    print("Relative reduction:", (history[0] - history[-1]) / history[0])
    print("Final minimum beam length:", stats["min_beam_history"][-1])
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
            stats["min_beam_history"],
            step_to_next,
            accepted_to_next,
        ]
    )
    np.savetxt(
        results_dir / "three_layer_timoshenko_compliance_history.csv",
        csv_data,
        delimiter=",",
        header="iteration,compliance,relative_reduction,min_beam_length,step_to_next,accepted_to_next",
        comments="",
    )
    np.savez(
        results_dir / "three_layer_timoshenko_shape_data.npz",
        V0=V0,
        Vopt=Vopt,
        beams=np.array(beams, dtype=int),
        groups=np.array(groups),
        areas=stats["areas"],
        inertias=stats["inertias"],
        fixed_nodes=np.array(fixed_nodes, dtype=int),
        fixed_shape_nodes=np.array(fixed_shape_nodes, dtype=int),
        load_node=np.array(load_node, dtype=int),
        f=f,
        history=history,
        step_history=stats["step_history"],
        accepted_history=stats["accepted_history"],
        min_beam_history=stats["min_beam_history"],
    )

    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o", markersize=3)
    plt.xlabel("shape iteration")
    plt.ylabel("compliance $J=f^T u$")
    plt.title("Three-layer Timoshenko beam compliance reduction")
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "three_layer_timoshenko_compliance_history.png", dpi=200)

    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    axs = axs.ravel()
    for ax, (it, (V, u, J)) in zip(axs, snapshots.items()):
        plot_beam_network(
            ax,
            V,
            beams,
            groups,
            stats["areas"],
            u=u,
            scale=0.25,
            title=f"iteration {it}, J={J:.4f}",
            fixed_nodes=fixed_nodes,
            load_node=load_node,
        )
    plt.tight_layout()
    plt.savefig(results_dir / "three_layer_timoshenko_shape_snapshots.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
