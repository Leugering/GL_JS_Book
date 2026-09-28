#!/usr/bin/env python3
"""
Shape optimization of a 2D cantilever nonlinear mass-spring network.

Idea:
- Build a rectangular grid of nodes.
- Connect horizontal, vertical, and diagonal springs.
- Clamp the right boundary.
- Apply a downward force at the left-middle node.
- Solve a geometrically nonlinear central-force spring equilibrium.
- Minimize compliance J(V) = f^T u by moving internal node positions.
- Boundary nodes remain fixed.

This is intentionally written as a clear research prototype rather than a
high-performance FEM code.
"""

import os
from pathlib import Path

import numpy as np


def build_grid(nx=9, ny=5, width=4.0, height=1.6, add_diagonals=True):
    nodes = []
    for ix in range(nx):
        for iy in range(ny):
            x = width * ix / (nx - 1)
            y = height * iy / (ny - 1)
            nodes.append([x, y])
    V = np.array(nodes, dtype=float)

    def idx(ix, iy):
        return ix * ny + iy

    edges = []
    for ix in range(nx):
        for iy in range(ny):
            if ix + 1 < nx:
                edges.append((idx(ix, iy), idx(ix + 1, iy)))
            if iy + 1 < ny:
                edges.append((idx(ix, iy), idx(ix, iy + 1)))

    if add_diagonals:
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                edges.append((idx(ix, iy), idx(ix + 1, iy + 1)))
                edges.append((idx(ix + 1, iy), idx(ix, iy + 1)))

    return V, edges, idx


def assemble_stiffness(V, edges, k0=30.0, length_scaling=True):
    n_nodes = len(V)
    K = np.zeros((2 * n_nodes, 2 * n_nodes))

    for i, j in edges:
        sx = V[j, 0] - V[i, 0]
        sy = V[j, 1] - V[i, 1]
        L2 = sx * sx + sy * sy
        if L2 < 1e-20:
            continue

        invL = 1.0 / np.sqrt(L2)
        nx = sx * invL
        ny = sy * invL
        ke = k0 * invL if length_scaling else k0

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


def solve_equilibrium(V, edges, fixed_nodes, f, k0=30.0, reg=1e-8):
    n_nodes = len(V)
    ndof = 2 * n_nodes

    K = assemble_stiffness(V, edges, k0=k0)

    fixed_dofs = []
    for i in fixed_nodes:
        fixed_dofs.extend([2 * i, 2 * i + 1])
    fixed_dofs = np.array(sorted(fixed_dofs), dtype=int)

    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    Kff = K[np.ix_(free_dofs, free_dofs)]
    ff = f[free_dofs]

    Kff = Kff + reg * np.eye(Kff.shape[0])

    uf = np.linalg.solve(Kff, ff)
    u = np.zeros(ndof)
    u[free_dofs] = uf

    compliance = float(f @ u)
    return u.reshape(n_nodes, 2), compliance



# -----------------------------
# Nonlinear static spring model
# -----------------------------

def nonlinear_internal_force_and_tangent(U_flat, V, edges, fixed_nodes, k0=30.0, length_scaling=True, reg=1e-8):
    """
    Internal force and tangent stiffness for geometrically nonlinear axial springs.

    Reference geometry: V.
    Displacement unknown: U_flat.
    Current positions: q = V + u.

    Each edge has rest length L0 = |V_j - V_i| and force

        f_e = k_e (L - L0) n,

    where L = |q_j - q_i| and n = (q_j-q_i)/L.

    The tangent contribution is

        k_e [ n n^T + (L-L0)/L (I - n n^T) ].

    This is the standard tangent for a central-force spring.
    """
    n_nodes = len(V)
    U = U_flat.reshape(n_nodes, 2)
    q = V + U

    fint = np.zeros(2 * n_nodes)
    Kt = np.zeros((2 * n_nodes, 2 * n_nodes))
    for i, j in edges:
        sx0 = V[j, 0] - V[i, 0]
        sy0 = V[j, 1] - V[i, 1]
        L02 = sx0 * sx0 + sy0 * sy0
        if L02 < 1e-24:
            continue
        L0 = np.sqrt(L02)

        rx = q[j, 0] - q[i, 0]
        ry = q[j, 1] - q[i, 1]
        L2 = rx * rx + ry * ry
        if L2 < 1e-24:
            continue
        L = np.sqrt(L2)

        nx = rx / L
        ny = ry / L
        ke = k0 / L0 if length_scaling else k0
        stretch = L - L0
        fx = ke * stretch * nx
        fy = ke * stretch * ny

        ix = 2 * i
        iy = ix + 1
        jx = 2 * j
        jy = jx + 1

        # Internal nodal forces: node i gets -fe, node j gets +fe.
        fint[ix] -= fx
        fint[iy] -= fy
        fint[jx] += fx
        fint[jy] += fy

        alpha = stretch / L
        nxx = nx * nx
        nxy = nx * ny
        nyy = ny * ny
        kxx = ke * (nxx + alpha * (1.0 - nxx))
        kxy = ke * (nxy - alpha * nxy)
        kyy = ke * (nyy + alpha * (1.0 - nyy))

        Kt[ix, ix] += kxx
        Kt[ix, iy] += kxy
        Kt[iy, ix] += kxy
        Kt[iy, iy] += kyy

        Kt[jx, jx] += kxx
        Kt[jx, jy] += kxy
        Kt[jy, jx] += kxy
        Kt[jy, jy] += kyy

        Kt[ix, jx] -= kxx
        Kt[ix, jy] -= kxy
        Kt[iy, jx] -= kxy
        Kt[iy, jy] -= kyy

        Kt[jx, ix] -= kxx
        Kt[jx, iy] -= kxy
        Kt[jy, ix] -= kxy
        Kt[jy, iy] -= kyy

    # A tiny diagonal regularization improves robustness near mechanisms.
    if reg > 0:
        Kt = Kt + reg * np.eye(2 * n_nodes)

    return fint, Kt


def solve_nonlinear_equilibrium(
    V,
    edges,
    fixed_nodes,
    f,
    k0=30.0,
    length_scaling=True,
    reg=1e-8,
    tol=1e-9,
    max_newton=30,
    line_search=True,
):
    """
    Solve nonlinear static spring equilibrium

        f_int(V+u) = f,

    with u=0 on fixed nodes.

    Returns displacement u, compliance f^T u, convergence flag and iteration count.
    """
    n_nodes = len(V)
    ndof = 2 * n_nodes
    U = np.zeros(ndof)

    fixed_dofs = []
    for i in fixed_nodes:
        fixed_dofs.extend([2 * i, 2 * i + 1])
    fixed_dofs = np.array(sorted(fixed_dofs), dtype=int)
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    converged = False
    for it in range(max_newton):
        fint, Kt = nonlinear_internal_force_and_tangent(
            U, V, edges, fixed_nodes, k0=k0, length_scaling=length_scaling, reg=reg
        )
        residual = fint - f
        r_free = residual[free_dofs]
        r_norm = np.linalg.norm(r_free)
        if r_norm < tol:
            converged = True
            break

        Kff = Kt[np.ix_(free_dofs, free_dofs)]
        try:
            delta_free = np.linalg.solve(Kff, -r_free)
        except np.linalg.LinAlgError:
            Kff = Kff + 1e-6 * np.eye(Kff.shape[0])
            delta_free = np.linalg.solve(Kff, -r_free)

        step = 1.0
        if line_search:
            # Backtrack until residual decreases.
            for _ in range(20):
                U_trial = U.copy()
                U_trial[free_dofs] += step * delta_free
                fint_trial, _ = nonlinear_internal_force_and_tangent(
                    U_trial, V, edges, fixed_nodes, k0=k0, length_scaling=length_scaling, reg=reg
                )
                trial_norm = np.linalg.norm((fint_trial - f)[free_dofs])
                if np.isfinite(trial_norm) and trial_norm <= (1.0 - 1e-4 * step) * r_norm:
                    break
                step *= 0.5

        U[free_dofs] += step * delta_free
        U[fixed_dofs] = 0.0

    compliance_value = float(f @ U)
    return U.reshape(n_nodes, 2), compliance_value, converged, it + 1

def compliance(V, edges, fixed_nodes, f, k0=30.0, model="linear_truss"):
    if model == "linear_truss":
        _, J = solve_equilibrium(V, edges, fixed_nodes, f, k0=k0)
        return J
    if model == "nonlinear_spring":
        _, J, converged, _ = solve_nonlinear_equilibrium(V, edges, fixed_nodes, f, k0=k0)
        if not converged:
            # Penalize failed nonlinear solves in line-search/finite-difference loops.
            return np.inf
        return J
    raise ValueError(f"unknown model: {model}")


def finite_difference_shape_gradient(V, edges, movable_nodes, fixed_nodes, f, k0=30.0, h=1e-5, model="linear_truss"):
    grad = np.zeros_like(V)
    base = compliance(V, edges, fixed_nodes, f, k0=k0, model=model)

    for i in movable_nodes:
        for a in range(2):
            Vp = V.copy()
            Vm = V.copy()
            Vp[i, a] += h
            Vm[i, a] -= h

            Jp = compliance(Vp, edges, fixed_nodes, f, k0=k0, model=model)
            Jm = compliance(Vm, edges, fixed_nodes, f, k0=k0, model=model)

            grad[i, a] = (Jp - Jm) / (2.0 * h)

    return grad, base


def free_and_fixed_dofs(n_nodes, fixed_nodes):
    fixed_dofs = []
    for i in fixed_nodes:
        fixed_dofs.extend([2 * i, 2 * i + 1])
    fixed_dofs = np.array(sorted(fixed_dofs), dtype=int)
    free_dofs = np.setdiff1d(np.arange(2 * n_nodes), fixed_dofs)
    return free_dofs, fixed_dofs


def nonlinear_adjoint_fd_shape_gradient(
    V,
    edges,
    u,
    movable_nodes,
    fixed_nodes,
    f,
    k0=30.0,
    h=1e-5,
    length_scaling=True,
    reg=1e-8,
):
    """
    Shape gradient for nonlinear springs using an adjoint and residual finite differences.

    The nonlinear equilibrium residual is R(u,V)=f_int(u,V)-f.  Since
    J(V)=f^T u(V), the adjoint lambda solves K_T^T lambda=f and

        dJ/dV = -lambda^T R_V.

    R_V is evaluated by centered finite differences at fixed displacement u.
    This avoids solving a full nonlinear equilibrium problem for every shape
    coordinate, unlike finite_difference_shape_gradient(..., model="nonlinear_spring").
    """
    n_nodes = len(V)
    U_flat = u.reshape(2 * n_nodes)
    _, Kt = nonlinear_internal_force_and_tangent(
        U_flat, V, edges, fixed_nodes, k0=k0, length_scaling=length_scaling, reg=reg
    )
    free_dofs, fixed_dofs = free_and_fixed_dofs(n_nodes, fixed_nodes)

    lam = np.zeros(2 * n_nodes)
    Kff = Kt[np.ix_(free_dofs, free_dofs)]
    try:
        lam[free_dofs] = np.linalg.solve(Kff.T, f[free_dofs])
    except np.linalg.LinAlgError:
        Kff = Kff + 1e-6 * np.eye(Kff.shape[0])
        lam[free_dofs] = np.linalg.solve(Kff.T, f[free_dofs])
    lam[fixed_dofs] = 0.0

    grad = np.zeros_like(V)
    for i in movable_nodes:
        for a in range(2):
            Vp = V.copy()
            Vm = V.copy()
            Vp[i, a] += h
            Vm[i, a] -= h

            rp, _ = nonlinear_internal_force_and_tangent(
                U_flat, Vp, edges, fixed_nodes, k0=k0, length_scaling=length_scaling, reg=0.0
            )
            rm, _ = nonlinear_internal_force_and_tangent(
                U_flat, Vm, edges, fixed_nodes, k0=k0, length_scaling=length_scaling, reg=0.0
            )
            dR = (rp - rm) / (2.0 * h)
            grad[i, a] = -float(lam @ dR)

    return grad


def analytical_shape_gradient(V, edges, u, movable_nodes, k0=30.0, length_scaling=True):
    """Compliance shape gradient d(f^T u)/dV for the current displacement u."""
    grad = np.zeros_like(V)

    for i, j in edges:
        s = V[j] - V[i]
        q = u[j] - u[i]
        L2 = float(s @ s)
        if L2 < 1e-20:
            continue

        b = float(s @ q)
        if length_scaling:
            L = np.sqrt(L2)
            edge_grad = k0 * (2.0 * b * q / (L**3) - 3.0 * b * b * s / (L**5))
        else:
            edge_grad = k0 * (2.0 * b * q / L2 - 2.0 * b * b * s / (L2**2))

        grad[i] += edge_grad
        grad[j] -= edge_grad

    movable_mask = np.zeros(len(V), dtype=bool)
    movable_mask[movable_nodes] = True
    grad[~movable_mask] = 0.0
    return grad


def build_movable_neighbors(n_nodes, edges, movable_nodes):
    neighbors = [[] for _ in range(n_nodes)]
    movable_set = set(movable_nodes)
    for i, j in edges:
        if j in movable_set:
            neighbors[i].append(j)
        if i in movable_set:
            neighbors[j].append(i)
    return neighbors


def smooth_shape_gradient(V, edges, grad, movable_nodes, alpha=0.25, n_smooth=3, movable_neighbors=None):
    n_nodes = len(V)
    if movable_neighbors is None:
        movable_neighbors = build_movable_neighbors(n_nodes, edges, movable_nodes)

    g = grad.copy()

    for _ in range(n_smooth):
        new_g = g.copy()
        for i in movable_nodes:
            neigh = movable_neighbors[i]
            if len(neigh) == 0:
                continue
            avg = np.mean(g[neigh], axis=0)
            new_g[i] = (1 - alpha) * g[i] + alpha * avg
        g = new_g

    return g


def enforce_no_boundary_motion(V, V0, boundary_nodes):
    V[boundary_nodes] = V0[boundary_nodes]
    return V


def step_shape_descent(V, grad, movable_nodes, step_size=0.015, max_step=0.04):
    Vnew = V.copy()
    for i in movable_nodes:
        step = -step_size * grad[i]
        norm = np.linalg.norm(step)
        if norm > max_step:
            step = step * (max_step / norm)
        Vnew[i] += step
    return Vnew


def minimum_edge_length(V, edges):
    if len(edges) == 0:
        return np.inf
    return float(min(np.linalg.norm(V[j] - V[i]) for i, j in edges))


def backtracking_shape_step(
    V,
    V0,
    edges,
    grad,
    movable_nodes,
    boundary_nodes,
    fixed_nodes,
    f,
    current_J,
    initial_step_size,
    k0=30.0,
    max_step=0.1,
    shrink=0.5,
    max_backtracking=25,
    min_edge_length=0.01,
    model="linear_truss",
):
    trial_step = initial_step_size

    for _ in range(max_backtracking):
        V_candidate = step_shape_descent(V, grad, movable_nodes, step_size=trial_step, max_step=max_step)
        V_candidate = enforce_no_boundary_motion(V_candidate, V0, boundary_nodes)

        if min_edge_length is not None and minimum_edge_length(V_candidate, edges) < min_edge_length:
            trial_step *= shrink
            continue

        candidate_J = compliance(V_candidate, edges, fixed_nodes, f, k0=k0, model=model)
        if np.isfinite(candidate_J) and candidate_J < current_J:
            return V_candidate, trial_step, True

        trial_step *= shrink

    return V.copy(), 0.0, False


def plot_network(ax, V, edges, u=None, scale=1.0, title="", fixed_nodes=None, load_node=None):
    for i, j in edges:
        ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], linewidth=0.8, alpha=0.45)

    ax.scatter(V[:, 0], V[:, 1], s=12, zorder=3)

    if fixed_nodes is not None:
        F = V[fixed_nodes]
        ax.scatter(F[:, 0], F[:, 1], s=30, marker="s", zorder=4)

    if load_node is not None:
        ax.scatter([V[load_node, 0]], [V[load_node, 1]], s=45, marker="o", zorder=5)

    if u is not None:
        Vd = V + scale * u
        for i, j in edges:
            ax.plot([Vd[i, 0], Vd[j, 0]], [Vd[i, 1], Vd[j, 1]], linewidth=1.2, linestyle="--", alpha=0.8)
        ax.scatter(Vd[:, 0], Vd[:, 1], s=10, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)


def run_shape_optimization(
    nx=9,
    ny=5,
    n_iter=40,
    step_size=0.5,
    k0=30.0,
    load_magnitude=-1.0,
    fd_h=2e-5,
    smooth=True,
    gradient_method="adjoint_fd",
    line_search=True,
    line_search_growth=1.05,
    line_search_shrink=0.5,
    max_backtracking=25,
    max_step=0.15,
    min_edge_length=0.05,
    model="nonlinear_spring",
):
    V0, edges, idx = build_grid(nx=nx, ny=ny, width=4.0, height=1.6, add_diagonals=True)
    V = V0.copy()

    n_nodes = len(V)
    ndof = 2 * n_nodes

    right_ix = nx - 1
    fixed_nodes = [idx(right_ix, iy) for iy in range(ny)]

    left_ix = 0
    load_iy = ny // 2
    load_node = idx(left_ix, load_iy)

    f = np.zeros(ndof)
    f[2 * load_node + 1] = load_magnitude

    boundary_nodes = set()
    for ix in range(nx):
        boundary_nodes.add(idx(ix, 0))
        boundary_nodes.add(idx(ix, ny - 1))
    for iy in range(ny):
        boundary_nodes.add(idx(0, iy))
        boundary_nodes.add(idx(nx - 1, iy))
    boundary_nodes = sorted(boundary_nodes)

    movable_nodes = [i for i in range(n_nodes) if i not in boundary_nodes and i not in fixed_nodes]
    movable_neighbors = build_movable_neighbors(n_nodes, edges, movable_nodes)

    history = []
    snapshots = {}
    step_history = []
    accepted_history = []
    min_edge_history = []
    current_step_size = step_size

    for it in range(n_iter + 1):
        if model == "linear_truss":
            u, J = solve_equilibrium(V, edges, fixed_nodes, f, k0=k0)
        elif model == "nonlinear_spring":
            u, J, converged, newton_its = solve_nonlinear_equilibrium(V, edges, fixed_nodes, f, k0=k0)
            if not converged:
                print(f"Warning: nonlinear solve not fully converged at iteration {it} (Newton iterations: {newton_its})")
        else:
            raise ValueError(f"unknown model: {model}")
        history.append(J)
        min_edge_history.append(minimum_edge_length(V, edges))

        if it in [0, n_iter // 3, 2 * n_iter // 3, n_iter]:
            snapshots[it] = (V.copy(), u.copy(), J)

        if it == n_iter:
            break

        if gradient_method == "finite_difference":
            grad, _ = finite_difference_shape_gradient(V, edges, movable_nodes, fixed_nodes, f, k0=k0, h=fd_h, model=model)
        elif gradient_method == "adjoint_fd":
            if model != "nonlinear_spring":
                raise ValueError("adjoint_fd shape gradient is implemented for model='nonlinear_spring'.")
            grad = nonlinear_adjoint_fd_shape_gradient(V, edges, u, movable_nodes, fixed_nodes, f, k0=k0, h=fd_h)
        elif gradient_method == "analytic":
            if model != "linear_truss":
                raise ValueError("analytic shape gradient is implemented only for model='linear_truss'. Use gradient_method='adjoint_fd' for nonlinear_spring.")
            grad = analytical_shape_gradient(V, edges, u, movable_nodes, k0=k0)
        else:
            raise ValueError(f"unknown gradient_method: {gradient_method}")

        if smooth:
            grad = smooth_shape_gradient(V, edges, grad, movable_nodes, movable_neighbors=movable_neighbors)

        if line_search:
            V, accepted_step, accepted = backtracking_shape_step(
                V,
                V0,
                edges,
                grad,
                movable_nodes,
                boundary_nodes,
                fixed_nodes,
                f,
                J,
                current_step_size,
                k0=k0,
                max_step=max_step,
                shrink=line_search_shrink,
                max_backtracking=max_backtracking,
                min_edge_length=min_edge_length,
                model=model,
            )
            step_history.append(accepted_step)
            accepted_history.append(accepted)
            if accepted:
                current_step_size = accepted_step * line_search_growth
            else:
                current_step_size = current_step_size * line_search_shrink
        else:
            V = step_shape_descent(V, grad, movable_nodes, step_size=current_step_size, max_step=max_step)
            V = enforce_no_boundary_motion(V, V0, boundary_nodes)
            step_history.append(current_step_size)
            accepted_history.append(True)

    stats = {
        "step_history": np.array(step_history, dtype=float),
        "accepted_history": np.array(accepted_history, dtype=bool),
        "min_edge_history": np.array(min_edge_history, dtype=float),
        "gradient_method": gradient_method,
        "line_search": line_search,
        "model": model,
        "min_edge_length": min_edge_length,
    }

    return V0, V, edges, fixed_nodes, load_node, f, history, snapshots, stats


def main(show=False):
    results_dir = Path("cantilever_nonlinear_shape_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = results_dir / "matplotlib_cache"
    matplotlib_cache.mkdir(exist_ok=True)
    font_cache = results_dir / "fontconfig_cache"
    font_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))

    import matplotlib.pyplot as plt

    V0, Vopt, edges, fixed_nodes, load_node, f, history, snapshots, stats = run_shape_optimization()
    history = np.array(history, dtype=float)

    print("Model:", stats["model"])
    print("Initial compliance:", history[0])
    print("Final compliance:  ", history[-1])
    print("Relative reduction:", (history[0] - history[-1]) / history[0])
    print("Final minimum edge length:", stats["min_edge_history"][-1])
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
            stats["min_edge_history"],
            step_to_next,
            accepted_to_next,
        ]
    )
    np.savetxt(
        results_dir / "cantilever_compliance_history.csv",
        csv_data,
        delimiter=",",
        header="iteration,compliance,relative_reduction,min_edge_length,step_to_next,accepted_to_next",
        comments="",
    )
    np.savez(
        results_dir / "cantilever_shape_data.npz",
        V0=V0,
        Vopt=Vopt,
        edges=np.array(edges, dtype=int),
        fixed_nodes=np.array(fixed_nodes, dtype=int),
        load_node=np.array(load_node, dtype=int),
        f=f,
        history=history,
        step_history=stats["step_history"],
        accepted_history=stats["accepted_history"],
        min_edge_history=stats["min_edge_history"],
    )

    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o", markersize=3)
    plt.xlabel("shape iteration")
    plt.ylabel("compliance $J=f^T u$")
    plt.title("Compliance reduction")
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "cantilever_compliance_history.png", dpi=200)

    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    axs = axs.ravel()
    for ax, (it, (V, u, J)) in zip(axs, snapshots.items()):
        plot_network(ax, V, edges, u=u, scale=0.35, title=f"iteration {it}, J={J:.4f}", fixed_nodes=fixed_nodes, load_node=load_node)
    plt.tight_layout()
    plt.savefig(results_dir / "cantilever_shape_snapshots.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
