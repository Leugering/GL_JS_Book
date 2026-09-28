# Corrected copy (2026-09-28): insertion sensitivity now includes the bar length L_e,
# i.e. it is the exact derivative -p^T dK/ds q. Original: bridge_truss_hante_insertion_growth.py
import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_bridge_hante_growth")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results_bridge_hante_insertion_growth_exact")
os.makedirs(OUT_DIR, exist_ok=True)


def unit_bar_stiffness(xi, xj):
    d = xj - xi
    L = np.linalg.norm(d)
    t = d / max(L, 1e-12)
    k3 = np.outer(t, t) / max(L, 1e-12)
    ke = np.block([[k3, -k3], [-k3, k3]])
    return ke, L, t


def build_twin_arch_bridge(span=24.0, rise=5.5, width=5.0, n=11):
    xs = np.linspace(0.0, span, n)

    def arch_y(x):
        return 4.0 * rise * x * (span - x) / max(span**2, 1e-12)

    nodes = []
    deck_left = []
    deck_right = []
    arch_left = []
    arch_right = []
    z_left = -0.5 * width
    z_right = 0.5 * width

    for x in xs:
        deck_left.append(len(nodes))
        nodes.append([x, 0.0, z_left])
        deck_right.append(len(nodes))
        nodes.append([x, 0.0, z_right])
        arch_left.append(len(nodes))
        nodes.append([x, arch_y(x), z_left])
        arch_right.append(len(nodes))
        nodes.append([x, arch_y(x), z_right])

    nodes = np.array(nodes, dtype=float)

    active_edges = []
    for k in range(n - 1):
        active_edges += [
            (deck_left[k], deck_left[k + 1], "deck"),
            (deck_right[k], deck_right[k + 1], "deck"),
            (arch_left[k], arch_left[k + 1], "arch"),
            (arch_right[k], arch_right[k + 1], "arch"),
        ]
    for k in range(n):
        active_edges += [
            (deck_left[k], deck_right[k], "deck_cross"),
            (arch_left[k], arch_right[k], "arch_cross"),
            (deck_left[k], arch_left[k], "hanger"),
            (deck_right[k], arch_right[k], "hanger"),
        ]

    candidate_edges = []
    for k in range(n - 1):
        candidate_edges += [
            (arch_left[k], arch_right[k + 1], "arch_brace"),
            (arch_right[k], arch_left[k + 1], "arch_brace"),
            (deck_left[k], deck_right[k + 1], "deck_brace"),
            (deck_right[k], deck_left[k + 1], "deck_brace"),
            (deck_left[k], arch_left[k + 1], "side_brace"),
            (arch_left[k], deck_left[k + 1], "side_brace"),
            (deck_right[k], arch_right[k + 1], "side_brace"),
            (arch_right[k], deck_right[k + 1], "side_brace"),
        ]

    groups = {
        "deck_left": deck_left,
        "deck_right": deck_right,
        "arch_left": arch_left,
        "arch_right": arch_right,
    }
    return nodes, active_edges, candidate_edges, groups


def base_edge_metric(edge_type):
    vals = {
        "deck": 170.0,
        "arch": 220.0,
        "hanger": 110.0,
        "deck_cross": 95.0,
        "arch_cross": 90.0,
        "arch_brace": 80.0,
        "deck_brace": 75.0,
        "side_brace": 70.0,
    }
    return vals[edge_type]


def support_dofs(groups):
    ends = [groups["deck_left"][0], groups["deck_right"][0], groups["deck_left"][-1], groups["deck_right"][-1]]
    return [3 * i + a for i in ends for a in range(3)]


def lane_load_vector(nodes, groups, magnitude=1.3, xc=None, sigma=2.1, lane="left"):
    if xc is None:
        xc = 0.72 * np.max(nodes[:, 0])
    f = np.zeros(3 * len(nodes), dtype=float)
    lane_nodes = groups["deck_left"] if lane == "left" else groups["deck_right"]
    for idx in lane_nodes:
        x = nodes[idx, 0]
        w = np.exp(-0.5 * ((x - xc) / max(sigma, 1e-12)) ** 2)
        f[3 * idx + 1] -= magnitude * w
    return f


def objective_gradient(q, groups, n_nodes, lane="left"):
    g = np.zeros(3 * n_nodes, dtype=float)
    lane_nodes = groups["deck_left"] if lane == "left" else groups["deck_right"]
    for idx in lane_nodes:
        g[3 * idx + 1] = q[3 * idx + 1]
    return g


def objective_value(q, groups, lane="left"):
    lane_nodes = groups["deck_left"] if lane == "left" else groups["deck_right"]
    vals = np.array([q[3 * idx + 1] for idx in lane_nodes], dtype=float)
    return 0.5 * np.sum(vals**2)


def mean_lane_drop(q, groups, lane="left"):
    lane_nodes = groups["deck_left"] if lane == "left" else groups["deck_right"]
    vals = np.array([-q[3 * idx + 1] for idx in lane_nodes], dtype=float)
    return float(np.mean(vals))


def build_edge_operator(nodes, edge):
    i, j, typ = edge
    ke, L, t = unit_bar_stiffness(nodes[i], nodes[j])
    ndof = 3 * len(nodes)
    K = np.zeros((ndof, ndof), dtype=float)
    dofs = [3 * i + a for a in range(3)] + [3 * j + a for a in range(3)]
    for a in range(6):
        for b in range(6):
            K[dofs[a], dofs[b]] += ke[a, b]
    E = np.zeros(ndof, dtype=float)
    E[3 * i:3 * i + 3] = -t / max(L, 1e-12)
    E[3 * j:3 * j + 3] = t / max(L, 1e-12)
    return K, E


def precompute_active(nodes, edges):
    basis = []
    E_ops = []
    mids = []
    for edge in edges:
        K, E = build_edge_operator(nodes, edge)
        basis.append(K)
        E_ops.append(E)
        i, j, _ = edge
        mids.append(0.5 * (nodes[i] + nodes[j]))
    return np.array(basis, dtype=float), np.array(E_ops, dtype=float), np.array(mids, dtype=float)


def assemble_from_metric(edge_basis, metric):
    return np.einsum("e,eij->ij", metric, edge_basis)


def solve_system(K, f, fixed_dofs):
    ndof = K.shape[0]
    fixed = set(int(x) for x in fixed_dofs)
    free = np.array([d for d in range(ndof) if d not in fixed], dtype=int)
    q = np.zeros(ndof, dtype=float)
    Kff = K[np.ix_(free, free)]
    ff = f[free]
    reg = 1e-6 * np.eye(Kff.shape[0], dtype=float)
    try:
        q[free] = np.linalg.solve(Kff + reg, ff)
    except np.linalg.LinAlgError:
        q[free] = np.linalg.lstsq(Kff + reg, ff, rcond=None)[0]
    return q


def compute_candidate_sensitivities(nodes, active_edges, candidate_edges, groups, load_vec):
    fixed_dofs = support_dofs(groups)
    edge_basis, E_ops, _ = precompute_active(nodes, active_edges)
    metric = np.array([base_edge_metric(typ) for _, _, typ in active_edges], dtype=float)
    K = assemble_from_metric(edge_basis, metric)
    q = solve_system(K, load_vec, fixed_dofs)
    p = solve_system(K.T, objective_gradient(q, groups, len(nodes)), fixed_dofs)

    sensitivities = []
    for edge in candidate_edges:
        _, E = build_edge_operator(nodes, edge)
        k = base_edge_metric(edge[2])
        L = float(np.linalg.norm(nodes[edge[1]] - nodes[edge[0]]))
        # exact derivative dJ/ds of K(s)=K+s*k*K_e with K_e = t t^T / L:
        # -p^T (k K_e) q = -k * L * eps_q * eps_p   (eps = strain = E @ q)
        sensitivities.append(-k * L * float(E @ q) * float(E @ p))
    return np.array(sensitivities, dtype=float), q, p


def plot_bridge_candidates(nodes, active_edges, candidate_edges, values, selected_edges, outfile, title):
    vmax = max(np.max(np.abs(values)) if len(values) else 0.0, 1e-12)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection="3d")
    for i, j, _ in active_edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 2], nodes[j, 2]], [nodes[i, 1], nodes[j, 1]], color="0.6", linewidth=1.2)
    for e, (i, j, _) in enumerate(candidate_edges):
        c = plt.get_cmap("coolwarm")(0.5 + 0.5 * values[e] / vmax)
        lw = 0.8 + 3.0 * abs(values[e]) / vmax
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 2], nodes[j, 2]], [nodes[i, 1], nodes[j, 1]], color=c, linewidth=lw, alpha=0.85)
    for i, j, _ in selected_edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 2], nodes[j, 2]], [nodes[i, 1], nodes[j, 1]], color="crimson", linewidth=3.0)
    ax.scatter(nodes[:, 0], nodes[:, 2], nodes[:, 1], color="black", s=10)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_growth_curves(objectives, drifts, best_sens):
    steps = np.arange(len(objectives))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, objectives, marker="o")
    axes[0].set_title("Objective")
    axes[1].plot(steps, drifts, marker="o")
    axes[1].set_title("Mean lane drop")
    axes[2].plot(steps[:-1], best_sens, marker="o")
    axes[2].set_title("Best insertion sensitivity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Inserted bars")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bridge_growth_curves.png"), dpi=180)
    plt.close(fig)


def plot_selected_sensitivities(selected_sens):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.arange(1, len(selected_sens) + 1), selected_sens, marker="o", color="firebrick")
    ax.grid(alpha=0.25)
    ax.set_xlabel("Insertion step")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Selected Hante-type insertion sensitivities")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bridge_selected_sensitivities.png"), dpi=180)
    plt.close(fig)


def main():
    nodes, active_edges, candidate_edges, groups = build_twin_arch_bridge()
    load_vec = lane_load_vector(nodes, groups, magnitude=1.35, xc=0.72 * np.max(nodes[:, 0]), sigma=2.0, lane="left")

    sens0, q0, p0 = compute_candidate_sensitivities(nodes, active_edges, candidate_edges, groups, load_vec)
    obj0 = objective_value(q0, groups)
    drift0 = mean_lane_drop(q0, groups)

    print(f"initial objective = {obj0:.6f}", flush=True)
    print(f"initial mean lane drop = {drift0:.6f}", flush=True)
    print(f"initial best sensitivity = {np.min(sens0):.6f}", flush=True)
    print(f"initial best candidate = {candidate_edges[int(np.argmin(sens0))]}", flush=True)

    plot_bridge_candidates(
        nodes,
        active_edges,
        candidate_edges,
        sens0,
        [],
        os.path.join(OUT_DIR, "bridge_initial_candidate_sensitivities.png"),
        "Initial Hante-type insertion sensitivities",
    )

    objectives = [obj0]
    drifts = [drift0]
    best_sens = []
    selected_edges = []

    n_insert = min(12, len(candidate_edges))
    stage_snapshots = {0: (list(active_edges), list(candidate_edges), sens0.copy(), list(selected_edges))}

    for step in range(n_insert):
        sens, q, p = compute_candidate_sensitivities(nodes, active_edges, candidate_edges, groups, load_vec)
        idx = int(np.argmin(sens))
        best_sens.append(float(sens[idx]))
        chosen = candidate_edges.pop(idx)
        active_edges.append(chosen)
        selected_edges.append(chosen)

        edge_basis, E_ops, _ = precompute_active(nodes, active_edges)
        metric = np.array([base_edge_metric(typ) for _, _, typ in active_edges], dtype=float)
        K = assemble_from_metric(edge_basis, metric)
        q_new = solve_system(K, load_vec, support_dofs(groups))
        objectives.append(objective_value(q_new, groups))
        drifts.append(mean_lane_drop(q_new, groups))

        if step + 1 in {1, 3, 6, n_insert}:
            sens_next, _, _ = compute_candidate_sensitivities(nodes, active_edges, candidate_edges, groups, load_vec) if candidate_edges else (np.array([]), None, None)
            stage_snapshots[step + 1] = (list(active_edges), list(candidate_edges), sens_next.copy(), list(selected_edges))

    print(f"final objective = {objectives[-1]:.6f}", flush=True)
    print(f"final mean lane drop = {drifts[-1]:.6f}", flush=True)
    print(f"selected edges = {selected_edges}", flush=True)

    plot_growth_curves(objectives, drifts, best_sens)
    plot_selected_sensitivities(best_sens)

    ordered_steps = sorted(stage_snapshots.keys())
    n_panels = len(ordered_steps)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(14, 4.8 * nrows))
    for k, step in enumerate(ordered_steps, start=1):
        ax = fig.add_subplot(nrows, ncols, k, projection="3d")
        act, cand, vals, sel = stage_snapshots[step]
        vmax = max(np.max(np.abs(vals)) if len(vals) else 0.0, 1e-12)
        for i, j, _ in act:
            ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 2], nodes[j, 2]], [nodes[i, 1], nodes[j, 1]], color="0.65", linewidth=1.0)
        for e, (i, j, _) in enumerate(cand):
            c = plt.get_cmap("coolwarm")(0.5 + 0.5 * vals[e] / vmax)
            lw = 0.8 + 2.6 * abs(vals[e]) / vmax
            ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 2], nodes[j, 2]], [nodes[i, 1], nodes[j, 1]], color=c, linewidth=lw)
        for i, j, _ in sel:
            ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 2], nodes[j, 2]], [nodes[i, 1], nodes[j, 1]], color="crimson", linewidth=2.6)
        ax.scatter(nodes[:, 0], nodes[:, 2], nodes[:, 1], color="black", s=8)
        ax.set_title(f"After {step} insertions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bridge_growth_progress.png"), dpi=180)
    plt.close(fig)

    print(f"Results saved in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
