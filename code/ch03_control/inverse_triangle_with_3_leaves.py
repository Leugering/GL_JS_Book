import itertools
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
from numpy.linalg import matrix_rank, lstsq
import matplotlib.pyplot as plt

"""
Triangle-with-three-leaves example.

What this script does
---------------------
1. Builds the graph and weighted Laplacian for the example
   V = {1,2,3,4,5,6}
   E = {(1,2),(2,3),(3,1),(1,4),(2,5),(3,6)}

2. Checks structural controllability-style conditions discussed in the text:
   - connectivity to leaders
   - distance-information preserving ordering
   - simple automorphism / symmetry obstruction (leader-preserving)

3. Builds the first-order realization for the second-order spring network
      M q¨ + L q = B u
   and computes the Kalman rank numerically.

4. Performs a simple edge-weight identifiability study by finite-difference
   sensitivities of Markov parameters / impulse response samples for
   chosen actuator/sensor sets.

This is meant as a worked computational companion to the book section.
"""

# ---------------------------------------------------------------------
# Graph data
# ---------------------------------------------------------------------
NODES = [1, 2, 3, 4, 5, 6]
EDGES = [
    (1, 2),  # e1
    (2, 3),  # e2
    (3, 1),  # e3
    (1, 4),  # e4
    (2, 5),  # e5
    (3, 6),  # e6
]

NODE_TO_IDX = {v: i for i, v in enumerate(NODES)}
RESULTS_DIR = Path(__file__).resolve().parent / "results_inverse_triangle_with_3_leaves"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NODE_POS = {
    1: np.array([0.0, 1.0]),
    2: np.array([1.0, 1.0]),
    3: np.array([0.5, 1.866]),
    4: np.array([-0.45, 0.10]),
    5: np.array([1.45, 0.10]),
    6: np.array([0.50, 2.85]),
}


def incidence_matrix(nodes, edges):
    """Node-edge incidence matrix D in the book's convention.

    D[v,e] = +1 if edge e enters node v,
             -1 if edge e leaves node v.
    """
    D = np.zeros((len(nodes), len(edges)))
    for e_idx, (u, v) in enumerate(edges):
        D[NODE_TO_IDX[u], e_idx] = -1.0
        D[NODE_TO_IDX[v], e_idx] = +1.0
    return D


def adjacency_list(nodes, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


D = incidence_matrix(NODES, EDGES)
ADJ = adjacency_list(NODES, EDGES)


# ---------------------------------------------------------------------
# Distances and structural conditions
# ---------------------------------------------------------------------
def bfs_distances(source, adj):
    dist = {v: np.inf for v in adj.keys()}
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def all_leader_distances(leaders, adj):
    out = {}
    for ell in leaders:
        out[ell] = bfs_distances(ell, adj)
    return out


def connected_to_leaders(leaders, nodes, adj):
    dists = all_leader_distances(leaders, adj)
    ok = {}
    for v in nodes:
        ok[v] = any(np.isfinite(dists[ell][v]) for ell in leaders)
    return ok


def find_distance_information_ordering(nodes, leaders, adj):
    """Brute-force search for a distance-information preserving ordering.

    Returns (ordering, chosen_leaders) or None.
    Suitable here because n=6 only.
    """
    dists = all_leader_distances(leaders, adj)

    def backtrack(remaining, ordering, chosen):
        if not remaining:
            return ordering, chosen
        rem = list(remaining)
        for v in rem:
            for ell in leaders:
                dv = dists[ell][v]
                if not np.isfinite(dv):
                    continue
                good = True
                for w in rem:
                    if w == v:
                        continue
                    if not (dv < dists[ell][w]):
                        good = False
                        break
                if good:
                    new_remaining = [w for w in rem if w != v]
                    ans = backtrack(new_remaining, ordering + [v], chosen + [ell])
                    if ans is not None:
                        return ans
        return None

    return backtrack(list(nodes), [], [])


# ---------------------------------------------------------------------
# Automorphisms preserving selected sets
# ---------------------------------------------------------------------
def is_graph_automorphism(perm, edges):
    edge_set = {frozenset(e) for e in edges}
    mapped = {frozenset((perm[u], perm[v])) for u, v in edges}
    return mapped == edge_set


def automorphisms_preserving_sets(nodes, edges, fixed_sets):
    """Return all automorphisms preserving each set in fixed_sets as a set."""
    autos = []
    node_list = list(nodes)
    for perm_tuple in itertools.permutations(node_list):
        perm = {node_list[i]: perm_tuple[i] for i in range(len(node_list))}
        if not is_graph_automorphism(perm, edges):
            continue
        good = True
        for S in fixed_sets:
            if {perm[v] for v in S} != set(S):
                good = False
                break
        if good:
            autos.append(perm)
    return autos


# ---------------------------------------------------------------------
# Weighted dynamics and Kalman controllability
# ---------------------------------------------------------------------
def make_laplacian(D, k):
    K = np.diag(k)
    return D @ K @ D.T


def first_order_matrices(M, L, controlled_nodes):
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    B2 = np.zeros((n, len(controlled_nodes)))
    for j, v in enumerate(controlled_nodes):
        B2[NODE_TO_IDX[v], j] = 1.0
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-Minv @ L, np.zeros((n, n))],
    ])
    B = np.vstack([np.zeros((n, len(controlled_nodes))), Minv @ B2])
    return A, B


def kalman_rank(A, B):
    n = A.shape[0]
    cols = [B]
    AB = B.copy()
    for _ in range(1, n):
        AB = A @ AB
        cols.append(AB)
    K = np.hstack(cols)
    return matrix_rank(K), K


# ---------------------------------------------------------------------
# Observability and identifiability by sampled Markov parameters
# ---------------------------------------------------------------------
def output_matrix(measured_nodes):
    C = np.zeros((len(measured_nodes), len(NODES)))
    for i, v in enumerate(measured_nodes):
        C[i, NODE_TO_IDX[v]] = 1.0
    return C


def first_order_output_matrix(measured_nodes):
    Cq = output_matrix(measured_nodes)
    n = len(NODES)
    return np.hstack([Cq, np.zeros((Cq.shape[0], n))])


def observability_rank(A, C):
    n = A.shape[0]
    blocks = [C]
    CA = C.copy()
    for _ in range(1, n):
        CA = CA @ A
        blocks.append(CA)
    O = np.vstack(blocks)
    return matrix_rank(O), O


def sampled_impulse_signature(M, k, controlled_nodes, measured_nodes, num_steps=10):
    """Use sampled Markov parameters C A^r B as a finite signature."""
    L = make_laplacian(D, k)
    A, B = first_order_matrices(M, L, controlled_nodes)
    C = first_order_output_matrix(measured_nodes)
    blocks = []
    ArB = B.copy()
    for _ in range(num_steps):
        blocks.append(C @ ArB)
        ArB = A @ ArB
    return np.concatenate([blk.ravel() for blk in blocks])


def sensitivity_rank(M, k, controlled_nodes, measured_nodes, num_steps=10, h=1e-6):
    base = sampled_impulse_signature(M, k, controlled_nodes, measured_nodes, num_steps=num_steps)
    m = len(k)
    S = np.zeros((base.size, m))
    for j in range(m):
        kp = np.array(k, dtype=float)
        kp[j] += h
        fp = sampled_impulse_signature(M, kp, controlled_nodes, measured_nodes, num_steps=num_steps)
        S[:, j] = (fp - base) / h
    return matrix_rank(S), S


def simulate_impulse_response(M, k, controlled_nodes, measured_nodes, num_steps=120, dt=0.03):
    """Discrete-time impulse-response samples for q-only outputs."""
    L = make_laplacian(D, k)
    A, B = first_order_matrices(M, L, controlled_nodes)
    C = first_order_output_matrix(measured_nodes)
    x = np.zeros(A.shape[0])
    x = x + B[:, 0]
    ys = []
    for _ in range(num_steps):
        ys.append((C @ x).copy())
        x = x + dt * (A @ x)
    return np.array(ys)


def build_trajectory_regression(M, controlled_nodes, measured_nodes, num_steps=120, dt=0.03):
    """Linearization y(k) ~ y0 + J (k-k0) from impulse-response samples."""
    k0 = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0], dtype=float)
    y0 = simulate_impulse_response(M, k0, controlled_nodes, measured_nodes, num_steps=num_steps, dt=dt).ravel()
    J = np.zeros((y0.size, len(k0)))
    h = 1e-5
    for j in range(len(k0)):
        kp = k0.copy()
        kp[j] += h
        yp = simulate_impulse_response(M, kp, controlled_nodes, measured_nodes, num_steps=num_steps, dt=dt).ravel()
        J[:, j] = (yp - y0) / h
    return k0, y0, J


def recover_stiffnesses_linearized(M, k_true, controlled_nodes, measured_nodes, num_steps=120, dt=0.03, noise_level=0.0):
    """Recover k from synthetic measured trajectories via linearized least squares."""
    k0, y0, J = build_trajectory_regression(M, controlled_nodes, measured_nodes, num_steps=num_steps, dt=dt)
    y_true = simulate_impulse_response(M, k_true, controlled_nodes, measured_nodes, num_steps=num_steps, dt=dt).ravel()
    if noise_level > 0.0:
        rng = np.random.default_rng(7)
        scale = np.maximum(np.linalg.norm(y_true) / np.sqrt(y_true.size), 1e-8)
        y_true = y_true + noise_level * scale * rng.standard_normal(y_true.shape)
    dk, *_ = lstsq(J, y_true - y0, rcond=None)
    k_rec = np.clip(k0 + dk, 1e-6, None)
    rel_err = np.linalg.norm(k_rec - k_true) / np.linalg.norm(k_true)
    return {
        "k0": k0,
        "k_true": np.array(k_true, dtype=float),
        "k_rec": k_rec,
        "rel_err": rel_err,
        "y_true": y_true,
        "y0": y0,
        "J_rank": matrix_rank(J),
        "num_obs": y0.size,
    }


# ---------------------------------------------------------------------
# Utility: simple clamping by removing DOFs
# ---------------------------------------------------------------------
def reduced_system_data(clamped_nodes, D, M, controlled_nodes, measured_nodes):
    free_nodes = [v for v in NODES if v not in clamped_nodes]
    free_idx = [NODE_TO_IDX[v] for v in free_nodes]
    Df = D[free_idx, :]
    Mf = M[np.ix_(free_idx, free_idx)]

    controlled_free = [v for v in controlled_nodes if v in free_nodes]
    measured_free = [v for v in measured_nodes if v in free_nodes]

    node_to_idx_free = {v: i for i, v in enumerate(free_nodes)}

    def first_order_mats_free(k):
        Lf = Df @ np.diag(k) @ Df.T
        n = len(free_nodes)
        Minv = np.linalg.inv(Mf)
        B2 = np.zeros((n, len(controlled_free)))
        for j, v in enumerate(controlled_free):
            B2[node_to_idx_free[v], j] = 1.0
        A = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-Minv @ Lf, np.zeros((n, n))],
        ])
        B = np.vstack([np.zeros((n, len(controlled_free))), Minv @ B2])
        Cq = np.zeros((len(measured_free), n))
        for i, v in enumerate(measured_free):
            Cq[i, node_to_idx_free[v]] = 1.0
        C = np.hstack([Cq, np.zeros((len(measured_free), n))])
        return A, B, C, free_nodes

    return first_order_mats_free, free_nodes


def analyze_case(leaders, sensors, k, M):
    conn = connected_to_leaders(leaders, NODES, ADJ)
    ordering = find_distance_information_ordering(NODES, leaders, ADJ)
    autos = automorphisms_preserving_sets(NODES, EDGES, [set(leaders), set(sensors)])
    L = make_laplacian(D, k)
    A, B = first_order_matrices(M, L, leaders)
    C = first_order_output_matrix(sensors)
    rk_ctrb, _ = kalman_rank(A, B)
    rk_obsv, _ = observability_rank(A, C)
    rk_id, _ = sensitivity_rank(M, k, leaders, sensors, num_steps=12)
    return {
        "leaders": tuple(leaders),
        "sensors": tuple(sensors),
        "connected_all": all(conn.values()),
        "ordering_found": ordering is not None,
        "ordering_nodes": ordering[0] if ordering is not None else None,
        "ordering_leaders": ordering[1] if ordering is not None else None,
        "num_autos": len(autos),
        "ctrb_rank": rk_ctrb,
        "obsv_rank": rk_obsv,
        "id_rank": rk_id,
        "state_dim": A.shape[0],
        "num_edges": len(k),
    }


def plot_graph(k, recovered=None):
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    k = np.asarray(k, dtype=float)
    kmax = np.max(k)
    for e_idx, (u, v) in enumerate(EDGES):
        pu = NODE_POS[u]
        pv = NODE_POS[v]
        lw = 1.5 + 4.0 * k[e_idx] / kmax
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="#3268a8", lw=lw, alpha=0.85)
        mid = 0.5 * (pu + pv)
        label = f"$k_{{{e_idx+1}}}={k[e_idx]:.1f}$"
        if recovered is not None:
            label += f"\n$\\hat k_{{{e_idx+1}}}={recovered[e_idx]:.2f}$"
        ax.text(mid[0], mid[1], label, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    for v in NODES:
        p = NODE_POS[v]
        is_leaf = v in {4, 5, 6}
        ax.scatter([p[0]], [p[1]], s=220 if is_leaf else 170,
                   color="#d95f02" if is_leaf else "#1b9e77", edgecolor="black", zorder=3)
        ax.text(p[0], p[1], str(v), color="white", ha="center", va="center", fontsize=11, weight="bold")
    ax.set_title("Triangle-with-three-leaves spring graph")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = RESULTS_DIR / "inverse_triangle_with_3_leaves_graph.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_recovery(k_true, k_rec):
    idx = np.arange(len(k_true))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.36
    ax.bar(idx - width / 2, k_true, width=width, color="#4c78a8", label="true")
    ax.bar(idx + width / 2, k_rec, width=width, color="#f58518", label="recovered")
    ax.set_xticks(idx, [f"$e_{j+1}$" for j in idx])
    ax.set_ylabel("spring stiffness")
    ax.set_title("Synthetic stiffness recovery from measured leaf responses")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = RESULTS_DIR / "inverse_triangle_with_3_leaves_recovery.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def write_outputs(case_rows, recovery):
    summary_path = RESULTS_DIR / "inverse_triangle_with_3_leaves_summary.txt"
    latex_table_path = RESULTS_DIR / "inverse_triangle_with_3_leaves_cases_table.tex"
    recovery_table_path = RESULTS_DIR / "inverse_triangle_with_3_leaves_recovery_table.tex"

    with summary_path.open("w") as f:
        f.write("Triangle-with-three-leaves identifiability study\n")
        f.write("=" * 60 + "\n\n")
        for row in case_rows:
            f.write(
                f"leaders={row['leaders']} sensors={row['sensors']} "
                f"connected={row['connected_all']} ordering={row['ordering_found']} "
                f"autos={row['num_autos']} ctrb={row['ctrb_rank']}/{row['state_dim']} "
                f"obsv={row['obsv_rank']}/{row['state_dim']} id={row['id_rank']}/{row['num_edges']}\n"
            )
        f.write("\nSynthetic recovery\n")
        f.write("-" * 60 + "\n")
        f.write(f"true k      = {np.array2string(recovery['k_true'], precision=4)}\n")
        f.write(f"recovered k = {np.array2string(recovery['k_rec'], precision=4)}\n")
        f.write(f"relative error = {recovery['rel_err']:.6e}\n")
        f.write(f"linearized Jacobian rank = {recovery['J_rank']} / {len(recovery['k_true'])}\n")

    with latex_table_path.open("w") as f:
        f.write("\\begin{tabular}{ccrrrrr}\n")
        f.write("\\hline\n")
        f.write("Leaders & Sensors & Autos & Ctrb & Obsv & Id & Ord. \\\\\n")
        f.write("\\hline\n")
        for row in case_rows:
            f.write(
                f"${set(row['leaders'])}$ & ${set(row['sensors'])}$ & {row['num_autos']} & "
                f"{row['ctrb_rank']}/{row['state_dim']} & {row['obsv_rank']}/{row['state_dim']} & "
                f"{row['id_rank']}/{row['num_edges']} & "
                f"{'yes' if row['ordering_found'] else 'no'} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    with recovery_table_path.open("w") as f:
        f.write("\\begin{tabular}{crr}\n")
        f.write("\\hline\n")
        f.write("Edge & True $k_e$ & Recovered $\\hat k_e$ \\\\\n")
        f.write("\\hline\n")
        for j, (kt, kr) in enumerate(zip(recovery["k_true"], recovery["k_rec"]), start=1):
            f.write(f"$e_{j}$ & {kt:.4f} & {kr:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return summary_path, latex_table_path, recovery_table_path


# ---------------------------------------------------------------------
# Run analyses
# ---------------------------------------------------------------------
def print_case(leaders, sensors=None, k=None, M=None):
    if sensors is None:
        sensors = leaders
    if k is None:
        k = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    if M is None:
        M = np.diag([1.0, 1.2, 1.4, 1.1, 1.3, 1.5])

    print("=" * 78)
    print(f"Leaders/actuators: {leaders}")
    print(f"Sensors:           {sensors}")

    # connectivity
    conn = connected_to_leaders(leaders, NODES, ADJ)
    print("Connected to a leader:", conn)

    # distance-ordering
    ordering = find_distance_information_ordering(NODES, leaders, ADJ)
    if ordering is None:
        print("Distance-information preserving ordering: NONE")
    else:
        ord_nodes, ord_leaders = ordering
        print("Distance-information preserving ordering found:")
        print("  nodes  =", ord_nodes)
        print("  leader =", ord_leaders)

    # automorphisms preserving leader and sensor sets
    autos = automorphisms_preserving_sets(NODES, EDGES, [set(leaders), set(sensors)])
    print(f"Automorphisms preserving leader/sensor sets: {len(autos)}")
    for idx, perm in enumerate(autos[:6]):
        print(f"  auto {idx+1}: {perm}")
    if len(autos) > 6:
        print("  ...")

    # Kalman controllability and observability ranks
    L = make_laplacian(D, k)
    A, B = first_order_matrices(M, L, leaders)
    C = first_order_output_matrix(sensors)
    rk_ctrb, _ = kalman_rank(A, B)
    rk_obsv, _ = observability_rank(A, C)
    print(f"Kalman controllability rank: {rk_ctrb} / {A.shape[0]}")
    print(f"Observability rank:          {rk_obsv} / {A.shape[0]}")

    # identifiability sensitivity rank
    rk_id, _ = sensitivity_rank(M, k, leaders, sensors, num_steps=12)
    print(f"Sensitivity/identifiability rank: {rk_id} / {len(k)}")


def main():
    print("Incidence matrix D:\n", D)

    k = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0], dtype=float)
    M = np.diag([1.0, 1.2, 1.4, 1.1, 1.3, 1.5])

    case_specs = [
        ([4], [4]),
        ([4, 5], [4, 5]),
        ([4, 5, 6], [4, 5, 6]),
        ([4], [4, 5, 6]),
        ([4, 5], [4, 5, 6]),
    ]
    case_rows = []
    for leaders, sensors in case_specs:
        print_case(leaders, sensors, k=k, M=M)
        case_rows.append(analyze_case(leaders, sensors, k, M))

    # Clamped-node example: clamp node 6
    print("=" * 78)
    print("Clamped-node study: clamp node 6, leaders/sensors = {4}")
    first_order_free, free_nodes = reduced_system_data([6], D, M, [4], [4])
    A, B, C, free_nodes = first_order_free(k)
    rk_ctrb, _ = kalman_rank(A, B)
    rk_obsv, _ = observability_rank(A, C)
    print("Free nodes:", free_nodes)
    print(f"Reduced controllability rank: {rk_ctrb} / {A.shape[0]}")
    print(f"Reduced observability rank:   {rk_obsv} / {A.shape[0]}")

    recovery = recover_stiffnesses_linearized(
        M=M,
        k_true=k,
        controlled_nodes=[4, 5, 6],
        measured_nodes=[4, 5, 6],
        num_steps=150,
        dt=0.02,
        noise_level=0.0,
    )
    print("=" * 78)
    print("Synthetic stiffness recovery from measured leaf responses")
    print("true k      =", np.array2string(recovery["k_true"], precision=4))
    print("recovered k =", np.array2string(recovery["k_rec"], precision=4))
    print(f"relative recovery error: {recovery['rel_err']:.6e}")
    print(f"linearized recovery Jacobian rank: {recovery['J_rank']} / {len(k)}")

    graph_path = plot_graph(k, recovery["k_rec"])
    recovery_plot_path = plot_recovery(recovery["k_true"], recovery["k_rec"])
    summary_path, latex_table_path, recovery_table_path = write_outputs(case_rows, recovery)
    print("Saved:")
    print(" ", graph_path)
    print(" ", recovery_plot_path)
    print(" ", summary_path)
    print(" ", latex_table_path)
    print(" ", recovery_table_path)


if __name__ == "__main__":
    main()
