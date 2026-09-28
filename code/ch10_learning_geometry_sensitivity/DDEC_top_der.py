import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ============================================================
# 1. Fine-scale scalar elliptic PDE on a Cartesian grid
#    -div(a grad u) = 0 in (0,1)^2
#    u = g on boundary
# ============================================================

def conductivity_field(nx, ny, alpha_inclusion=6.0, radius=0.18):
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    a = np.ones((nx, ny), dtype=float)
    mask = (X - 0.5) ** 2 + (Y - 0.5) ** 2 < radius ** 2
    a[mask] = alpha_inclusion
    return X, Y, a


def boundary_data(x, y):
    return x + 0.25 * np.sin(np.pi * y)


def solve_fine_poisson(nx=41, ny=41, alpha_inclusion=6.0, radius=0.18,
                       max_iter=20000, tol=1e-10, omega=1.0):
    X, Y, a = conductivity_field(nx, ny, alpha_inclusion, radius)
    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)

    u = np.zeros((nx, ny), dtype=float)

    u[0, :] = boundary_data(X[0, :], Y[0, :])
    u[-1, :] = boundary_data(X[-1, :], Y[-1, :])
    u[:, 0] = boundary_data(X[:, 0], Y[:, 0])
    u[:, -1] = boundary_data(X[:, -1], Y[:, -1])

    def harm(x, y):
        return 2.0 * x * y / max(x + y, 1e-14)

    for it in range(max_iter):
        maxdiff = 0.0

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                ae = harm(a[i, j], a[i + 1, j]) / hx**2
                aw = harm(a[i, j], a[i - 1, j]) / hx**2
                an = harm(a[i, j], a[i, j + 1]) / hy**2
                a_s = harm(a[i, j], a[i, j - 1]) / hy**2

                denom = ae + aw + an + a_s
                u_new = (
                    ae * u[i + 1, j]
                    + aw * u[i - 1, j]
                    + an * u[i, j + 1]
                    + a_s * u[i, j - 1]
                ) / denom

                diff = abs(u_new - u[i, j])
                maxdiff = max(maxdiff, diff)
                u[i, j] = (1.0 - omega) * u[i, j] + omega * u_new

        if maxdiff < tol:
            print(f"[fine solve] converged in {it} iterations")
            break

    return X, Y, a, u


# ============================================================
# 2. Coarsening and graph
# ============================================================

def coarse_partition(nx, ny, cx, cy):
    block_id = np.zeros((nx, ny), dtype=int)

    xcuts = np.linspace(0, nx, cx + 1, dtype=int)
    ycuts = np.linspace(0, ny, cy + 1, dtype=int)

    bid = 0
    for I in range(cx):
        for J in range(cy):
            block_id[xcuts[I]:xcuts[I + 1], ycuts[J]:ycuts[J + 1]] = bid
            bid += 1

    return block_id


def build_coarse_graph(cx, cy):
    n_nodes = cx * cy

    def idx(I, J):
        return I * cy + J

    edges = []
    for I in range(cx):
        for J in range(cy):
            if I + 1 < cx:
                edges.append((idx(I, J), idx(I + 1, J)))
            if J + 1 < cy:
                edges.append((idx(I, J), idx(I, J + 1)))
    return n_nodes, edges


def coarse_node_positions(cx, cy):
    pos = []
    for I in range(cx):
        for J in range(cy):
            pos.append([(I + 0.5) / cx, (J + 0.5) / cy])
    return np.array(pos)


def coarse_averages(field, block_id, n_nodes):
    vals = np.zeros(n_nodes, dtype=float)
    counts = np.zeros(n_nodes, dtype=float)

    nx, ny = field.shape
    for i in range(nx):
        for j in range(ny):
            b = block_id[i, j]
            vals[b] += field[i, j]
            counts[b] += 1.0

    vals /= np.maximum(counts, 1.0)
    return vals


def incidence_matrix(n_nodes, edges):
    m = len(edges)
    B = np.zeros((m, n_nodes), dtype=float)
    for e, (i, j) in enumerate(edges):
        B[e, i] = -1.0
        B[e, j] = 1.0
    return B


def boundary_nodes_from_layout(cx, cy):
    bnd = []
    for I in range(cx):
        for J in range(cy):
            if I == 0 or I == cx - 1 or J == 0 or J == cy - 1:
                bnd.append(I * cy + J)
    return np.array(sorted(set(bnd)), dtype=int)


# ============================================================
# 3. Forward graph solve
# ============================================================

def graph_solve_from_weights(theta, B_torch, dirichlet_nodes, dirichlet_values, rhs):
    w = F.softplus(theta) + 1e-6
    L = B_torch.T @ torch.diag(w) @ B_torch

    n = L.shape[0]
    all_idx = torch.arange(n, dtype=torch.long)
    mask = torch.ones(n, dtype=torch.bool)
    mask[dirichlet_nodes] = False
    free = all_idx[mask]

    u = torch.zeros(n, dtype=torch.float32)
    u[dirichlet_nodes] = dirichlet_values

    L_ff = L[free][:, free]
    L_fb = L[free][:, dirichlet_nodes]
    rhs_f = rhs[free] - L_fb @ dirichlet_values

    u_f = torch.linalg.solve(L_ff, rhs_f)
    u[free] = u_f
    return u, w, L, free


# ============================================================
# 4. Adjoint and sensitivities
# ============================================================

def graph_adjoint_and_sensitivity(u, u_obs, B_torch, w, L, free, dirichlet_nodes):
    """
    J(u) = 0.5 * ||u - u_obs||^2 over interior nodes only.
    Adjoint:
        L^T p = dJ/du
    with homogeneous adjoint Dirichlet condition on boundary nodes.
    Sensitivity:
        dJ/dw_e = - (Bu)_e (Bp)_e
    """
    n = len(u)
    gradJ = torch.zeros(n, dtype=torch.float32)
    gradJ[free] = u[free] - u_obs[free]

    p = torch.zeros(n, dtype=torch.float32)

    L_ff = L[free][:, free]
    gradJ_f = gradJ[free]

    # L is symmetric, but write as transpose conceptually
    p_f = torch.linalg.solve(L_ff.T, gradJ_f)
    p[free] = p_f

    Bu = B_torch @ u
    Bp = B_torch @ p

    sens = -(Bu * Bp)
    return p, sens, Bu, Bp


# ============================================================
# 5. Learning graph metric
# ============================================================

def learn_graph_metric(u_obs, edges, cx, cy, n_iter=4000, lr=5e-2,
                       reg_weight=1e-5, out_dir="results_ddec_scalar_poisson"):
    os.makedirs(out_dir, exist_ok=True)

    n_nodes = cx * cy
    B = incidence_matrix(n_nodes, edges)
    B_torch = torch.tensor(B, dtype=torch.float32)

    dirichlet_nodes_np = boundary_nodes_from_layout(cx, cy)
    dirichlet_nodes = torch.tensor(dirichlet_nodes_np, dtype=torch.long)

    dirichlet_values = torch.tensor(u_obs[dirichlet_nodes_np], dtype=torch.float32)
    rhs = torch.zeros(n_nodes, dtype=torch.float32)
    u_obs_torch = torch.tensor(u_obs, dtype=torch.float32)

    theta = torch.nn.Parameter(torch.zeros(len(edges), dtype=torch.float32))
    optimizer = torch.optim.Adam([theta], lr=lr)

    history = []

    for it in range(n_iter):
        optimizer.zero_grad()

        u_pred, w_pred, L, free = graph_solve_from_weights(
            theta, B_torch, dirichlet_nodes, dirichlet_values, rhs
        )

        misfit = torch.mean((u_pred[free] - u_obs_torch[free]) ** 2)
        reg = reg_weight * torch.mean(w_pred ** 2)
        loss = misfit + reg

        loss.backward()
        optimizer.step()

        history.append(float(loss.item()))

        if it % 250 == 0:
            print(f"[train] iter={it:4d} loss={loss.item():.6e} misfit={misfit.item():.6e}")

    with torch.no_grad():
        u_pred, w_pred, L, free = graph_solve_from_weights(
            theta, B_torch, dirichlet_nodes, dirichlet_values, rhs
        )
        p, sens, Bu, Bp = graph_adjoint_and_sensitivity(
            u_pred, u_obs_torch, B_torch, w_pred, L, free, dirichlet_nodes
        )

    return {
        "u_pred": u_pred.detach().cpu().numpy(),
        "w_pred": w_pred.detach().cpu().numpy(),
        "L_pred": L.detach().cpu().numpy(),
        "p": p.detach().cpu().numpy(),
        "sens": sens.detach().cpu().numpy(),
        "Bu": Bu.detach().cpu().numpy(),
        "Bp": Bp.detach().cpu().numpy(),
        "history": np.array(history),
        "B": B,
        "free": free.detach().cpu().numpy(),
    }


# ============================================================
# 6. Visualization
# ============================================================

def plot_fields(X, Y, a, u_fine, u_obs_coarse, u_pred_coarse, cx, cy, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    im0 = ax[0].imshow(a.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="equal")
    ax[0].set_title("fine conductivity")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(u_fine.T, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", aspect="equal")
    ax[1].set_title("fine PDE solution")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    Upred = u_pred_coarse.reshape(cx, cy)
    vmax = max(np.max(np.abs(u_obs_coarse)), np.max(np.abs(u_pred_coarse)))

    im2 = ax[2].imshow(Upred.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="coolwarm", aspect="equal", vmin=-vmax, vmax=vmax)
    ax[2].set_title("learned coarse graph solution")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fields.png"), dpi=200)
    plt.close()


def plot_graph_edge_values(edges, edge_values, cx, cy, title, outfile):
    pos = coarse_node_positions(cx, cy)

    vmax = max(np.max(np.abs(edge_values)), 1e-12)
    cmap = plt.get_cmap("coolwarm")

    plt.figure(figsize=(6, 6))
    for e, (i, j) in enumerate(edges):
        val = edge_values[e]
        t = 0.5 + 0.5 * val / vmax
        color = cmap(t)
        lw = 0.5 + 5.0 * abs(val) / vmax

        plt.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=color,
            linewidth=lw,
        )

    plt.scatter(pos[:, 0], pos[:, 1], color="black", s=30)
    plt.axis("equal")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_weights(edges, w_pred, cx, cy, out_dir):
    pos = coarse_node_positions(cx, cy)
    wmax = max(np.max(w_pred), 1e-12)

    plt.figure(figsize=(6, 6))
    for e, (i, j) in enumerate(edges):
        lw = 0.5 + 5.0 * w_pred[e] / wmax
        plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], linewidth=lw)

    plt.scatter(pos[:, 0], pos[:, 1], color="black", s=30)
    plt.axis("equal")
    plt.axis("off")
    plt.title("learned graph metric weights")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learned_weights.png"), dpi=200)
    plt.close()


def plot_history(history, out_dir):
    plt.figure(figsize=(6, 4))
    plt.semilogy(history)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("training history")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=200)
    plt.close()


def plot_coarse_node_fields(u_obs, u_pred, p, cx, cy, out_dir):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    Uobs = u_obs.reshape(cx, cy)
    Upred = u_pred.reshape(cx, cy)
    P = p.reshape(cx, cy)

    im0 = ax[0].imshow(Uobs.T, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", aspect="equal")
    ax[0].set_title("observed coarse state")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(Upred.T, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", aspect="equal")
    ax[1].set_title("predicted coarse state")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(P.T, origin="lower", extent=[0, 1, 0, 1], cmap="PuOr", aspect="equal")
    ax[2].set_title("adjoint field")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coarse_nodes_and_adjoint.png"), dpi=200)
    plt.close()


# ============================================================
# 7. Main
# ============================================================

def run_ddec_scalar_poisson_with_sensitivity(
    nx=41, ny=41,
    cx=6, cy=6,
    alpha_inclusion=6.0,
    radius=0.18,
    out_dir="results_ddec_scalar_poisson_sensitivity"
):
    os.makedirs(out_dir, exist_ok=True)

    print("Solving fine scalar elliptic PDE ...")
    X, Y, a, u_fine = solve_fine_poisson(
        nx=nx, ny=ny,
        alpha_inclusion=alpha_inclusion,
        radius=radius
    )

    print("Building coarse graph from partition ...")
    block_id = coarse_partition(nx, ny, cx, cy)
    n_nodes, edges = build_coarse_graph(cx, cy)

    u_obs = coarse_averages(u_fine, block_id, n_nodes)

    print("Learning graph metric weights ...")
    result = learn_graph_metric(u_obs, edges, cx, cy, out_dir=out_dir)

    u_pred = result["u_pred"]
    w_pred = result["w_pred"]
    p = result["p"]
    sens = result["sens"]
    Bu = result["Bu"]
    Bp = result["Bp"]
    history = result["history"]

    rel_err = np.linalg.norm(u_pred - u_obs) / max(np.linalg.norm(u_obs), 1e-14)

    print(f"\nRelative coarse error = {rel_err:.6e}")
    print(f"Number of coarse nodes = {n_nodes}")
    print(f"Number of coarse edges = {len(edges)}")

    plot_fields(X, Y, a, u_fine, u_obs, u_pred, cx, cy, out_dir)
    plot_weights(edges, w_pred, cx, cy, out_dir)
    plot_history(history, out_dir)
    plot_coarse_node_fields(u_obs, u_pred, p, cx, cy, out_dir)

    plot_graph_edge_values(
        edges, Bu, cx, cy,
        "discrete forward gradient $Bu$",
        os.path.join(out_dir, "edge_forward_gradient.png")
    )
    plot_graph_edge_values(
        edges, Bp, cx, cy,
        "discrete adjoint gradient $Bp$",
        os.path.join(out_dir, "edge_adjoint_gradient.png")
    )
    plot_graph_edge_values(
        edges, sens, cx, cy,
        "graph Eulerian / topological sensitivity",
        os.path.join(out_dir, "edge_sensitivity.png")
    )

    # ranking
    order = np.argsort(-np.abs(sens))
    print("\nTop 10 most sensitive edges (by |dJ/dw_e|):")
    for k in range(min(10, len(edges))):
        e = order[k]
        print(f"  edge {e:2d} {edges[e]} | sensitivity = {sens[e]: .6e}")

    print(f"\nResults saved in: {out_dir}")

    return {
        "X": X,
        "Y": Y,
        "a": a,
        "u_fine": u_fine,
        "u_obs": u_obs,
        "u_pred": u_pred,
        "p": p,
        "edges": edges,
        "weights": w_pred,
        "sens": sens,
        "history": history,
    }


if __name__ == "__main__":
    run_ddec_scalar_poisson_with_sensitivity()