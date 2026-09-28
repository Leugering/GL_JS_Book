"""
Topological derivative of the TOTAL STRAIN ENERGY for a Lewinski-type
Timoshenko 4-star with a small inserted hole subgraph (cycle C4 or K4),
Dirichlet data at the outer vertices (book chapter chap:example_star).

For prescribed end displacements g the physically relevant objective is
    E(delta) = min_U 1/2 W^T K^delta W,   W = (U, g),
i.e. the total strain energy of all beams (shortened arms + interconnections).
Its topological derivative is (Ch. 15, thm:topo_deriv with lifting)
    TD = short + shear + rel,
    short = -1/2 sum_i rho_i d_i^T dK/dL(L_i) d_i,       d_i = (u_c, g_i)
    shear = 1/2 sum_(i,j) tau_ij [G As_z th_y^2 + G As_y th_z^2]
    rel   = -1/2 sum_i r_i^T u1_i,   Lap(S_hat) u1 = r
The script
  * checks TD against Richardson-extrapolated divided differences of the
    exact perturbed energy,
  * checks the linear scaling of TD in rho,
  * writes figures and a LaTeX table to results_timoshenko_star_hole_energy/.

Replaces the objective 1/2 U^T K U - U^T f + sum_arms 1/2 d^T K d used in
Timoshenko_Lewinski_top_derivative*.py, which is not a physical energy
(it omits 1/2 g^T K_gg(delta) g and the interconnection energy).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_timoshenko_star_hole_energy")
os.makedirs(OUT, exist_ok=True)

# ---------------- data (steel, square section 0.1 m) ----------------
E = 2.1e11; nu = 0.3; G = E / (2 * (1 + nu))
A = 1.0e-2; Iy = Iz = 8.333e-6; J = 1.406e-5; kappa = 0.85; L = 1.0
As_y = As_z = kappa * A
n = 4
e_dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)
CYCLE = [(0, 1), (1, 2), (2, 3), (3, 0)]
K4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]


# ---------------- exact (Lewinski) element, book sign convention ----------------
def lewinski(l):
    Phy = 12 * E * Iz / (G * As_y * l ** 2); Phz = 12 * E * Iy / (G * As_z * l ** 2)
    K = np.zeros((12, 12))
    for (a, b), k in (((0, 6), E * A / l), ((3, 9), G * J / l)):
        K[a, a] += k; K[b, b] += k; K[a, b] -= k; K[b, a] -= k
    fz = E * Iy / (l ** 3 * (1 + Phz))
    Kbz = fz * np.array([[12, 6 * l, -12, 6 * l], [6 * l, (4 + Phz) * l ** 2, -6 * l, (2 - Phz) * l ** 2],
                         [-12, -6 * l, 12, -6 * l], [6 * l, (2 - Phz) * l ** 2, -6 * l, (4 + Phz) * l ** 2]])
    fy = E * Iz / (l ** 3 * (1 + Phy))
    Kby = fy * np.array([[12, -6 * l, -12, -6 * l], [-6 * l, (4 + Phy) * l ** 2, 6 * l, (2 - Phy) * l ** 2],
                         [-12, 6 * l, 12, 6 * l], [-6 * l, (2 - Phy) * l ** 2, 6 * l, (4 + Phy) * l ** 2]])
    for r, Kb in (([2, 4, 8, 10], Kbz), ([1, 5, 7, 11], Kby)):
        K[np.ix_(r, r)] += Kb
    return K


def frame(ex):
    ex = ex / np.linalg.norm(ex)
    tmp = np.array([1.0, 0, 0]) if abs(ex[0]) < 0.9 else np.array([0, 1.0, 0])
    ey = tmp - tmp @ ex * ex; ey /= np.linalg.norm(ey)
    return np.vstack([ex, ey, np.cross(ex, ey)])


def Kg(ex, l):
    R = frame(ex); T = np.kron(np.eye(4), R)
    return T.T @ lewinski(l) @ T


def dKdL(ex, l, h=1e-5):
    return (Kg(ex, l * (1 + h)) - Kg(ex, l * (1 - h))) / (2 * l * h)


# ---------------- unperturbed and exact perturbed problems ----------------
def unperturbed(g):
    Ks = [Kg(e_dirs[i], L) for i in range(n)]
    K = sum(k[:6, :6] for k in Ks)
    f = -sum(Ks[i][:6, 6:] @ g[i] for i in range(n))
    uc = np.linalg.solve(K, f)
    E0 = sum(0.5 * np.r_[uc, g[i]] @ Ks[i] @ np.r_[uc, g[i]] for i in range(n))
    return uc, E0


def exact_energy(delta, rho, g, edges):
    rho = np.broadcast_to(rho, (n,)).astype(float)
    Ks = [Kg(e_dirs[i], L - delta * rho[i]) for i in range(n)]
    K = np.zeros((24, 24)); f = np.zeros(24)
    for i in range(n):
        K[6 * i:6 * i + 6, 6 * i:6 * i + 6] += Ks[i][:6, :6]
        f[6 * i:6 * i + 6] -= Ks[i][:6, 6:] @ g[i]
    Kij = []
    for (i, j) in edges:
        d = delta * (rho[j] * e_dirs[j] - rho[i] * e_dirs[i]); l = np.linalg.norm(d)
        k = Kg(d / l, l); Kij.append(k)
        idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        K[np.ix_(idx, idx)] += k
    U = np.linalg.solve(K, f)
    Ea = sum(0.5 * np.r_[U[6 * i:6 * i + 6], g[i]] @ Ks[i] @ np.r_[U[6 * i:6 * i + 6], g[i]] for i in range(n))
    Ei = sum(0.5 * np.r_[U[6 * i:6 * i + 6], U[6 * j:6 * j + 6]] @ k @ np.r_[U[6 * i:6 * i + 6], U[6 * j:6 * j + 6]]
             for (i, j), k in zip(edges, Kij))
    return Ea + Ei


# ---------------- asymptotic topological derivative ----------------
def S_hat(i, j, rho):
    d = rho[j] * e_dirs[j] - rho[i] * e_dirs[i]; tau = np.linalg.norm(d); R = frame(d / tau)
    S = np.diag([E * A, G * As_y, G * As_z, G * J, E * Iy, E * Iz]) / tau
    Q = np.kron(np.eye(2), R)
    return Q.T @ S @ Q, tau, R


def O1_force(i, j, rho, uc):
    """O(1) coefficient K^(0)_ij (uc,uc) of the interconnection, by Richardson in delta."""
    w = np.r_[uc, uc]
    F = []
    for dl in (4e-4, 2e-4, 1e-4):
        d = dl * (rho[j] * e_dirs[j] - rho[i] * e_dirs[i]); l = np.linalg.norm(d)
        F.append(Kg(d / l, l) @ w)
    a, b = 2 * F[1] - F[0], 2 * F[2] - F[1]
    return (4 * b - a) / 3


def topological_derivative(rho, g, edges):
    rho = np.broadcast_to(rho, (n,)).astype(float)
    uc, E0 = unperturbed(g)
    Ks = [Kg(e_dirs[i], L) for i in range(n)]
    short = sum(-0.5 * rho[i] * np.r_[uc, g[i]] @ dKdL(e_dirs[i], L) @ np.r_[uc, g[i]] for i in range(n))
    shear = 0.0
    r = np.array([-(Ks[i] @ np.r_[uc, g[i]])[:6] for i in range(n)])
    Lap = np.zeros((24, 24))
    for (i, j) in edges:
        S, tau, R = S_hat(i, j, rho)
        th = -R @ uc[3:]                      # right-handed rotation in the local frame
        shear += 0.5 * tau * (G * As_z * th[1] ** 2 + G * As_y * th[2] ** 2)
        F = O1_force(i, j, rho, uc); r[i] -= F[:6]; r[j] -= F[6:]
        a, b = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
        Lap[a, a] += S; Lap[b, b] += S; Lap[a, b] -= S; Lap[b, a] -= S
    u1 = np.linalg.lstsq(Lap, r.ravel(), rcond=None)[0]
    rel = -0.5 * r.ravel() @ u1
    return dict(E0=E0, uc=uc, short=short, shear=shear, rel=rel, TD=short + shear + rel)


def richardson_dd(rho, g, edges, E0, h=4e-4):
    d = [(exact_energy(h / 2 ** k, rho, g, edges) - E0) / (h / 2 ** k) for k in range(3)]
    a, b = 2 * d[1] - d[0], 2 * d[2] - d[1]
    return (4 * b - a) / 3


if __name__ == "__main__":
    g0 = np.zeros(6); gz = np.array([0, 0, 0.01, 0, 0, 0.]); gth = np.array([0, 0, 0, 0, 0.01, 0.])
    cases = {"LC1": [gz] * 4, "LC2": [gz, -gz, -gz, gz], "LC3": [gth, g0, g0, g0]}
    rho = 0.5
    deltas = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    res = {}
    for sg, edges in (("C4", CYCLE), ("K4", K4)):
        for c, g in cases.items():
            a = topological_derivative(rho, g, edges)
            dd = np.array([(exact_energy(dl, rho, g, edges) - a["E0"]) / dl for dl in deltas])
            a["dd"] = dd; a["rich"] = richardson_dd(rho, g, edges, a["E0"])
            a["scale"] = {s: topological_derivative(s, g, edges)["TD"] for s in (0.25, 1.0)}
            res[(sg, c)] = a
            print(f"{sg} {c}: E0={a['E0']:.6e} short={a['short']:.6e} shear={a['shear']:.6e} "
                  f"rel={a['rel']:.6e} TD={a['TD']:.9e} Richardson={a['rich']:.9e} "
                  f"TD(0.25)={a['scale'][0.25]:.6e} TD(1)={a['scale'][1.0]:.6e}")

    # convergence figure
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    mk = {"C4": "o", "K4": "s"}; ls = {"LC2": "-", "LC3": "--"}
    for (sg, c), a in res.items():
        if c == "LC1":
            continue
        ax.loglog(deltas, np.abs(a["dd"] - a["TD"]) / abs(a["TD"]), mk[sg] + ls[c], label=f"{c}, {sg}")
    ax.loglog(deltas, deltas, "k:", label=r"$O(\delta)$")
    ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$|(\mathcal{E}^\delta-\mathcal{E}^0)/\delta-\nabla_T\mathcal{E}|/|\nabla_T\mathcal{E}|$")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence.pdf")); plt.close(fig)

    # decomposition figure
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, c in zip(axs, ("LC2", "LC3")):
        x = np.arange(4); w = 0.38
        for k, sg in enumerate(("C4", "K4")):
            a = res[(sg, c)]
            ax.bar(x + (k - 0.5) * w, [a["short"], a["shear"], a["rel"], a["TD"]], w, label=sg)
        ax.set_xticks(x); ax.set_xticklabels(["short", "shear", "rel", "total"])
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"Load case {c[-1]}"); ax.legend(fontsize=8)
        ax.set_ylabel(r"contribution to $\nabla_T\mathcal{E}$ [J/m]")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "decomposition.pdf")); plt.close(fig)

    # geometry figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 3.6))
    for k, (title, edges, dvis) in enumerate((("unperturbed 4-star", [], 0.0),
                                              ("cycle $C_4$ inserted", CYCLE, 0.45),
                                              ("complete graph $K_4$ inserted", K4, 0.45))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for i in range(n):
            p0 = dvis * e_dirs[i]; p1 = L * e_dirs[i]
            ax.plot(*np.c_[p0, p1], color="0.3", lw=2)
            ax.scatter(*p1, color="k", s=18)
        for (i, j) in edges:
            ax.plot(*np.c_[dvis * e_dirs[i], dvis * e_dirs[j]], color="crimson", lw=2)
        ax.scatter(*(dvis * e_dirs.T), color="crimson" if edges else "k", s=14)
        ax.set_title(title, fontsize=10); ax.set_box_aspect((1, 1, 1))
        for s in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
            s([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "star_geometry.pdf")); plt.close(fig)

    # LaTeX table
    with open(os.path.join(OUT, "td_table.tex"), "w") as fh:
        for (sg, c), a in res.items():
            fh.write(f"{c} & ${sg}$ & {a['short']:.6e} & {a['shear']:.6e} & {a['rel']:.6e} & "
                     f"{a['TD']:.9e} & {a['rich']:.9e} \\\\\n")
    print("results in", OUT)
