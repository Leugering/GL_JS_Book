from __future__ import annotations

"""
Cantilever truss topology optimization as a full (hybrid) MINLP in Pyomo.

This script formulates the standard compliance minimization problem on a
2D cantilever ground structure with
    - binary edge-presence variables z[e,t] in {0,1}
    - continuous cross-sectional areas a[e,t] >= 0
    - nodal displacements u[d,t]

The model is a genuine MINLP because equilibrium contains bilinear terms
between design variables and displacements:
    K(z,a) u = f.

A hybrid/time-staged extension is included:
    - multiple stages t = 0,...,T-1
    - optional switching penalty on z[e,t]-z[e,t-1]
    - optional stage-to-stage area variation penalty

Recommended solver stack
------------------------
Pyomo itself is just the modeling layer. For actual solution install, e.g.:
    pip install pyomo
    conda install -c conda-forge ipopt bonmin couenne glpk
or use SCIP / Knitro / BARON if available.

For Pyomo's MindtPy decomposition one typically uses:
    - MILP master: glpk / cbc / gurobi / highs
    - NLP subproblem: ipopt

If no MINLP solver is available, this script still provides a clear and
research-grade formulation that can be adapted to your environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyomo.environ as pyo
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pyomo is required for this script. Install with `pip install pyomo`."
    ) from exc


# ============================================================
# Problem data
# ============================================================

@dataclass
class CantileverData:
    nx: int = 5
    ny: int = 3
    young: float = 1.0
    a_min: float = 0.05
    a_max: float = 1.0
    volume_frac: float = 0.35
    n_stages: int = 1
    switch_penalty: float = 0.05
    area_rate_penalty: float = 0.0
    enforce_same_topology_across_stages: bool = False

    def __post_init__(self) -> None:
        self.coords = np.array([[i, j] for j in range(self.ny) for i in range(self.nx)], dtype=float)
        self.n_nodes = len(self.coords)
        self.ndof = 2 * self.n_nodes
        self.edges = self._build_ground_structure()
        self.n_edges = len(self.edges)
        self.lengths = np.array([self._edge_length(e) for e in self.edges], dtype=float)
        self.ke_unit = [self._element_stiffness_unit(e) for e in self.edges]
        self.fixed = self._fixed_dofs()
        self.free = [d for d in range(self.ndof) if d not in self.fixed]
        self.loads = self._build_loads()
        self.volume_max = self.volume_frac * np.sum(self.lengths) * self.a_max

    def _build_ground_structure(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                # local neighborhood ground structure
                if np.linalg.norm(self.coords[i] - self.coords[j]) <= np.sqrt(2.0) + 1.0e-9:
                    edges.append((i, j))
        return edges

    def _edge_length(self, edge: Tuple[int, int]) -> float:
        i, j = edge
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        return float(np.hypot(xj - xi, yj - yi))

    def _element_stiffness_unit(self, edge: Tuple[int, int]) -> np.ndarray:
        i, j = edge
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        L = np.hypot(xj - xi, yj - yi)
        c = (xj - xi) / L
        s = (yj - yi) / L
        return (self.young / L) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ],
            dtype=float,
        )

    def _fixed_dofs(self) -> List[int]:
        fixed: List[int] = []
        for i, (x, _) in enumerate(self.coords):
            if x == 0:
                fixed.extend([2 * i, 2 * i + 1])
        return fixed

    def _build_loads(self) -> np.ndarray:
        loads = np.zeros((self.n_stages, self.ndof), dtype=float)
        # Standard cantilever load at top-right node for stage 0.
        for i, (x, y) in enumerate(self.coords):
            if x == self.nx - 1 and y == self.ny - 1:
                loads[0, 2 * i + 1] = -1.0
        # If multiple stages, add mild load variation to demonstrate hybrid setup.
        for t in range(1, self.n_stages):
            loads[t] = loads[0]
            # alternate a small horizontal component at the loaded node
            for i, (x, y) in enumerate(self.coords):
                if x == self.nx - 1 and y == self.ny - 1:
                    loads[t, 2 * i] = 0.2 * ((-1) ** t)
        return loads


@dataclass
class SolveReport:
    backend: str
    termination: str
    is_true_minlp: bool
    is_optimal: bool
    note: str = ""


# ============================================================
# Model construction
# ============================================================

def build_minlp_model(data: CantileverData) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name="cantilever_truss_hybrid_minlp")

    # Sets
    m.E = pyo.RangeSet(0, data.n_edges - 1)
    m.T = pyo.RangeSet(0, data.n_stages - 1)
    m.D = pyo.RangeSet(0, data.ndof - 1)
    m.FREE = pyo.Set(initialize=data.free, ordered=True)

    # Parameters
    edge_i = {e: data.edges[e][0] for e in range(data.n_edges)}
    edge_j = {e: data.edges[e][1] for e in range(data.n_edges)}
    dofs = {e: [2 * edge_i[e], 2 * edge_i[e] + 1, 2 * edge_j[e], 2 * edge_j[e] + 1] for e in range(data.n_edges)}

    m.length = pyo.Param(m.E, initialize={e: float(data.lengths[e]) for e in range(data.n_edges)})
    m.fext = pyo.Param(m.T, m.D, initialize={(t, d): float(data.loads[t, d]) for t in range(data.n_stages) for d in range(data.ndof)})

    # Unit stiffness entries per edge and local dof pair
    ke_map: Dict[Tuple[int, int, int], float] = {}
    for e in range(data.n_edges):
        ke = data.ke_unit[e]
        for a in range(4):
            for b in range(4):
                ke_map[(e, a, b)] = float(ke[a, b])
    m.ke = pyo.Param(m.E, pyo.RangeSet(0, 3), pyo.RangeSet(0, 3), initialize=ke_map)

    # Variables
    m.z = pyo.Var(m.E, m.T, domain=pyo.Binary)                 # topology
    m.a = pyo.Var(m.E, m.T, bounds=(0.0, data.a_max))          # area
    m.u = pyo.Var(m.D, m.T, domain=pyo.Reals)                  # displacement
    m.c = pyo.Var(m.T, domain=pyo.Reals)                       # compliance per stage

    # Optional switch auxiliaries
    if data.n_stages > 1:
        m.s = pyo.Var(m.E, pyo.RangeSet(1, data.n_stages - 1), domain=pyo.Binary)

    # Boundary conditions
    def fixed_bc_rule(mm: pyo.ConcreteModel, d: int, t: int):
        if d in data.fixed:
            return mm.u[d, t] == 0.0
        return pyo.Constraint.Skip
    m.fixed_bc = pyo.Constraint(m.D, m.T, rule=fixed_bc_rule)

    # Linking constraints: if edge absent then area zero, if edge present area >= a_min
    def area_upper_rule(mm: pyo.ConcreteModel, e: int, t: int):
        return mm.a[e, t] <= data.a_max * mm.z[e, t]
    m.area_upper = pyo.Constraint(m.E, m.T, rule=area_upper_rule)

    def area_lower_rule(mm: pyo.ConcreteModel, e: int, t: int):
        return mm.a[e, t] >= data.a_min * mm.z[e, t]
    m.area_lower = pyo.Constraint(m.E, m.T, rule=area_lower_rule)

    # Volume constraint stagewise
    def volume_rule(mm: pyo.ConcreteModel, t: int):
        return sum(mm.length[e] * mm.a[e, t] for e in mm.E) <= data.volume_max
    m.volume = pyo.Constraint(m.T, rule=volume_rule)

    # Optional same topology across stages
    if data.enforce_same_topology_across_stages and data.n_stages > 1:
        def same_topology_rule(mm: pyo.ConcreteModel, e: int, t: int):
            if t == 0:
                return pyo.Constraint.Skip
            return mm.z[e, t] == mm.z[e, 0]
        m.same_topology = pyo.Constraint(m.E, m.T, rule=same_topology_rule)

    # Switching TV penalty linearization
    if data.n_stages > 1:
        def switch_pos_rule(mm: pyo.ConcreteModel, e: int, t: int):
            return mm.z[e, t] - mm.z[e, t - 1] <= mm.s[e, t]
        def switch_neg_rule(mm: pyo.ConcreteModel, e: int, t: int):
            return mm.z[e, t - 1] - mm.z[e, t] <= mm.s[e, t]
        m.switch_pos = pyo.Constraint(m.E, pyo.RangeSet(1, data.n_stages - 1), rule=switch_pos_rule)
        m.switch_neg = pyo.Constraint(m.E, pyo.RangeSet(1, data.n_stages - 1), rule=switch_neg_rule)

    # Equilibrium constraints on free dofs: sum_e a_e * K_e * u = f
    # This is bilinear in (a,u) -> full MINLP.
    def equilibrium_rule(mm: pyo.ConcreteModel, d: int, t: int):
        if d not in data.free:
            return pyo.Constraint.Skip
        expr = 0.0
        for e in range(data.n_edges):
            local_dofs = dofs[e]
            if d in local_dofs:
                a_idx = local_dofs.index(d)
                for b_idx, db in enumerate(local_dofs):
                    expr += mm.a[e, t] * mm.ke[e, a_idx, b_idx] * mm.u[db, t]
        return expr == mm.fext[t, d]
    m.equilibrium = pyo.Constraint(m.D, m.T, rule=equilibrium_rule)

    # Compliance definition c_t = f_t^T u_t
    def compliance_rule(mm: pyo.ConcreteModel, t: int):
        return mm.c[t] == sum(mm.fext[t, d] * mm.u[d, t] for d in mm.D)
    m.compliance_def = pyo.Constraint(m.T, rule=compliance_rule)

    # Objective: stage compliance + optional switching penalty + optional area-rate penalty
    expr = sum(m.c[t] for t in m.T)
    if data.n_stages > 1 and data.switch_penalty > 0.0:
        expr += data.switch_penalty * sum(m.s[e, t] for e in m.E for t in range(1, data.n_stages))
    if data.n_stages > 1 and data.area_rate_penalty > 0.0:
        # quadratic absolute-value surrogate using squares -> still nonlinear, but acceptable in MINLP/NLP context
        expr += data.area_rate_penalty * sum((m.a[e, t] - m.a[e, t - 1]) ** 2 for e in m.E for t in range(1, data.n_stages))
    m.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

    return m


# ============================================================
# Solving utilities
# ============================================================

def initialize_design(model: pyo.ConcreteModel, data: CantileverData) -> None:
    for e in model.E:
        for t in model.T:
            model.z[e, t].value = 1
            model.a[e, t].value = min(data.a_max, max(data.a_min, 0.5 * data.a_max))
    for d in model.D:
        for t in model.T:
            model.u[d, t].value = 0.0


def solve_model(model: pyo.ConcreteModel, prefer_mindtpy: bool = True) -> SolveReport:
    """Attempt to solve the MINLP with common Pyomo strategies.

    Order:
    1. MindtPy (true MINLP OA with GLPK + IPOPT)
    2. Direct MINLP solvers (bonmin / couenne / scip)
    3. IPOPT only as an explicit NLP relaxation fallback
    """

    def _available(name: str) -> bool:
        try:
            s = pyo.SolverFactory(name)
            return s is not None and s.available(exception_flag=False)
        except Exception:
            return False

    have_glpk = _available("glpk")
    have_ipopt = _available("ipopt")

    # MindtPy decomposition: only call if BOTH MIP and NLP solvers are available.
    if prefer_mindtpy and have_glpk and have_ipopt:
        try:
            solver = pyo.SolverFactory("mindtpy")
            if solver is not None and solver.available(exception_flag=False):
                res = solver.solve(
                    model,
                    strategy="OA",
                    mip_solver="glpk",
                    nlp_solver="ipopt",
                    init_strategy="rNLP",
                    iteration_limit=20,
                    add_no_good_cuts=True,
                    integer_to_binary=False,
                    mip_solver_tee=True,
                    nlp_solver_tee=True,
                    tee=True,
                )
                term = str(res.solver.termination_condition)
                return SolveReport(
                    backend="mindtpy",
                    termination=term,
                    is_true_minlp=True,
                    is_optimal=term.lower() == "optimal",
                    note="OA decomposition with GLPK master and IPOPT NLP subproblems.",
                )
        except Exception as exc:
            print(f"MindtPy failed, falling back. Reason: {exc}")

    # Direct MINLP solvers first.
    for name in ["bonmin", "couenne", "scip"]:
        try:
            solver = pyo.SolverFactory(name)
            if solver is not None and solver.available(exception_flag=False):
                res = solver.solve(model, tee=True)
                term = str(res.solver.termination_condition)
                return SolveReport(
                    backend=name,
                    termination=term,
                    is_true_minlp=True,
                    is_optimal=term.lower() == "optimal",
                )
        except Exception:
            continue

    # Explicit NLP relaxation fallback (not a true discrete solution).
    if have_ipopt:
        print("WARNING: No discrete MINLP solver path succeeded; solving NLP relaxation with IPOPT.")
        print("WARNING: Binary variables will be treated as continuous in [0,1].")
        solver = pyo.SolverFactory("ipopt")
        res = solver.solve(model, tee=True)
        return SolveReport(
            backend="ipopt-relaxation",
            termination=str(res.solver.termination_condition),
            is_true_minlp=False,
            is_optimal=False,
            note="Continuous fallback only; this is not a discrete topology solve.",
        )

    raise RuntimeError(
        "No suitable MINLP/NLP solver found. Install Pyomo plus one of: "
        "MindtPy+GLPK+IPOPT, BONMIN, COUENNE, or SCIP."
    )


# ============================================================
# Post-processing and plots
# ============================================================

def extract_stage_solution(model: pyo.ConcreteModel, t: int, n_edges: int, ndof: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.array([pyo.value(model.z[e, t]) for e in range(n_edges)], dtype=float)
    a = np.array([pyo.value(model.a[e, t]) for e in range(n_edges)], dtype=float)
    u = np.array([pyo.value(model.u[d, t]) for d in range(ndof)], dtype=float)
    return z, a, u


def plot_design(ax: plt.Axes, coords: np.ndarray, edges: Sequence[Tuple[int, int]], z: np.ndarray, a: np.ndarray, title: str, show_all: bool = False) -> None:
    for e, (i, j) in enumerate(edges):
        xi, yi = coords[i]
        xj, yj = coords[j]
        if show_all:
            ax.plot([xi, xj], [yi, yj], color="0.82", linewidth=0.9, alpha=0.8)
        elif z[e] >= 0.5:
            lw = 1.0 + 3.0 * float(a[e]) / max(np.max(a), 1e-8)
            ax.plot([xi, xj], [yi, yj], color="#2b6cb0", linewidth=lw, alpha=0.95)
    ax.scatter(coords[:, 0], coords[:, 1], c="k", s=18, zorder=3)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)


# ============================================================
# Main driver
# ============================================================

def main() -> None:
    # With the locally available solver stack (MindtPy + GLPK + IPOPT),
    # the 5x3 ground structure is often too hard to certify reliably.
    # The 4x3 instance below is still a genuine MINLP and solves cleanly.
    data = CantileverData(
        nx=4,
        ny=3,
        volume_frac=0.40,
        n_stages=1,
        switch_penalty=0.05,
        area_rate_penalty=0.0,
        enforce_same_topology_across_stages=False,
    )

    model = build_minlp_model(data)
    initialize_design(model, data)
    report = solve_model(model)
    print("Solver status:", f"{report.backend}: {report.termination}")

    zf, af, uf = extract_stage_solution(model, t=0, n_edges=data.n_edges, ndof=data.ndof)
    fractional_count = int(np.sum((zf > 1.0e-6) & (zf < 1.0 - 1.0e-6)))
    if (not report.is_true_minlp) or (not report.is_optimal) or fractional_count > 0:
        raise RuntimeError(
            "True MINLP not solved to an integral optimal solution. "
            f"backend={report.backend}, termination={report.termination}, "
            f"fractional_binaries={fractional_count}."
        )
    # # --------------------------------------------------
    # # Extract design
    # # --------------------------------------------------
    # z_sol = np.array([pyo.value(model.z[e]) for e in model.E])
    # z_bin = (z_sol >= 0.5).astype(int)

    # print("Binary design summary:")
    # print("min(z) =", z_sol.min())
    # print("max(z) =", z_sol.max())
    # print("non-binary count =", np.sum((z_sol > 1e-6) & (z_sol < 1 - 1e-6)))
    # print("active edges =", int(np.sum(z_bin)))

    # # --------------------------------------------------
    # # Plot ground structure vs optimized design
    # # --------------------------------------------------
    # plt.figure(figsize=(10, 4))

    # plt.subplot(1, 2, 1)
    # for e, (i, j) in enumerate(edges):
    #     xi, yi = coords[i]
    #     xj, yj = coords[j]
    #     plt.plot([xi, xj], [yi, yj], color="lightgray", linewidth=1)
    # plt.scatter(coords[:, 0], coords[:, 1], c="k", s=20)
    # plt.title("Ground structure")
    # plt.axis("equal")
    # plt.grid(True, alpha=0.3)

    # plt.subplot(1, 2, 2)
    # for e, (i, j) in enumerate(edges):
    #     xi, yi = coords[i]
    #     xj, yj = coords[j]
    #     if z_bin[e] == 1:
    #         plt.plot([xi, xj], [yi, yj], "b-", linewidth=2)
    #     else:
    #         plt.plot([xi, xj], [yi, yj], color="lightgray", linewidth=0.7, alpha=0.25)
    # plt.scatter(coords[:, 0], coords[:, 1], c="k", s=20)
    # plt.title("Optimized topology")
    # plt.axis("equal")
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()
    out_dir = Path(__file__).resolve().parent / "results_cantilever_truss_hybrid_minlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initial and final plots for stage 0.
    z0 = np.ones(data.n_edges)
    a0 = data.a_max * np.ones(data.n_edges)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_design(axes[0], data.coords, data.edges, z0, a0, "Initial ground structure", show_all=True)
    plot_design(axes[1], data.coords, data.edges, zf, af, f"Final MINLP design ({int(np.sum(zf >= 0.5))} active edges)")
    fig.tight_layout()
    fig.savefig(out_dir / "cantilever_truss_hybrid_minlp_designs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # If several stages are used, also plot stagewise results.
    if data.n_stages > 1:
        fig2, axes2 = plt.subplots(1, data.n_stages, figsize=(4 * data.n_stages, 4))
        if data.n_stages == 1:
            axes2 = [axes2]
        for t in range(data.n_stages):
            zt, at, _ = extract_stage_solution(model, t=t, n_edges=data.n_edges, ndof=data.ndof)
            plot_design(axes2[t], data.coords, data.edges, zt, at, f"Stage {t}")
        fig2.tight_layout()
        fig2.savefig(out_dir / "cantilever_truss_hybrid_minlp_stages.png", dpi=180, bbox_inches="tight")
        plt.close(fig2)

    # Compliance summary
    compliance_vals = [float(pyo.value(model.c[t])) for t in model.T]
    with (out_dir / "cantilever_truss_hybrid_minlp_summary.txt").open("w", encoding="utf-8") as fsum:
        fsum.write("Cantilever truss hybrid MINLP summary\n")
        fsum.write(f"n_nodes: {data.n_nodes}\n")
        fsum.write(f"n_edges: {data.n_edges}\n")
        fsum.write(f"n_stages: {data.n_stages}\n")
        fsum.write(f"volume_max: {data.volume_max:.6f}\n")
        fsum.write(f"solver_backend: {report.backend}\n")
        fsum.write(f"solver_termination: {report.termination}\n")
        fsum.write(f"is_true_minlp: {report.is_true_minlp}\n")
        fsum.write(f"is_optimal: {report.is_optimal}\n")
        if report.note:
            fsum.write(f"solver_note: {report.note}\n")
        fsum.write(f"stage_compliances: {compliance_vals}\n")
        fsum.write(f"active_edges_stage0: {int(np.sum(zf >= 0.5))}\n")
        fsum.write(f"fractional_binaries_stage0: {fractional_count}\n")

    np.savetxt(out_dir / "cantilever_truss_hybrid_minlp_z_stage0.txt", zf, fmt="%.6f")
    np.savetxt(out_dir / "cantilever_truss_hybrid_minlp_a_stage0.txt", af, fmt="%.6f")
    np.savetxt(out_dir / "cantilever_truss_hybrid_minlp_u_stage0.txt", uf, fmt="%.6f")

    print("Results written to:", out_dir)
    print("Stage-0 compliances:", compliance_vals)
   

if __name__ == "__main__":
    main()
