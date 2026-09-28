from __future__ import annotations

"""
Joint inverse pipeline for switching-fatigue damage identification on a spring network.

Model
-----
Forward dynamics:
    M q_ddot + C_damp q_dot + K(delta, d) q = f(t)
with
    K(delta, d) = sum_e delta_e(t) * (1 - d_e(t)) * k_e * (b_e b_e^T).

Damage law:
    d_dot_e = alpha_e (1 - d_e)
            + beta_e (1 - d_e) * < (eps_e(q) - eps_safe_e)/eps_crit_e >_+^2

Unknowns to recover from measurements:
    - switching variables delta_e^n in {0,1}
    - damage trajectories d_e^n in [0,1]

Pipeline
--------
1. Simulate synthetic data with the forward model.
2. Recover effective coefficients alpha_eff_e^n ~= delta_e^n * (1 - d_e^n)
   by projected gradient descent on a least-squares objective.
3. Infer delta_e^n from alpha_eff using thresholding + temporal regularization.
4. Reconstruct d_e^n from alpha_eff and delta_e^n.
5. Optionally refine delta by local search.

This is a research prototype intended to be readable and modifiable.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
from pathlib import Path
from itertools import product
import numpy as np

Array = np.ndarray
Edge = Tuple[int, int]


# ============================================================
# Utilities
# ============================================================


def positive_part(x: Array | float) -> Array | float:
    return np.maximum(x, 0.0)



def moving_average(x: Array, window: int) -> Array:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], kernel, mode="same")
    return out


def normalize_edgewise_signal(
    x: Array,
    percentile: float = 90.0,
    floor: float = 1.0e-6,
) -> Tuple[Array, Array]:
    """Normalize each edge time series by a robust high-percentile scale."""
    Tm1, m = x.shape
    normalized = np.zeros_like(x)
    scales = np.zeros(m, dtype=float)
    for e in range(m):
        scale = float(np.percentile(x[:, e], percentile))
        scale = max(scale, floor)
        normalized[:, e] = x[:, e] / scale
        scales[e] = scale
    return normalized, scales


def complete_state_from_observations(
    model: NetworkModel,
    q_obs: Array,
    observed_idx: Array,
) -> Array:
    """Complete hidden nodal states from observed ones by harmonic extension.

    On a connected graph with observed leaves fixed, the hidden internal nodes
    are reconstructed as the minimizer of the all-active elastic energy.
    """
    observed_idx = np.asarray(observed_idx, dtype=int)
    if q_obs.shape[1] == model.n_nodes and observed_idx.size == model.n_nodes:
        return q_obs.copy()

    all_idx = np.arange(model.n_nodes, dtype=int)
    hidden_idx = np.array([i for i in all_idx if i not in set(observed_idx.tolist())], dtype=int)
    K_ref = model.assemble_stiffness(np.ones(model.n_edges), np.zeros(model.n_edges))

    q_full = np.zeros((q_obs.shape[0], model.n_nodes), dtype=float)
    q_full[:, observed_idx] = q_obs
    if hidden_idx.size == 0:
        return q_full

    K_uu = K_ref[np.ix_(hidden_idx, hidden_idx)]
    K_uo = K_ref[np.ix_(hidden_idx, observed_idx)]
    K_uu = K_uu + 1.0e-8 * np.eye(hidden_idx.size)
    for k in range(q_obs.shape[0]):
        rhs = -K_uo @ q_obs[k]
        q_full[k, hidden_idx] = np.linalg.solve(K_uu, rhs)
    return q_full


def as_experiment_list(x: Array | Sequence[Array]) -> List[Array]:
    if isinstance(x, np.ndarray):
        return [x]
    return list(x)


def edge_failure_weights(
    model: NetworkModel,
    observed_idx: Array,
    leaf_weight: float = 0.4,
    internal_weight: float = 1.2,
) -> Array:
    """Prefer failures on observed leaf edges over internal edges."""
    observed_set = set(np.asarray(observed_idx, dtype=int).tolist())
    weights = internal_weight * np.ones(model.n_edges, dtype=float)
    for e, (i, j) in enumerate(model.edges):
        if i in observed_set or j in observed_set:
            weights[e] = leaf_weight
    return weights


def default_tree_positions(n_nodes: int) -> Dict[int, Tuple[float, float]]:
    if n_nodes == 8:
        return {
            0: (0.0, 0.0),
            1: (1.2, 0.0),
            2: (2.4, 0.0),
            3: (-0.8, 1.0),
            4: (-0.8, -1.0),
            5: (1.2, 1.05),
            6: (3.2, 1.0),
            7: (3.2, -1.0),
        }
    return {i: (float(i), 0.0) for i in range(n_nodes)}


def draw_tree_state(
    ax,
    model: NetworkModel,
    positions: Dict[int, Tuple[float, float]],
    observed_idx: Array,
    failure_times: Array,
    final_damage: Array,
    Tm1: int,
    title: str,
) -> None:
    observed_set = set(np.asarray(observed_idx, dtype=int).tolist())

    for e, (i, j) in enumerate(model.edges):
        xi, yi = positions[i]
        xj, yj = positions[j]
        failed = int(failure_times[e]) < Tm1
        damage = float(final_damage[e])
        if failed:
            color = "#c53a2f"
            lw = 3.2
            ls = "--"
        else:
            color = "#4f7db8" if damage < 0.6 else "#c98624"
            lw = 1.8 + 2.2 * damage
            ls = "-"
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=1)

    for i in range(model.n_nodes):
        x, y = positions[i]
        is_leaf_measured = i in observed_set
        face = "#2c7fb8" if is_leaf_measured else "#f7f7f7"
        edge = "#1b1b1b"
        size = 120 if is_leaf_measured else 90
        ax.scatter([x], [y], s=size, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3)
        ax.text(x, y + 0.14, str(i), ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


# ============================================================
# Problem definition
# ============================================================


@dataclass
class NetworkModel:
    n_nodes: int
    edges: List[Edge]
    lengths: Array           # shape (m,)
    stiffness: Array         # shape (m,)
    mass: Array              # shape (n,)
    damping: float
    alpha: Array             # damage aging coeffs, shape (m,)
    beta: Array              # damage fatigue coeffs, shape (m,)
    eps_safe: Array          # shape (m,)
    eps_crit: Array          # shape (m,)

    def __post_init__(self) -> None:
        self.n_edges = len(self.edges)
        self.B = np.zeros((self.n_edges, self.n_nodes), dtype=float)
        for e, (i, j) in enumerate(self.edges):
            self.B[e, i] = -1.0
            self.B[e, j] = 1.0
        self.M = np.diag(self.mass)
        self.C_damp = self.damping * np.eye(self.n_nodes)
        self.K_blocks = [self.stiffness[e] * np.outer(self.B[e], self.B[e]) for e in range(self.n_edges)]

    def strain(self, q: Array) -> Array:
        # q shape (n_nodes,)
        eps = np.zeros(self.n_edges, dtype=float)
        for e, (i, j) in enumerate(self.edges):
            eps[e] = (q[j] - q[i]) / self.lengths[e]
        return eps

    def assemble_stiffness(self, delta: Array, damage: Array) -> Array:
        # delta, damage shape (m,)
        K = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for e in range(self.n_edges):
            K += delta[e] * (1.0 - damage[e]) * self.K_blocks[e]
        return K


@dataclass
class TimeGrid:
    dt: float
    n_steps: int

    @property
    def times(self) -> Array:
        return np.arange(self.n_steps) * self.dt


@dataclass
class NewmarkParameters:
    beta: float = 0.25
    gamma: float = 0.5


# ============================================================
# Forward simulation
# ============================================================


def simulate_forward(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array,
    delta_schedule: Array,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    d0: Optional[Array] = None,
    newmark: Optional[NewmarkParameters] = None,
) -> Dict[str, Array]:
    """Simulate the forward switching-damage dynamics.

    forcing: shape (T, n)
    delta_schedule: shape (T-1, m), piecewise constant per time step interval [t_n,t_{n+1})
    """
    if newmark is None:
        newmark = NewmarkParameters()

    T = time.n_steps
    n = model.n_nodes
    m = model.n_edges
    dt = time.dt
    beta = newmark.beta
    gamma = newmark.gamma

    q = np.zeros((T, n), dtype=float)
    v = np.zeros((T, n), dtype=float)
    a = np.zeros((T, n), dtype=float)
    d = np.zeros((T, m), dtype=float)
    eps = np.zeros((T, m), dtype=float)

    if q0 is not None:
        q[0] = q0
    if v0 is not None:
        v[0] = v0
    if d0 is not None:
        d[0] = d0

    # initial acceleration from equilibrium
    K0 = model.assemble_stiffness(delta_schedule[0], d[0])
    rhs0 = forcing[0] - model.C_damp @ v[0] - K0 @ q[0]
    a[0] = np.linalg.solve(model.M, rhs0)
    eps[0] = model.strain(q[0])

    for nstep in range(T - 1):
        Kt = model.assemble_stiffness(delta_schedule[nstep], d[nstep])
        Keff = Kt + (gamma / (beta * dt)) * model.C_damp + (1.0 / (beta * dt * dt)) * model.M

        rhs = forcing[nstep + 1].copy()
        rhs += model.M @ (
            (1.0 / (beta * dt * dt)) * q[nstep]
            + (1.0 / (beta * dt)) * v[nstep]
            + (1.0 / (2.0 * beta) - 1.0) * a[nstep]
        )
        rhs += model.C_damp @ (
            (gamma / (beta * dt)) * q[nstep]
            + (gamma / beta - 1.0) * v[nstep]
            + dt * (gamma / (2.0 * beta) - 1.0) * a[nstep]
        )

        q[nstep + 1] = np.linalg.solve(Keff, rhs)
        a[nstep + 1] = (
            (1.0 / (beta * dt * dt)) * (q[nstep + 1] - q[nstep] - dt * v[nstep])
            - (1.0 / (2.0 * beta) - 1.0) * a[nstep]
        )
        v[nstep + 1] = v[nstep] + dt * ((1.0 - gamma) * a[nstep] + gamma * a[nstep + 1])

        eps[nstep + 1] = model.strain(q[nstep + 1])

        # Explicit Euler damage update
        excess = (eps[nstep] - model.eps_safe) / model.eps_crit
        fatigue = positive_part(excess) ** 2
        d_dot = model.alpha * (1.0 - d[nstep]) + model.beta * (1.0 - d[nstep]) * fatigue
        d[nstep + 1] = np.clip(d[nstep] + dt * d_dot, 0.0, 1.0)

    alpha_eff = delta_schedule * (1.0 - d[:-1])
    return {
        "q": q,
        "v": v,
        "a": a,
        "d": d,
        "eps": eps,
        "delta": delta_schedule,
        "alpha_eff": alpha_eff,
    }


# ============================================================
# Identification of effective coefficients alpha_eff
# ============================================================


def reconstruct_effective_coefficients(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array | Sequence[Array],
    q_meas: Array | Sequence[Array],
    observed_idx: Optional[Array] = None,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    n_iters: int = 500,
    step_size: float = 0.01,
    reg_tv: float = 0.1,
    reg_l2: float = 1.0e-3,
    window_size: int = 10,
    newmark: Optional[NewmarkParameters] = None,
) -> Array:
    """Recover alpha_eff[n,e] in [0,1] from measured displacement snapshots.

    This patched version reconstructs alpha on time windows, i.e. it assumes
    alpha_eff is piecewise constant on windows of length `window_size`. This
    strongly suppresses the spurious time-step-to-time-step flickering that was
    present in the fully pointwise reconstruction.
    """
    if newmark is None:
        newmark = NewmarkParameters()
    if observed_idx is None:
        observed_idx = np.arange(model.n_nodes, dtype=int)
    observed_idx = np.asarray(observed_idx, dtype=int)
    forcing_list = as_experiment_list(forcing)
    q_meas_list = as_experiment_list(q_meas)
    q_full_list = [complete_state_from_observations(model, q_obs, observed_idx) for q_obs in q_meas_list]

    T = time.n_steps
    m = model.n_edges
    dt = time.dt
    beta = newmark.beta
    gamma = newmark.gamma
    Tm1 = T - 1
    n_windows = int(np.ceil(Tm1 / window_size))

    # Estimate v,a from q_meas by finite differences for plug-in residuals.
    va_list: List[Tuple[Array, Array]] = []
    for q_full in q_full_list:
        v_est = np.zeros_like(q_full)
        a_est = np.zeros_like(q_full)
        for k in range(1, T - 1):
            v_est[k] = (q_full[k + 1] - q_full[k - 1]) / (2.0 * dt)
            a_est[k] = (q_full[k + 1] - 2.0 * q_full[k] + q_full[k - 1]) / (dt * dt)
        v_est[0] = v_est[1]
        v_est[-1] = v_est[-2]
        a_est[0] = a_est[1]
        a_est[-1] = a_est[-2]
        va_list.append((v_est, a_est))

    # Window variables: one alpha per edge per time window.
    alpha_win = 0.8 * np.ones((n_windows, m), dtype=float)

    for _ in range(n_iters):
        grad = np.zeros_like(alpha_win)

        for forcing_r, q_full, (v_est, a_est) in zip(forcing_list, q_full_list, va_list):
            for k in range(Tm1):
                w = min(k // window_size, n_windows - 1)
                qk = q_full[k]
                qkp1 = q_full[k + 1]
                vk = v_est[k]
                ak = a_est[k]

                rhs = forcing_r[k + 1].copy()
                rhs += model.M @ (
                    (1.0 / (beta * dt * dt)) * qk
                    + (1.0 / (beta * dt)) * vk
                    + (1.0 / (2.0 * beta) - 1.0) * ak
                )
                rhs += model.C_damp @ (
                    (gamma / (beta * dt)) * qk
                    + (gamma / beta - 1.0) * vk
                    + dt * (gamma / (2.0 * beta) - 1.0) * ak
                )

                Keff_q = (1.0 / (beta * dt * dt)) * (model.M @ qkp1) + (gamma / (beta * dt)) * (model.C_damp @ qkp1)
                residual = Keff_q - rhs
                edge_contrib = np.zeros((m, model.n_nodes), dtype=float)
                for e in range(m):
                    edge_contrib[e] = model.K_blocks[e] @ qkp1
                    residual += alpha_win[w, e] * edge_contrib[e]

                for e in range(m):
                    grad[w, e] += edge_contrib[e] @ residual

        # Regularization in window-time.
        if reg_tv > 0.0:
            for w in range(n_windows - 1):
                diff = alpha_win[w + 1] - alpha_win[w]
                grad[w] += -reg_tv * diff
                grad[w + 1] += reg_tv * diff

        if reg_l2 > 0.0:
            grad += reg_l2 * alpha_win

        alpha_win -= step_size * grad
        alpha_win = np.clip(alpha_win, 0.0, 1.0)

    # Expand windowwise coefficients back to per-step alpha_eff.
    alpha_eff = np.zeros((Tm1, m), dtype=float)
    for k in range(Tm1):
        w = min(k // window_size, n_windows - 1)
        alpha_eff[k] = alpha_win[w]

    return alpha_eff


# ============================================================
# Recover delta from alpha_eff
# ============================================================


def infer_switching_from_alpha(
    alpha_eff: Array,
    on_threshold: float = 0.45,
    off_threshold: float = 0.25,
    smooth_window: int = 5,
    min_dwell: int = 3,
) -> Array:
    """Infer Boolean switching from effective coefficients using hysteresis.

    alpha_eff: shape (T-1, m)
    returns delta_hat: same shape, values in {0,1}
    """
    smoothed = moving_average(alpha_eff, smooth_window)
    signal, _ = normalize_edgewise_signal(smoothed)
    Tm1, m = signal.shape
    delta = np.zeros((Tm1, m), dtype=int)

    for e in range(m):
        early_mean = float(np.mean(signal[: min(5, Tm1), e]))
        state = 1 if early_mean >= off_threshold else 0
        last_switch = 0
        delta[0, e] = state
        for k in range(1, Tm1):
            if state == 0 and signal[k, e] >= on_threshold and (k - last_switch) >= min_dwell:
                state = 1
                last_switch = k
            elif state == 1 and signal[k, e] <= off_threshold and (k - last_switch) >= min_dwell:
                state = 0
                last_switch = k
            delta[k, e] = state
    return delta


# ============================================================
# Recover damage from alpha_eff and delta
# ============================================================


def reconstruct_damage_from_alpha_and_delta(
    alpha_eff: Array,
    delta: Array,
    max_active_damage: float = 0.95,
    relax: float = 0.25,
    damage_margin: float = 0.10,
    max_step_increase: float = 0.03,
) -> Array:
    """Compute a monotone damage estimate from alpha_eff and delta.

    If delta=1, we target d pprox 1-alpha_eff but enforce monotonicity and a
    modest relaxation to reduce noise sensitivity. If delta=0, damage is not
    directly observable from alpha_eff; we keep the previous value.

    max_active_damage is a separation prior: as long as an edge is believed
    active, we do not allow damage to jump all the way to 1.0, because that is
    indistinguishable from delta=0 in the product alpha_eff=delta*(1-d).
    """
    alpha_sm = moving_average(alpha_eff, 5)
    Tm1, m = alpha_sm.shape
    d_hat = np.zeros((Tm1 + 1, m), dtype=float)
    baseline_scales = np.zeros(m, dtype=float)
    for e in range(m):
        active_idx = np.flatnonzero(delta[:, e] == 1)
        if active_idx.size >= 3:
            first_idx = active_idx[: min(5, active_idx.size)]
            baseline_scales[e] = max(float(np.median(alpha_sm[first_idx, e])), 1.0e-6)
        elif active_idx.size > 0:
            baseline_scales[e] = max(float(np.max(alpha_sm[active_idx, e])), 1.0e-6)
        else:
            baseline_scales[e] = max(float(np.percentile(alpha_sm[:, e], 90.0)), 1.0e-6)

    for k in range(Tm1):
        d_hat[k + 1] = d_hat[k]
        for e in range(m):
            if delta[k, e] == 1:
                intact_fraction = np.clip(alpha_sm[k, e] / baseline_scales[e], 0.0, 1.0)
                target = np.clip(1.0 - intact_fraction - damage_margin, 0.0, max_active_damage)
                proposal = (1.0 - relax) * d_hat[k, e] + relax * target
                monotone = np.clip(max(d_hat[k, e], proposal), 0.0, 1.0)
                d_hat[k + 1, e] = min(monotone, d_hat[k, e] + max_step_increase)
            else:
                d_hat[k + 1, e] = d_hat[k, e]
    return d_hat


# ============================================================
# Local search refinement of delta
# ============================================================


def objective_for_delta_damage(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array,
    q_meas: Array,
    delta: Array,
    d_hat: Array,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    newmark: Optional[NewmarkParameters] = None,
) -> float:
    sim = simulate_forward(model, time, forcing, delta, q0=q0, v0=v0, d0=d_hat[0], newmark=newmark)
    misfit = q_meas - sim["q"]
    return 0.5 * float(np.sum(misfit * misfit))



def local_refine_delta(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array,
    q_meas: Array,
    delta_init: Array,
    d_hat: Array,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    max_passes: int = 2,
) -> Array:
    delta = delta_init.copy()
    base_obj = objective_for_delta_damage(model, time, forcing, q_meas, delta, d_hat, q0=q0, v0=v0)
    Tm1, m = delta.shape
    for _ in range(max_passes):
        improved = False
        for k in range(Tm1):
            for e in range(m):
                delta_trial = delta.copy()
                delta_trial[k, e] = 1 - delta_trial[k, e]
                obj_trial = objective_for_delta_damage(model, time, forcing, q_meas, delta_trial, d_hat, q0=q0, v0=v0)
                if obj_trial < base_obj:
                    delta = delta_trial
                    base_obj = obj_trial
                    improved = True
        if not improved:
            break
    return delta


def refine_delta_with_cpsat(
    alpha_eff_hat: Array,
    d_hat: Array,
    lambda_switch: int = 8,
    lambda_inactive: int = 4,
    scale: int = 1000,
    block_size: int = 10,
) -> Array:
    """Mixed-integer refinement of switching using OR-Tools CP-SAT.

    We solve, with d_hat frozen,
        min sum |(1-d_hat[k,e]) delta[k,e] - alpha_eff_hat[k,e]|
            + lambda_switch * TV(delta)
            + lambda_inactive * sum (1-delta)
    where delta[k,e] in {0,1}.

    The inactivity penalty reflects the prior used in the synthetic example:
    most edges are active most of the time, with only occasional outages.
    If OR-Tools is unavailable, a RuntimeError is raised.
    """
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OR-Tools is required for CP-SAT refinement. Install with `pip install ortools`.") from exc

    Tm1, m = alpha_eff_hat.shape
    model = cp_model.CpModel()
    delta = {(k, e): model.NewBoolVar(f"delta_{k}_{e}") for k in range(Tm1) for e in range(m)}

    obj_terms = []

    for k in range(Tm1):
        for e in range(m):
            coef = int(round(scale * max(0.0, 1.0 - d_hat[k, e])))
            target = int(round(scale * alpha_eff_hat[k, e]))
            pred = model.NewIntVar(0, scale, f"pred_{k}_{e}")
            model.Add(pred == coef * delta[(k, e)])

            diff = model.NewIntVar(-scale, scale, f"diff_{k}_{e}")
            model.Add(diff == pred - target)
            abs_diff = model.NewIntVar(0, scale, f"abs_{k}_{e}")
            model.AddAbsEquality(abs_diff, diff)
            obj_terms.append(abs_diff)
            if lambda_inactive > 0:
                inactive = model.NewIntVar(0, 1, f"inactive_{k}_{e}")
                model.Add(inactive == 1 - delta[(k, e)])
                obj_terms.append(lambda_inactive * inactive)

    if block_size > 1:
        for e in range(m):
            for start in range(0, Tm1, block_size):
                end = min(start + block_size, Tm1)
                for k in range(start + 1, end):
                    model.Add(delta[(k, e)] == delta[(start, e)])

    if lambda_switch > 0:
        for k in range(1, Tm1):
            for e in range(m):
                sw = model.NewBoolVar(f"sw_{k}_{e}")
                model.Add(delta[(k, e)] - delta[(k - 1, e)] <= sw)
                model.Add(delta[(k - 1, e)] - delta[(k, e)] <= sw)
                obj_terms.append(lambda_switch * sw)

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT refinement failed with status {solver.StatusName(status)}")

    delta_out = np.zeros((Tm1, m), dtype=int)
    for k in range(Tm1):
        for e in range(m):
            delta_out[k, e] = solver.Value(delta[(k, e)])
    return delta_out


def build_mode_library_from_signals(
    alpha_eff_hat: Array,
    switch_times: Array,
    delta_init: Optional[Array] = None,
    max_modes: Optional[int] = None,
) -> Array:
    """Build a small shared-mode library from an initial switching guess.

    Mode 0 is always the all-active pattern. Additional modes are single-edge
    outages for the edges with the strongest abrupt signal changes near the
    detected global switch times. This favors true switching edges over edges
    that simply drift low due to damage.
    """
    smoothed = moving_average(alpha_eff_hat, 9)
    signal, _ = normalize_edgewise_signal(smoothed)
    Tm1, m = signal.shape
    grad = np.abs(np.diff(signal, axis=0, prepend=signal[[0], :]))
    mask = switch_times[:Tm1].astype(float)[:, None]
    scores = np.sum(mask * grad, axis=0)
    candidate_edges: List[int] = []
    for e in np.argsort(-scores):
        if scores[e] > 1.0e-6:
            candidate_edges.append(int(e))
    if delta_init is not None:
        inactive_fraction = 1.0 - np.mean(delta_init, axis=0)
        for e in np.argsort(-inactive_fraction):
            if inactive_fraction[e] > 0.05 and int(e) not in candidate_edges:
                candidate_edges.append(int(e))
    if max_modes is None:
        max_modes = m + 1
    candidate_edges = candidate_edges[: max(0, max_modes - 1)]

    patterns: List[Array] = [np.ones(m, dtype=int)]
    for e in candidate_edges:
        mode = np.ones(m, dtype=int)
        mode[e] = 0
        patterns.append(mode)
    return np.asarray(patterns, dtype=int)


def refine_delta_with_mode_library_cpsat(
    alpha_eff_hat: Array,
    d_hat: Array,
    mode_patterns: Array,
    lambda_switch: int = 8,
    scale: int = 1000,
    block_size: int = 10,
) -> Tuple[Array, Array]:
    """Refine switching using a small shared-mode library on time blocks."""
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OR-Tools is required for CP-SAT refinement. Install with `pip install ortools`.") from exc

    Tm1, m = alpha_eff_hat.shape
    n_blocks = int(np.ceil(Tm1 / block_size))
    n_modes = mode_patterns.shape[0]

    model = cp_model.CpModel()
    z = {(b, r): model.NewBoolVar(f"z_{b}_{r}") for b in range(n_blocks) for r in range(n_modes)}
    for b in range(n_blocks):
        model.Add(sum(z[(b, r)] for r in range(n_modes)) == 1)

    obj_terms = []
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, Tm1)
        for k in range(start, end):
            for e in range(m):
                target = int(round(scale * alpha_eff_hat[k, e]))
                pred_terms = []
                for r in range(n_modes):
                    coef = int(round(scale * max(0.0, 1.0 - d_hat[k, e]) * mode_patterns[r, e]))
                    if coef != 0:
                        pred_terms.append(coef * z[(b, r)])
                pred = model.NewIntVar(0, scale, f"pred_{k}_{e}")
                if pred_terms:
                    model.Add(pred == sum(pred_terms))
                else:
                    model.Add(pred == 0)

                diff = model.NewIntVar(-scale, scale, f"diff_mode_{k}_{e}")
                model.Add(diff == pred - target)
                abs_diff = model.NewIntVar(0, scale, f"abs_mode_{k}_{e}")
                model.AddAbsEquality(abs_diff, diff)
                obj_terms.append(abs_diff)

    if lambda_switch > 0:
        for b in range(1, n_blocks):
            changed = model.NewBoolVar(f"changed_{b}")
            for r in range(n_modes):
                model.Add(z[(b - 1, r)] - z[(b, r)] <= changed)
                model.Add(z[(b, r)] - z[(b - 1, r)] <= changed)
            obj_terms.append(lambda_switch * changed)

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Mode-library CP-SAT refinement failed with status {solver.StatusName(status)}")

    delta_out = np.ones((Tm1, m), dtype=int)
    active_mode = np.zeros(n_blocks, dtype=int)
    for b in range(n_blocks):
        chosen = 0
        for r in range(n_modes):
            if solver.Value(z[(b, r)]):
                chosen = r
                break
        active_mode[b] = chosen
        start = b * block_size
        end = min(start + block_size, Tm1)
        delta_out[start:end] = mode_patterns[chosen]
    return delta_out, active_mode


# ============================================================
# Full inverse pipeline
# ============================================================


def detect_switch_candidates_from_residuals(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array | Sequence[Array],
    q_meas: Array | Sequence[Array],
    observed_idx: Optional[Array] = None,
    threshold_scale: float = 2.5,
) -> Array:
    """Detect likely switching times from residual jumps.

    We form a discrete acceleration from q_meas and compute the dynamic residual
        r_k = M a_k + C v_k + K_ref q_k - f_k,
    with K_ref assembled from the all-active undamaged graph. Large changes in
    ||r_k|| indicate likely switches.
    Returns a Boolean array switch_times[k] for k=1,...,T-2 aligned with time step k.
    """
    if observed_idx is None:
        observed_idx = np.arange(model.n_nodes, dtype=int)
    observed_idx = np.asarray(observed_idx, dtype=int)
    forcing_list = as_experiment_list(forcing)
    q_meas_list = as_experiment_list(q_meas)

    T = time.n_steps
    dt = time.dt
    K_ref = model.assemble_stiffness(np.ones(model.n_edges), np.zeros(model.n_edges))
    resid = np.zeros(T)
    for forcing_r, q_obs in zip(forcing_list, q_meas_list):
        q_full = complete_state_from_observations(model, q_obs, observed_idx)
        v_est = np.zeros_like(q_full)
        a_est = np.zeros_like(q_full)
        for k in range(1, T - 1):
            v_est[k] = (q_full[k + 1] - q_full[k - 1]) / (2.0 * dt)
            a_est[k] = (q_full[k + 1] - 2.0 * q_full[k] + q_full[k - 1]) / (dt * dt)
        v_est[0] = v_est[1]
        v_est[-1] = v_est[-2]
        a_est[0] = a_est[1]
        a_est[-1] = a_est[-2]

        for k in range(T):
            rk = model.M @ a_est[k] + model.C_damp @ v_est[k] + K_ref @ q_full[k] - forcing_r[k]
            resid[k] += np.linalg.norm(rk)

    jump = np.abs(np.diff(resid))
    med = np.median(jump)
    mad = np.median(np.abs(jump - med)) + 1e-12
    thr = med + threshold_scale * mad
    switch_times = np.zeros(T - 1, dtype=int)
    for k in range(1, T - 1):
        if jump[k - 1] >= thr:
            switch_times[k] = 1
    return switch_times


def infer_switching_from_alpha_with_candidates(
    alpha_eff: Array,
    switch_times: Array,
    on_threshold: float = 0.45,
    off_threshold: float = 0.25,
    smooth_window: int = 9,
    min_dwell: int = 8,
) -> Array:
    smoothed = moving_average(alpha_eff, smooth_window)
    signal, _ = normalize_edgewise_signal(smoothed)
    Tm1, m = signal.shape
    delta = np.zeros((Tm1, m), dtype=int)

    for e in range(m):
        early_mean = float(np.mean(signal[: min(5, Tm1), e]))
        state = 1 if early_mean >= off_threshold else 0
        last_switch = 0
        delta[0, e] = state
        for k in range(1, Tm1):
            allow = bool(switch_times[k]) or (k - last_switch) >= 2 * min_dwell
            if allow and state == 0 and signal[k, e] >= on_threshold and (k - last_switch) >= min_dwell:
                state = 1
                last_switch = k
            elif allow and state == 1 and signal[k, e] <= off_threshold and (k - last_switch) >= min_dwell:
                state = 0
                last_switch = k
            delta[k, e] = state
    return delta


def project_delta_to_blocks(delta: Array, block_size: int = 10) -> Array:
    if block_size <= 1:
        return delta.copy()
    Tm1, m = delta.shape
    out = delta.copy()
    for e in range(m):
        for start in range(0, Tm1, block_size):
            end = min(start + block_size, Tm1)
            block = delta[start:end, e]
            value = 1 if np.mean(block) >= 0.5 else 0
            out[start:end, e] = value
    return out


def failure_times_to_delta(failure_times: Array, Tm1: int) -> Array:
    """Convert irreversible edge failure times to a delta schedule.

    failure_times[e] = first time index k at which edge e is failed.
    If failure_times[e] >= Tm1, the edge never fails.
    """
    m = failure_times.shape[0]
    delta = np.ones((Tm1, m), dtype=int)
    for e in range(m):
        tau = int(failure_times[e])
        if tau < Tm1:
            delta[tau:, e] = 0
    return delta


def delta_to_failure_times(delta: Array) -> Array:
    """Extract first failure times from an irreversible delta schedule."""
    Tm1, m = delta.shape
    tau = np.full(m, Tm1, dtype=int)
    for e in range(m):
        off_idx = np.flatnonzero(delta[:, e] == 0)
        if off_idx.size > 0:
            tau[e] = int(off_idx[0])
    return tau


def infer_failure_times_from_alpha(
    alpha_eff_hat: Array,
    off_threshold: float = 0.25,
    smooth_window: int = 9,
    min_off_block: int = 8,
) -> Array:
    """Initial discrete failure-time guess from reconstructed effective coefficients."""
    smoothed = moving_average(alpha_eff_hat, smooth_window)
    signal, _ = normalize_edgewise_signal(smoothed)
    Tm1, m = signal.shape
    tau = np.full(m, Tm1, dtype=int)
    for e in range(m):
        off_count = 0
        for k in range(Tm1):
            if signal[k, e] <= off_threshold:
                off_count += 1
                if off_count >= min_off_block:
                    tau[e] = k - min_off_block + 1
                    break
            else:
                off_count = 0
    return tau


def objective_for_failure_times(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array | Sequence[Array],
    q_meas: Array | Sequence[Array],
    failure_times: Array,
    observed_idx: Optional[Array] = None,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    newmark: Optional[NewmarkParameters] = None,
    lambda_fail: float = 0.25,
    lambda_early: float = 0.05,
    fail_weights: Optional[Array] = None,
) -> float:
    delta = failure_times_to_delta(failure_times, time.n_steps - 1)
    forcing_list = as_experiment_list(forcing)
    q_meas_list = as_experiment_list(q_meas)
    if observed_idx is None:
        observed_idx = np.arange(model.n_nodes, dtype=int)
    observed_idx = np.asarray(observed_idx, dtype=int)
    misfit_value = 0.0
    for forcing_r, q_obs in zip(forcing_list, q_meas_list):
        sim = simulate_forward(model, time, forcing_r, delta, q0=q0, v0=v0, d0=np.zeros(model.n_edges), newmark=newmark)
        misfit = q_obs - sim["q"][:, observed_idx]
        misfit_value += 0.5 * float(np.sum(misfit * misfit))
    Tm1 = time.n_steps - 1
    fail_mask = (failure_times < Tm1).astype(float)
    early_cost = np.clip((Tm1 - failure_times) / max(1, Tm1), 0.0, 1.0)
    if fail_weights is None:
        fail_weights = np.ones_like(failure_times, dtype=float)
    penalty = lambda_fail * float(np.sum(fail_weights * fail_mask)) + lambda_early * float(np.sum(fail_weights * early_cost))
    return misfit_value + penalty


def candidate_failure_times(
    alpha_eff_hat: Array,
    switch_times: Array,
    block_size: int = 10,
) -> Array:
    """Build a small discrete candidate set of possible failure times."""
    Tm1 = alpha_eff_hat.shape[0]
    times = {Tm1}
    for k in np.flatnonzero(switch_times):
        kk = int(block_size * round(k / max(1, block_size)))
        for shift in (-block_size, 0, block_size):
            cand = kk + shift
            cand = max(0, min(Tm1 - 1, cand))
            times.add(cand)
    for frac in (0.25, 0.5, 0.75):
        kk = int(frac * Tm1)
        kk = max(0, min(Tm1 - 1, kk))
        times.add(kk)
    return np.array(sorted(times), dtype=int)


def edgewise_candidate_failure_times(
    alpha_eff_hat: Array,
    tau_init: Array,
    switch_times: Array,
    block_size: int = 10,
    max_candidates_per_edge: int = 5,
) -> List[Array]:
    """Build a small candidate set for each edge separately."""
    smoothed = moving_average(alpha_eff_hat, 9)
    signal, _ = normalize_edgewise_signal(smoothed)
    Tm1, m = signal.shape
    edge_candidates: List[Array] = []
    for e in range(m):
        candidates = {Tm1}
        grad = np.diff(signal[:, e], prepend=signal[0, e])
        strongest_drop_idx = np.argsort(grad)[: max(1, max_candidates_per_edge - 1)]
        for k in strongest_drop_idx:
            kk = int(block_size * round(int(k) / max(1, block_size)))
            for shift in (-block_size, 0, block_size):
                cand = int(np.clip(kk + shift, 0, Tm1))
                candidates.add(cand)

        if tau_init[e] < Tm1:
            candidates.add(int(tau_init[e]))

        for k in np.flatnonzero(switch_times):
            kk = int(block_size * round(int(k) / max(1, block_size)))
            if grad[min(int(k), Tm1 - 1)] < -0.05:
                candidates.add(int(np.clip(kk, 0, Tm1)))

        cand_sorted = sorted(candidates)
        if len(cand_sorted) > max_candidates_per_edge:
            # Keep the no-failure option and the strongest drop locations.
            scored = []
            for cand in cand_sorted:
                if cand == Tm1:
                    score = -math.inf
                else:
                    idx = min(max(0, cand), Tm1 - 1)
                    score = grad[idx]
                scored.append((score, cand))
            ordered = [cand for _, cand in sorted(scored, key=lambda t: t[0])]
            keep = sorted(set([Tm1] + ordered[: max_candidates_per_edge - 1]))
            if Tm1 not in keep:
                keep[-1] = Tm1
                keep = sorted(set(keep))
            cand_sorted = keep
        edge_candidates.append(np.array(cand_sorted, dtype=int))
    return edge_candidates


def exact_search_failure_times(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array | Sequence[Array],
    q_meas: Array | Sequence[Array],
    edge_candidates: Sequence[Array],
    observed_idx: Optional[Array] = None,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    lambda_fail: float = 0.25,
    lambda_early: float = 0.05,
    fail_weights: Optional[Array] = None,
    max_combinations: int = 250000,
) -> Tuple[Array, float, int]:
    """Exact discrete solve on a reduced Cartesian product of edgewise candidates."""
    sizes = [len(c) for c in edge_candidates]
    total = int(np.prod(sizes, dtype=np.int64))
    if total > max_combinations:
        raise RuntimeError(f"Exact search space too large: {total} combinations")

    best_tau: Optional[Array] = None
    best_obj = math.inf
    evaluated = 0
    for choice in product(*[list(map(int, c)) for c in edge_candidates]):
        tau = np.array(choice, dtype=int)
        obj = objective_for_failure_times(
            model,
            time,
            forcing,
            q_meas,
            tau,
            observed_idx=observed_idx,
            q0=q0,
            v0=v0,
            lambda_fail=lambda_fail,
            lambda_early=lambda_early,
            fail_weights=fail_weights,
        )
        evaluated += 1
        if obj < best_obj:
            best_obj = obj
            best_tau = tau.copy()

    if best_tau is None:
        raise RuntimeError("Exact discrete search failed to evaluate any candidate")
    return best_tau, float(best_obj), evaluated


def local_refine_failure_times(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array | Sequence[Array],
    q_meas: Array | Sequence[Array],
    tau_init: Array,
    candidates: Array,
    observed_idx: Optional[Array] = None,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    max_passes: int = 4,
    lambda_fail: float = 0.25,
    lambda_early: float = 0.05,
    fail_weights: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Greedy discrete refinement over edgewise failure times."""
    tau = tau_init.copy()
    obj = objective_for_failure_times(
        model,
        time,
        forcing,
        q_meas,
        tau,
        observed_idx=observed_idx,
        q0=q0,
        v0=v0,
        lambda_fail=lambda_fail,
        lambda_early=lambda_early,
        fail_weights=fail_weights,
    )
    history = [obj]
    m = tau.shape[0]
    for _ in range(max_passes):
        improved = False
        for e in range(m):
            best_tau_e = int(tau[e])
            best_obj = obj
            for cand in candidates:
                cand = int(cand)
                if cand == tau[e]:
                    continue
                tau_trial = tau.copy()
                tau_trial[e] = cand
                obj_trial = objective_for_failure_times(
                    model,
                    time,
                    forcing,
                    q_meas,
                    tau_trial,
                    observed_idx=observed_idx,
                    q0=q0,
                    v0=v0,
                    lambda_fail=lambda_fail,
                    lambda_early=lambda_early,
                    fail_weights=fail_weights,
                )
                if obj_trial + 1.0e-10 < best_obj:
                    best_obj = obj_trial
                    best_tau_e = cand
            if best_tau_e != tau[e]:
                tau[e] = best_tau_e
                obj = best_obj
                improved = True
        history.append(obj)
        if not improved:
            break
    return tau, np.array(history, dtype=float)


def joint_inverse_pipeline(
    model: NetworkModel,
    time: TimeGrid,
    forcing: Array | Sequence[Array],
    q_meas: Array | Sequence[Array],
    observed_idx: Optional[Array] = None,
    q0: Optional[Array] = None,
    v0: Optional[Array] = None,
    use_cpsat_refinement: bool = True,
    n_outer_iters: int = 4,
    lambda_fail: float = 0.05,
    lambda_early: float = 0.01,
    use_exact_discrete_search: bool = True,
) -> Dict[str, Array]:
    if observed_idx is None:
        observed_idx = np.arange(model.n_nodes, dtype=int)
    observed_idx = np.asarray(observed_idx, dtype=int)
    fail_weights = edge_failure_weights(model, observed_idx)

    alpha_eff_hat = reconstruct_effective_coefficients(
        model=model,
        time=time,
        forcing=forcing,
        q_meas=q_meas,
        observed_idx=observed_idx,
        q0=q0,
        v0=v0,
    )

    switch_times = detect_switch_candidates_from_residuals(
        model, time, forcing, q_meas, observed_idx=observed_idx
    )
    tau_hat = infer_failure_times_from_alpha(alpha_eff_hat)
    candidates = candidate_failure_times(alpha_eff_hat, switch_times, block_size=10)
    tau_local, objective_history_local = local_refine_failure_times(
        model=model,
        time=time,
        forcing=forcing,
        q_meas=q_meas,
        tau_init=tau_hat,
        candidates=candidates,
        observed_idx=observed_idx,
        q0=q0,
        v0=v0,
        max_passes=max(1, n_outer_iters),
        lambda_fail=lambda_fail,
        lambda_early=lambda_early,
        fail_weights=fail_weights,
    )

    edge_candidates = edgewise_candidate_failure_times(
        alpha_eff_hat,
        tau_local,
        switch_times,
        block_size=10,
        max_candidates_per_edge=3,
    )
    exact_evaluations = 0
    if use_exact_discrete_search:
        try:
            tau_refined, best_obj, exact_evaluations = exact_search_failure_times(
                model=model,
                time=time,
                forcing=forcing,
                q_meas=q_meas,
                edge_candidates=edge_candidates,
                observed_idx=observed_idx,
                q0=q0,
                v0=v0,
                lambda_fail=lambda_fail,
                lambda_early=lambda_early,
                fail_weights=fail_weights,
                max_combinations=50000,
            )
            objective_history = np.concatenate([objective_history_local, np.array([best_obj], dtype=float)])
            local_best = float(objective_history_local[-1])
            if best_obj + 1.0e-10 < local_best:
                refinement_method = "exact-discrete-search"
            else:
                tau_refined = tau_local
                refinement_method = "local-discrete-search (exact checked)"
        except Exception:
            tau_refined = tau_local
            objective_history = objective_history_local
            refinement_method = "local-discrete-search"
    else:
        tau_refined = tau_local
        objective_history = objective_history_local
        refinement_method = "local-discrete-search"

    delta_hat = failure_times_to_delta(tau_hat, time.n_steps - 1)
    delta_refined = failure_times_to_delta(tau_refined, time.n_steps - 1)

    forcing_list = as_experiment_list(forcing)
    simulation_refined = [
        simulate_forward(
            model=model,
            time=time,
            forcing=forcing_r,
            delta_schedule=delta_refined,
            q0=q0,
            v0=v0,
            d0=np.zeros(model.n_edges),
        )
        for forcing_r in forcing_list
    ]
    d_hat = simulation_refined[0]["d"]
    d_refined = simulation_refined[0]["d"]

    return {
        "alpha_eff_hat": alpha_eff_hat,
        "switch_times": switch_times,
        "delta_hat": delta_hat,
        "delta_refined": delta_refined,
        "tau_hat": tau_hat,
        "tau_refined": tau_refined,
        "edge_candidates": edge_candidates,
        "d_hat": d_hat,
        "d_refined": d_refined,
        "simulation_refined": simulation_refined,
        "refinement_method": refinement_method,
        "objective_history": np.array(objective_history, dtype=float),
        "candidates": candidates,
        "lambda_fail": lambda_fail,
        "lambda_early": lambda_early,
        "fail_weights": fail_weights,
        "exact_evaluations": exact_evaluations,
    }


# ============================================================
# Synthetic example
# ============================================================


def make_demo_problem() -> Tuple[NetworkModel, TimeGrid, List[Array], Dict[str, Array]]:
    np.random.seed(3)
    n_nodes = 8
    # 8-node tree with three branching nodes of degree 3 and five leaves.
    edges = [(0, 1), (1, 2), (0, 3), (0, 4), (1, 5), (2, 6), (2, 7)]
    m = len(edges)

    lengths = np.ones(m)
    stiffness = 12.0 * np.ones(m)
    mass = np.ones(n_nodes)
    damping = 0.15
    alpha = 0.004 * np.ones(m)
    beta = 2.5 * np.ones(m)
    eps_safe = 0.03 * np.ones(m)
    eps_crit = 0.06 * np.ones(m)

    model = NetworkModel(
        n_nodes=n_nodes,
        edges=edges,
        lengths=lengths,
        stiffness=stiffness,
        mass=mass,
        damping=damping,
        alpha=alpha,
        beta=beta,
        eps_safe=eps_safe,
        eps_crit=eps_crit,
    )

    time = TimeGrid(dt=0.02, n_steps=180)
    T = time.n_steps

    forcing_1 = np.zeros((T, n_nodes), dtype=float)
    forcing_1[:, 3] = 0.8
    forcing_1[:, 4] = 0.6 * np.sin(2.0 * np.pi * 0.5 * time.times)
    forcing_1[:, 5] = -0.5
    forcing_1[:, 6] = 0.9 * np.sin(2.0 * np.pi * 0.7 * time.times + 0.3)
    forcing_1[:, 7] = 0.7 * np.cos(2.0 * np.pi * 0.4 * time.times)

    forcing_2 = np.zeros((T, n_nodes), dtype=float)
    forcing_2[:, 3] = -0.4 + 0.5 * np.sin(2.0 * np.pi * 0.3 * time.times)
    forcing_2[:, 4] = 0.9
    forcing_2[:, 5] = 0.4 * np.cos(2.0 * np.pi * 0.6 * time.times + 0.5)
    forcing_2[:, 6] = -0.7
    forcing_2[:, 7] = 0.6 * np.sin(2.0 * np.pi * 0.8 * time.times)

    forcing_3 = np.zeros((T, n_nodes), dtype=float)
    forcing_3[:, 3] = 0.5 * np.cos(2.0 * np.pi * 0.45 * time.times)
    forcing_3[:, 4] = -0.6
    forcing_3[:, 5] = 0.8 * np.sin(2.0 * np.pi * 0.55 * time.times + 0.2)
    forcing_3[:, 6] = 0.5
    forcing_3[:, 7] = -0.9 * np.cos(2.0 * np.pi * 0.35 * time.times)
    forcings = [forcing_1, forcing_2, forcing_3]

    # True breakage schedule: irreversible edge failures on a tree.
    true_failure_times = np.full(m, T - 1, dtype=int)
    true_failure_times[3] = 60   # edge (0,4)
    true_failure_times[1] = 110  # edge (1,2)
    true_failure_times[6] = 140  # edge (2,7)
    delta_true = failure_times_to_delta(true_failure_times, T - 1)

    q0 = np.zeros(n_nodes)
    v0 = np.zeros(n_nodes)
    d0 = np.zeros(m)

    truth_runs = [simulate_forward(model, time, frc, delta_true, q0=q0, v0=v0, d0=d0) for frc in forcings]

    observed_idx = np.array([3, 4, 5, 6, 7], dtype=int)
    q_meas_list = [
        run["q"][:, observed_idx] + 5.0e-4 * np.random.randn(T, observed_idx.size)
        for run in truth_runs
    ]
    truth = {
        "runs": truth_runs,
        "q_meas_list": q_meas_list,
        "observed_idx": observed_idx,
        "positions": default_tree_positions(n_nodes),
        "q0": q0,
        "v0": v0,
        "delta": delta_true,
        "true_failure_times": true_failure_times,
        "d": truth_runs[0]["d"],
    }

    return model, time, forcings, truth


# ============================================================
# Main
# ============================================================


def main() -> None:
    model, time, forcings, truth = make_demo_problem()
    result = joint_inverse_pipeline(
        model=model,
        time=time,
        forcing=forcings,
        q_meas=truth["q_meas_list"],
        observed_idx=truth["observed_idx"],
        q0=truth["q0"],
        v0=truth["v0"],
    )

    print("=== Joint switching-damage inverse pipeline ===")
    print(f"Refinement method: {result['refinement_method']}")
    print("Final true damage:")
    print(np.round(truth["d"][-1], 4))
    print("Final reconstructed damage:")
    print(np.round(result["d_refined"][-1], 4))
    print("Outer objective history:")
    print(np.round(result["objective_history"], 6))
    print(f"Exact discrete evaluations: {result['exact_evaluations']}")
    print(f"Failure prior weights: lambda_fail={result['lambda_fail']}, lambda_early={result['lambda_early']}")
    print("Edge failure weights:")
    print(np.round(result["fail_weights"], 3))
    print("Candidate failure times:")
    print(result["candidates"])
    print("Edgewise candidate sets:")
    for e, cset in enumerate(result["edge_candidates"]):
        print(f"  edge {e:02d}: {cset}")
    print("True failure times:")
    print(delta_to_failure_times(truth["delta"]))
    print("Estimated failure times:")
    print(result["tau_refined"])

    print("\nSwitching summary (true / inferred active counts per edge):")
    true_counts = truth["delta"].sum(axis=0)
    est_counts = result["delta_refined"].sum(axis=0)
    for e, edge in enumerate(model.edges):
        print(f"edge {e:02d} {edge}: true={true_counts[e]:3d}, est={est_counts[e]:3d}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=False)

        tq = time.times
        ta = time.times[:-1]
        leaf_plot_idx = 0
        observed_node = int(truth["observed_idx"][leaf_plot_idx])
        truth0 = truth["runs"][0]
        q_meas0 = truth["q_meas_list"][0]
        sim0 = result["simulation_refined"][0]

        for r, (q_meas_r, sim_r) in enumerate(zip(truth["q_meas_list"], result["simulation_refined"])):
            axes[0].plot(tq, q_meas_r[:, leaf_plot_idx], lw=1.0, alpha=0.8, label=f"meas load {r+1}" if r == 0 else None)
            axes[0].plot(tq, sim_r["q"][:, observed_node], lw=1.4, linestyle="--", alpha=0.9, label=f"recon load {r+1}" if r == 0 else None)
        axes[0].set_ylabel("displacement")
        axes[0].set_title(f"Leaf measurement/reconstruction at node {observed_node} across load cases")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for e in range(model.n_edges):
            axes[1].plot(ta, result["alpha_eff_hat"][:, e], lw=1.0)
        axes[1].set_ylabel(r"$\alpha_e^n$")
        axes[1].set_title("Recovered effective coefficients")
        axes[1].grid(True, alpha=0.3)

        for e in range(model.n_edges):
            axes[2].step(ta, truth["delta"][:, e] + 1.2 * e, where="post", lw=1.2, alpha=0.7)
            axes[2].step(ta, result["delta_refined"][:, e] + 1.2 * e, where="post", lw=0.9, linestyle="--")
        axes[2].set_ylabel("switching layers")
        axes[2].set_xlabel("time")
        axes[2].set_title("True (solid) and reconstructed (dashed) switching")
        for k, sw in enumerate(result["switch_times"]):
            if sw:
                axes[2].axvline(ta[k], color="k", alpha=0.08, linewidth=0.8)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(np.arange(len(result["objective_history"])), result["objective_history"], marker="o", lw=1.5)
        axes[3].set_xlabel("outer iteration")
        axes[3].set_ylabel("objective")
        axes[3].set_title("Discrete failure-time objective history")
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        out_dir = Path(__file__).resolve().parent / "results_joint_recovery_switching_delta_d_adjoint"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "joint_recovery_switching_delta_d_adjoint_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig_graph, axes_graph = plt.subplots(1, 3, figsize=(14, 4.8))
        Tm1 = time.n_steps - 1
        positions = truth["positions"]
        draw_tree_state(
            axes_graph[0],
            model,
            positions,
            truth["observed_idx"],
            np.full(model.n_edges, Tm1, dtype=int),
            np.zeros(model.n_edges),
            Tm1,
            "Initial Intact Tree\n(Observed leaves in blue)",
        )
        draw_tree_state(
            axes_graph[1],
            model,
            positions,
            truth["observed_idx"],
            truth["true_failure_times"],
            truth["d"][-1],
            Tm1,
            "True Final Damage / Failures",
        )
        draw_tree_state(
            axes_graph[2],
            model,
            positions,
            truth["observed_idx"],
            result["tau_refined"],
            result["d_refined"][-1],
            Tm1,
            "Recovered Final Damage / Failures",
        )
        fig_graph.tight_layout()
        fig_graph.savefig(
            out_dir / "joint_recovery_switching_delta_d_adjoint_graphs.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig_graph)

        fig_disc, ax_disc = plt.subplots(1, 1, figsize=(11, 4.8))
        true_tau = truth["true_failure_times"]
        est_tau = result["tau_refined"]
        for e, cset in enumerate(result["edge_candidates"]):
            y = np.full_like(cset, e, dtype=float)
            ax_disc.scatter(cset, y, s=70, facecolor="#d9d9d9", edgecolor="#666666", zorder=1)
            ax_disc.hlines(e, np.min(cset), np.max(cset), color="#cfcfcf", linewidth=1.0, zorder=0)
            ax_disc.scatter(
                [true_tau[e]],
                [e],
                s=110,
                marker="o",
                facecolor="#d62728",
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                label="true failure time" if e == 0 else None,
            )
            ax_disc.scatter(
                [est_tau[e]],
                [e],
                s=95,
                marker="D",
                facecolor="#1f77b4",
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
                label="recovered failure time" if e == 0 else None,
            )
        ax_disc.set_yticks(range(model.n_edges))
        ax_disc.set_yticklabels([f"{e}: {edge}" for e, edge in enumerate(model.edges)])
        ax_disc.set_xlabel("candidate failure time")
        ax_disc.set_ylabel("edge")
        ax_disc.set_title("Discrete candidate sets with true and recovered failure times")
        ax_disc.grid(True, axis="x", alpha=0.25)
        ax_disc.legend(loc="upper left")
        fig_disc.tight_layout()
        fig_disc.savefig(
            out_dir / "joint_recovery_switching_delta_d_adjoint_discrete_search.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig_disc)

        summary_txt = out_dir / "joint_recovery_switching_delta_d_adjoint_summary.txt"
        with summary_txt.open("w", encoding="utf-8") as f:
            f.write("Joint recovery with shared failure times on a measured-at-leaves tree\n")
            f.write(f"refinement_method: {result['refinement_method']}\n")
            f.write(f"lambda_fail: {result['lambda_fail']}\n")
            f.write(f"lambda_early: {result['lambda_early']}\n")
            f.write(f"fail_weights: {np.array2string(result['fail_weights'], precision=3)}\n")
            f.write(f"exact_evaluations: {result['exact_evaluations']}\n")
            f.write(f"true_failure_times: {np.array2string(truth['true_failure_times'])}\n")
            f.write(f"estimated_failure_times: {np.array2string(result['tau_refined'])}\n")
            f.write(f"objective_history: {np.array2string(result['objective_history'], precision=6)}\n")
            f.write("discrete_search_plot: joint_recovery_switching_delta_d_adjoint_discrete_search.png\n")
            f.write("edgewise candidate sets:\n")
            for e, cset in enumerate(result["edge_candidates"]):
                f.write(f"  edge {e:02d}: {np.array2string(cset)}\n")
            f.write("edge counts (true / estimated active time steps):\n")
            for e, edge in enumerate(model.edges):
                f.write(f"  edge {e:02d} {edge}: true={true_counts[e]}, est={est_counts[e]}\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
