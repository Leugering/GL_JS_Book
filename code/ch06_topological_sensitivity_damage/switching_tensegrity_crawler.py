from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results_switching_tensegrity_crawler")
os.makedirs(OUT_DIR, exist_ok=True)

Array = np.ndarray
Edge = Tuple[int, int]


BARS: List[Edge] = [(0, 3), (1, 4), (2, 5)]
PASSIVE_STRINGS: List[Edge] = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 2), (3, 5)]
ACTIVE_GROUP_A: List[Edge] = [(0, 4), (1, 5)]
ACTIVE_GROUP_B: List[Edge] = [(1, 3), (2, 4)]
FEET = [3, 4, 5]


def make_initial_positions() -> Array:
    return np.array(
        [
            [0.05, 1.12],
            [1.35, 1.38],
            [2.75, 1.12],
            [0.00, 0.04],
            [1.42, 0.08],
            [2.88, 0.04],
        ],
        dtype=float,
    )


def edge_lengths(X: Array, edges: Sequence[Edge]) -> Array:
    return np.array([np.linalg.norm(X[j] - X[i]) for i, j in edges], dtype=float)


def center_of_mass(X: Array, m: Array) -> Array:
    return np.sum(m[:, None] * X, axis=0) / max(float(np.sum(m)), 1.0e-12)


@dataclass
class ContactParameters:
    k_n: float = 4200.0
    c_n: float = 34.0
    mu_slide: float = 0.12
    anchor_k: float = 950.0
    anchor_c: float = 30.0
    v_reg: float = 2.0e-3


@dataclass
class CrawlerParameters:
    dt: float = 0.0025
    T: float = 8.0
    mass: float = 0.55
    top_mass: float = 0.80
    gravity: float = 9.81
    k_bar: float = 1800.0
    k_passive: float = 220.0
    k_active: float = 560.0
    damping: float = 2.8
    velocity_drag: float = 0.020
    phase_period: float = 0.78
    active_scale: float = 0.62
    shape_blend: float = 0.055
    radial_shape_blend: float = 0.12
    min_height_ratio: float = 0.82
    top_height_gain: float = 0.0
    foot_spread_gain: float = 0.0
    allow_switching: bool = True
    frozen_phase: int = 0
    allow_contact_switching: bool = True
    contact: ContactParameters = field(default_factory=ContactParameters)


class SwitchingTensegrityCrawler:
    def __init__(self, params: CrawlerParameters) -> None:
        self.p = params
        self.X0 = make_initial_positions()
        self.X = self.X0.copy()
        self.V = np.zeros_like(self.X)
        self.m = self.p.mass * np.ones(len(self.X), dtype=float)
        self.m[:3] = self.p.top_mass
        self.bar_L0 = edge_lengths(self.X0, BARS)
        self.passive_L0 = edge_lengths(self.X0, PASSIVE_STRINGS)
        self.active_L0 = {
            "A": edge_lengths(self.X0, ACTIVE_GROUP_A),
            "B": edge_lengths(self.X0, ACTIVE_GROUP_B),
        }
        self.ref_shape = self.X0 - center_of_mass(self.X0, self.m)[None, :]
        self.ref_radii = np.linalg.norm(self.ref_shape, axis=1)
        self.ref_height = float(np.max(self.X0[:, 1]) - np.min(self.X0[:, 1]))
        self.ref_top_from_feet = self.X0[:3] - np.mean(self.X0[FEET], axis=0)[None, :]
        self.ref_foot_spread = self.X0[5, 0] - self.X0[3, 0]
        self.anchor_x: Dict[int, float] = {}
        self.prev_phase = -1
        self.phase_history: List[int] = []
        self.com_history: List[Array] = []
        self.anchor_history: List[Array] = []

    def phase_index(self, t: float) -> int:
        if not self.p.allow_switching:
            return self.p.frozen_phase
        return int(math.floor(t / self.p.phase_period)) % 2

    def contact_phase_index(self, t: float) -> int:
        if not self.p.allow_contact_switching:
            return self.p.frozen_phase
        return int(math.floor(t / self.p.phase_period)) % 2

    def active_edges(self, phase: int) -> List[Edge]:
        return ACTIVE_GROUP_A if phase == 0 else ACTIVE_GROUP_B

    def anchored_feet(self, phase: int) -> List[int]:
        return [3, 4] if phase == 0 else [4, 5]

    def update_switching_state(self, t: float) -> None:
        phase = self.phase_index(t)
        contact_phase = self.contact_phase_index(t)
        if phase == self.prev_phase:
            if self.anchor_x and self.anchored_feet(contact_phase) == list(self.anchor_x.keys()):
                return
        self.prev_phase = phase
        anchors = self.anchored_feet(contact_phase)
        self.anchor_x = {}
        for foot in anchors:
            self.anchor_x[foot] = float(self.X[foot, 0])

    def spring_force(self, edge: Edge, rest: float, stiffness: float, tension_only: bool) -> Array:
        i, j = edge
        d = self.X[j] - self.X[i]
        L = np.linalg.norm(d)
        if L < 1.0e-10:
            return np.zeros(2)
        stretch = L - rest
        if tension_only and stretch <= 0.0:
            return np.zeros(2)
        return stiffness * stretch * d / L

    def regularized_coulomb(self, vx: float, fn: float) -> float:
        p = self.p.contact
        return -p.mu_slide * fn * vx / math.sqrt(vx * vx + p.v_reg * p.v_reg)

    def forces(self, t: float) -> Array:
        self.update_switching_state(t)
        phase = self.prev_phase
        F = np.zeros_like(self.X)

        for k, edge in enumerate(BARS):
            fij = self.spring_force(edge, self.bar_L0[k], self.p.k_bar, tension_only=False)
            i, j = edge
            F[i] += fij
            F[j] -= fij

        for k, edge in enumerate(PASSIVE_STRINGS):
            fij = self.spring_force(edge, self.passive_L0[k], self.p.k_passive, tension_only=True)
            i, j = edge
            F[i] += fij
            F[j] -= fij

        active_name = "A" if phase == 0 else "B"
        active_edges = ACTIVE_GROUP_A if phase == 0 else ACTIVE_GROUP_B
        active_rest = self.p.active_scale * self.active_L0[active_name]
        for k, edge in enumerate(active_edges):
            fij = self.spring_force(edge, active_rest[k], self.p.k_active, tension_only=True)
            i, j = edge
            F[i] += fij
            F[j] -= fij

        contact = self.p.contact
        anchors = self.anchored_feet(self.contact_phase_index(t))
        anchor_mask = np.zeros(len(self.X), dtype=float)
        anchor_mask[anchors] = 1.0
        for i in FEET:
            y = self.X[i, 1]
            vy = self.V[i, 1]
            pen = max(0.0, -y)
            if pen <= 0.0:
                continue
            fn = contact.k_n * pen + contact.c_n * max(0.0, -vy)
            F[i, 1] += fn
            if i in self.anchor_x:
                dx = self.X[i, 0] - self.anchor_x[i]
                F[i, 0] += -contact.anchor_k * dx - contact.anchor_c * self.V[i, 0]
            else:
                F[i, 0] += self.regularized_coulomb(float(self.V[i, 0]), fn)

        F[:, 1] -= self.m * self.p.gravity
        F -= self.p.damping * self.V
        F -= self.p.velocity_drag * self.V

        # Mild shape regularization keeps the structure tensegrity-like rather than collapsing into a chain.
        c = center_of_mass(self.X, self.m)
        rel = self.X - c[None, :]
        target = c[None, :] + (1.0 - self.p.shape_blend) * rel + self.p.shape_blend * self.ref_shape
        F += 28.0 * (target - self.X)

        # Keep the top triangle lifted relative to the current support feet.
        feet_center = np.mean(self.X[FEET], axis=0)
        target_top = feet_center[None, :] + self.ref_top_from_feet
        F[:3] += self.p.top_height_gain * (target_top - self.X[:3])

        # Resist horizontal foot collapse so the crawler remains legible in snapshots.
        foot_spread = self.X[5, 0] - self.X[3, 0]
        spread_err = foot_spread - self.ref_foot_spread
        F[3, 0] += self.p.foot_spread_gain * spread_err
        F[5, 0] -= self.p.foot_spread_gain * spread_err
        return F

    def step(self, t: float) -> None:
        F = self.forces(t)
        self.V += self.p.dt * F / self.m[:, None]
        self.X += self.p.dt * self.V
        self.apply_shape_constraint()
        below = self.X[:, 1] < -0.015
        if np.any(below):
            self.X[below, 1] = -0.015
            self.V[below, 1] = np.maximum(self.V[below, 1], 0.0)

    def apply_shape_constraint(self) -> None:
        c = center_of_mass(self.X, self.m)
        rel = self.X - c[None, :]
        radii = np.linalg.norm(rel, axis=1)
        dirs = rel / np.maximum(radii[:, None], 1.0e-12)

        # Keep the tensegrity shell near its reference radial profile without freezing the gait.
        target_rel = self.ref_radii[:, None] * dirs
        self.X[:] = c[None, :] + (1.0 - self.p.radial_shape_blend) * rel + self.p.radial_shape_blend * target_rel

        # Prevent vertical pancaking by maintaining a minimum fraction of the initial height.
        ymin = float(np.min(self.X[:, 1]))
        ymax = float(np.max(self.X[:, 1]))
        height = ymax - ymin
        min_height = self.p.min_height_ratio * self.ref_height
        if 0.0 < height < min_height:
            cy = float(c[1])
            rel_y = self.X[:, 1] - cy
            scale = min_height / max(height, 1.0e-12)
            self.X[:, 1] = cy + scale * rel_y

        # Dampen purely radial collapse velocities generated by the switching phases.
        c = center_of_mass(self.X, self.m)
        rel = self.X - c[None, :]
        dirs = rel / np.maximum(np.linalg.norm(rel, axis=1)[:, None], 1.0e-12)
        radial_v = np.sum(self.V * dirs, axis=1)
        self.V[:] = self.V - 0.08 * radial_v[:, None] * dirs

    def simulate(self) -> Tuple[Array, Array, Array, Array, Array]:
        n_steps = int(np.ceil(self.p.T / self.p.dt))
        ts: List[float] = [0.0]
        traj: List[Array] = [self.X.copy()]
        coms: List[Array] = [center_of_mass(self.X, self.m)]
        phases: List[int] = [self.phase_index(0.0)]
        anchors: List[Array] = [np.array(self.anchored_feet(self.contact_phase_index(0.0)), dtype=int)]
        sample_every = 6
        for k in range(1, n_steps + 1):
            t_prev = (k - 1) * self.p.dt
            self.step(t_prev)
            if k % sample_every == 0 or k == n_steps:
                t = min(k * self.p.dt, self.p.T)
                ts.append(t)
                traj.append(self.X.copy())
                coms.append(center_of_mass(self.X, self.m))
                phases.append(self.phase_index(t))
                anchors.append(np.array(self.anchored_feet(self.contact_phase_index(t)), dtype=int))
        return np.array(ts), np.array(traj), np.array(coms), np.array(phases), np.array(anchors, dtype=object)


def draw_robot(ax: plt.Axes, X: Array, phase: int, initial: Array | None = None) -> None:
    if initial is not None:
        for edge in PASSIVE_STRINGS + BARS:
            i, j = edge
            ax.plot([initial[i, 0], initial[j, 0]], [initial[i, 1], initial[j, 1]], color="#d9d9d9", linewidth=1.0, alpha=0.8)
        ax.scatter(initial[:, 0], initial[:, 1], color="#c7c7c7", s=20, zorder=2)

    for edge in PASSIVE_STRINGS:
        i, j = edge
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="tab:green", linewidth=1.3, alpha=0.9)
    for edge in BARS:
        i, j = edge
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="tab:red", linewidth=3.0, alpha=0.9)
    active = ACTIVE_GROUP_A if phase == 0 else ACTIVE_GROUP_B
    active_color = "tab:orange" if phase == 0 else "tab:purple"
    for edge in active:
        i, j = edge
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color=active_color, linewidth=2.6, alpha=0.95)
    ax.scatter(X[:3, 0], X[:3, 1], color="black", s=32, zorder=3)
    ax.scatter(X[3:, 0], X[3:, 1], color="tab:blue", s=38, zorder=3)
    ax.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")


def save_gif(ts: Array, traj: Array, coms: Array, phases: Array) -> str:
    ids = np.linspace(0, len(ts) - 1, 20, dtype=int)
    x_all = traj[:, :, 0]
    y_all = traj[:, :, 1]
    xlim = (float(np.min(x_all)) - 0.35, float(np.max(x_all)) + 0.35)
    ylim = (-0.15, float(np.max(y_all)) + 0.35)
    frames: List[Image.Image] = []
    X0 = traj[0]
    for sid in ids:
        fig, axs = plt.subplots(2, 1, figsize=(8.6, 6.0), height_ratios=[3.0, 1.4])
        draw_robot(axs[0], traj[sid], int(phases[sid]), initial=X0)
        axs[0].set_xlim(*xlim)
        axs[0].set_ylim(*ylim)
        axs[0].set_title(f"Switching tensegrity crawler, t={ts[sid]:.2f}, phase={int(phases[sid])}")

        axs[1].plot(ts[: sid + 1], coms[: sid + 1, 0], color="crimson", linewidth=2.0, label="COM x")
        axs[1].plot(ts[: sid + 1], coms[: sid + 1, 1], color="royalblue", linewidth=2.0, label="COM y")
        axs[1].step(ts[: sid + 1], phases[: sid + 1], where="post", color="black", linewidth=1.3, label="phase")
        axs[1].grid(alpha=0.25)
        axs[1].legend(loc="best", fontsize=8)
        axs[1].set_xlabel("time")

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    path = os.path.join(OUT_DIR, "switching_tensegrity_crawler_animation.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    return path


def save_snapshots(ts: Array, traj: Array, phases: Array) -> str:
    ids = np.linspace(0, len(ts) - 1, 6, dtype=int)
    x_all = traj[:, :, 0]
    y_all = traj[:, :, 1]
    xspan = float(np.max(x_all) - np.min(x_all))
    fig, axs = plt.subplots(2, 3, figsize=(12.4, 6.8))
    X0 = traj[0]
    for ax, sid in zip(axs.flat, ids):
        X = traj[sid]
        draw_robot(ax, X, int(phases[sid]), initial=X0)
        cx = float(np.mean(X[:, 0]))
        ax.set_xlim(cx - 1.8, cx + 1.8)
        ax.set_ylim(-0.12, float(np.max(y_all)) + 0.28)
        ax.set_title(f"t={ts[sid]:.2f}, phase={int(phases[sid])}")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "switching_tensegrity_crawler_snapshots.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_pose_only(ts: Array, traj: Array, phases: Array) -> str:
    ids = np.linspace(0, len(ts) - 1, 5, dtype=int)
    fig, axs = plt.subplots(1, 5, figsize=(15.0, 3.6))
    X0 = traj[0]
    y_all = traj[:, :, 1]
    for ax, sid in zip(axs.flat, ids):
        X = traj[sid]
        draw_robot(ax, X, int(phases[sid]), initial=X0)
        cx = float(np.mean(X[:, 0]))
        ax.set_xlim(cx - 1.55, cx + 1.55)
        ax.set_ylim(-0.08, float(np.max(y_all)) + 0.20)
        ax.set_title(f"t={ts[sid]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "switching_tensegrity_crawler_pose_only.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_diagnostics(ts: Array, traj: Array, coms: Array, phases: Array) -> str:
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axs[0].plot(ts, coms[:, 0], color="crimson", linewidth=2.0, label="COM x")
    axs[0].plot(ts, coms[:, 1], color="royalblue", linewidth=2.0, label="COM y")
    axs[0].set_title("Center of Mass")
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc="best")

    active_lengths_a = np.array([edge_lengths(X, ACTIVE_GROUP_A) for X in traj])
    active_lengths_b = np.array([edge_lengths(X, ACTIVE_GROUP_B) for X in traj])
    axs[1].plot(ts, np.mean(active_lengths_a, axis=1), color="tab:orange", linewidth=2.0, label="group A")
    axs[1].plot(ts, np.mean(active_lengths_b, axis=1), color="tab:purple", linewidth=2.0, label="group B")
    axs[1].set_title("Switching Cable Lengths")
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="best")

    axs[2].step(ts, phases, where="post", color="black", linewidth=1.6, label="active cable set")
    axs[2].plot(ts, traj[:, 3, 0], color="#1f77b4", alpha=0.9, label="rear foot x")
    axs[2].plot(ts, traj[:, 5, 0], color="#2ca02c", alpha=0.9, label="front foot x")
    axs[2].set_title("Hybrid Phase And Feet")
    axs[2].grid(alpha=0.25)
    axs[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "switching_tensegrity_crawler_diagnostics.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_switching_comparison(
    ts_sw: Array,
    com_sw: Array,
    phases_sw: Array,
    ts_f0: Array,
    com_f0: Array,
    ts_f1: Array,
    com_f1: Array,
) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axs[0].plot(ts_sw, com_sw[:, 0], color="crimson", linewidth=2.2, label="switching COM x")
    axs[0].plot(ts_f0, com_f0[:, 0], color="black", linewidth=1.9, linestyle="--", label="frozen A COM x")
    axs[0].plot(ts_f1, com_f1[:, 0], color="tab:gray", linewidth=1.9, linestyle="-.", label="frozen B COM x")
    axs[0].set_title("Switching vs Frozen Topologies")
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc="best", fontsize=8)

    dx_sw = float(com_sw[-1, 0] - com_sw[0, 0])
    dx_f0 = float(com_f0[-1, 0] - com_f0[0, 0])
    dx_f1 = float(com_f1[-1, 0] - com_f1[0, 0])
    dy_sw = float(com_sw[-1, 1] - com_sw[0, 1])
    dy_f0 = float(com_f0[-1, 1] - com_f0[0, 1])
    dy_f1 = float(com_f1[-1, 1] - com_f1[0, 1])
    axs[1].bar([0, 1, 2], [dx_sw, dx_f0, dx_f1], color=["crimson", "black", "tab:gray"], alpha=0.85)
    axs[1].bar([4, 5, 6], [dy_sw, dy_f0, dy_f1], color=["royalblue", "dimgray", "lightgray"], alpha=0.85)
    axs[1].set_xticks([0, 1, 2, 4, 5, 6])
    axs[1].set_xticklabels(["dx switch", "dx frozen A", "dx frozen B", "dy switch", "dy frozen A", "dy frozen B"])
    axs[1].set_title("Net Displacement Comparison")
    axs[1].grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "switching_tensegrity_crawler_topology_comparison.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    switching_crawler = SwitchingTensegrityCrawler(CrawlerParameters())
    ts, traj, coms, phases, _ = switching_crawler.simulate()
    frozen_crawler_0 = SwitchingTensegrityCrawler(
        CrawlerParameters(allow_switching=False, frozen_phase=0, allow_contact_switching=True)
    )
    ts_f0, traj_f0, coms_f0, phases_f0, _ = frozen_crawler_0.simulate()
    frozen_crawler_1 = SwitchingTensegrityCrawler(
        CrawlerParameters(allow_switching=False, frozen_phase=1, allow_contact_switching=True)
    )
    ts_f1, traj_f1, coms_f1, phases_f1, _ = frozen_crawler_1.simulate()

    gif_path = save_gif(ts, traj, coms, phases)
    snapshots_path = save_snapshots(ts, traj, phases)
    pose_only_path = save_pose_only(ts, traj, phases)
    diagnostics_path = save_diagnostics(ts, traj, coms, phases)
    comparison_path = save_switching_comparison(ts, coms, phases, ts_f0, coms_f0, ts_f1, coms_f1)
    disp = coms[-1] - coms[0]
    disp_f0 = coms_f0[-1] - coms_f0[0]
    disp_f1 = coms_f1[-1] - coms_f1[0]

    print("Simulation finished: switching tensegrity crawler")
    print(f"COM displacement: dx={disp[0]:.6f}, dy={disp[1]:.6f}")
    print(f"Animation saved in: {gif_path}")
    print(f"Snapshots:          {snapshots_path}")
    print(f"Pose-only figure:   {pose_only_path}")
    print(f"Diagnostics:        {diagnostics_path}")
    print(f"Topology compare:   {comparison_path}")
    print(f"Frozen-topology A displacement: dx={disp_f0[0]:.6f}, dy={disp_f0[1]:.6f}")
    print(f"Frozen-topology B displacement: dx={disp_f1[0]:.6f}, dy={disp_f1[1]:.6f}")


if __name__ == "__main__":
    main()
