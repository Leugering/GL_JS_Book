from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results_arch_bridge_crawler_shape_control")
os.makedirs(OUT_DIR, exist_ok=True)

Array = np.ndarray
Edge = Tuple[int, int]


@dataclass
class ContactParameters:
    k_n: float = 4.5e3
    c_n: float = 40.0
    mu: float = 0.95
    v_reg: float = 3.0e-3


@dataclass
class SimulationParameters:
    gravity: float = 9.81
    spring_stiffness: float = 280.0
    damping: float = 0.9
    mass: float = 1.0
    velocity_drag: float = 0.028
    radial_stiffness: float = 0.0
    contact: ContactParameters = field(default_factory=ContactParameters)


@dataclass
class ArchShapeTrackingControl:
    controlled_edges: Sequence[Edge]
    edge_to_bays: Dict[Edge, List[int]]
    base_length: Dict[Edge, float]
    base_arch_profile: Array
    edge_drive_weight: Dict[Edge, float]
    wave_amplitude: float = 0.10
    wave_frequency: float = 0.90
    shape_gain: float = 0.18
    damping_gain: float = 0.035
    travel_gain: float = 0.025
    target_speed_x: float = 0.060
    bias_gain: float = 0.15
    wave_gain: float = 0.060
    dynamic_delta: Dict[Edge, float] = field(default_factory=dict)

    def target_profile(self, t: float) -> Array:
        n = len(self.base_arch_profile)
        xi = np.linspace(0.0, 1.0, n)
        phase = 2.0 * math.pi * (self.wave_frequency * t - xi)
        bump = self.wave_amplitude * np.sin(phase)
        taper = 0.6 + 0.4 * np.sin(math.pi * xi)
        return self.base_arch_profile * (1.0 + taper * bump)

    def update_state(self, model: "ArchBridgeCrawler", x: Array, v: Array, t: float) -> None:
        heights = model.arch_heights(x)
        vel_heights = model.arch_height_rates(v)
        target = self.target_profile(t)
        err = heights - target
        com_vx = model.center_of_mass_velocity(v)[0]
        travel_bias = self.travel_gain * max(0.0, self.target_speed_x - com_vx)

        self.dynamic_delta = {}
        for edge in self.controlled_edges:
            bays = self.edge_to_bays.get(edge, [])
            if not bays:
                self.dynamic_delta[edge] = 0.0
                continue
            mean_err = float(np.mean(err[bays]))
            mean_rate = float(np.mean(vel_heights[bays]))
            xi = float(np.mean(bays)) / max(len(self.base_arch_profile) - 1, 1)
            phase = 2.0 * math.pi * (self.wave_frequency * t - xi)
            drive_env = 0.65 + 0.9 * (1.0 - xi)
            wave_drive = self.wave_gain * self.edge_drive_weight.get(edge, 1.0) * drive_env * math.sin(phase)
            delta = (
                -self.shape_gain * mean_err
                - self.damping_gain * mean_rate
                + self.bias_gain * travel_bias
                + wave_drive
            )
            self.dynamic_delta[edge] = float(np.clip(delta, -0.20, 0.20))

    def rest_length(self, t: float, edge: Edge, ell0: float) -> float:
        if edge not in self.base_length and (edge[1], edge[0]) not in self.base_length:
            return ell0
        e = edge if edge in self.base_length else (edge[1], edge[0])
        base = self.base_length[e]
        delta = self.dynamic_delta.get(e, 0.0)
        return base * (1.0 + delta)


@dataclass
class ArchBridgeCrawler:
    x0: Array
    lower_nodes: List[int]
    upper_nodes: List[int]
    edges: List[Edge]
    edge_lengths0: Array
    masses: Array
    params: SimulationParameters

    @property
    def n_nodes(self) -> int:
        return self.x0.shape[0]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def center_of_mass(self, x: Array) -> Array:
        m = np.sum(self.masses)
        return np.sum(self.masses[:, None] * x, axis=0) / max(m, 1e-12)

    def center_of_mass_velocity(self, v: Array) -> Array:
        m = np.sum(self.masses)
        return np.sum(self.masses[:, None] * v, axis=0) / max(m, 1e-12)

    def arch_heights(self, x: Array) -> Array:
        vals = []
        for k, u in enumerate(self.upper_nodes):
            l0 = self.lower_nodes[k]
            l1 = self.lower_nodes[k + 1]
            deck_mid_y = 0.5 * (x[l0, 1] + x[l1, 1])
            vals.append(x[u, 1] - deck_mid_y)
        return np.array(vals, dtype=float)

    def arch_height_rates(self, v: Array) -> Array:
        vals = []
        for k, u in enumerate(self.upper_nodes):
            l0 = self.lower_nodes[k]
            l1 = self.lower_nodes[k + 1]
            deck_mid_v = 0.5 * (v[l0, 1] + v[l1, 1])
            vals.append(v[u, 1] - deck_mid_v)
        return np.array(vals, dtype=float)


def regularized_coulomb(v_tan: Array, normal_force: float, mu: float, v_reg: float) -> Array:
    s = float(np.linalg.norm(v_tan))
    if s < 1e-14:
        return np.zeros(2)
    return -mu * normal_force * v_tan / math.sqrt(s * s + v_reg * v_reg)


def build_arch_bridge_crawler(span: float = 14.0, rise: float = 2.0, n_bays: int = 12) -> Tuple[ArchBridgeCrawler, ArchShapeTrackingControl]:
    xs = np.linspace(0.0, span, n_bays + 1)
    lower_nodes: List[int] = []
    upper_nodes: List[int] = []
    nodes: List[List[float]] = []

    for x in xs:
        lower_nodes.append(len(nodes))
        nodes.append([x, 0.0])

    for k in range(n_bays):
        x_mid = 0.5 * (xs[k] + xs[k + 1])
        arch_y = rise * math.sin(math.pi * x_mid / max(span, 1e-12))
        upper_nodes.append(len(nodes))
        nodes.append([x_mid, arch_y])

    x0 = np.array(nodes, dtype=float)
    edges: List[Edge] = []
    controlled_edges: List[Edge] = []
    edge_to_bays: Dict[Edge, List[int]] = {}
    edge_drive_weight: Dict[Edge, float] = {}

    # Lower chord and upper arch.
    for k in range(n_bays):
        edges.append((lower_nodes[k], lower_nodes[k + 1]))
    for k in range(n_bays - 1):
        e = (upper_nodes[k], upper_nodes[k + 1])
        edges.append(e)
        controlled_edges.append(e)
        edge_to_bays.setdefault(e, []).extend([k, k + 1])
        edge_drive_weight[e] = 0.55

    # Triangulation: each bay is a triangle pair around one arch node.
    for k in range(n_bays):
        u = upper_nodes[k]
        l0 = lower_nodes[k]
        l1 = lower_nodes[k + 1]
        e0 = (l0, u)
        e1 = (u, l1)
        edges.append(e0)
        edges.append(e1)
        controlled_edges.extend([e0, e1])
        edge_to_bays.setdefault(e0, []).append(k)
        edge_to_bays.setdefault(e1, []).append(k)
        edge_drive_weight[e0] = 1.05
        edge_drive_weight[e1] = 1.05

    # Dense diagonals between neighboring bays for a more video-like crawling truss.
    for k in range(n_bays - 1):
        e_left = (lower_nodes[k + 1], upper_nodes[k])
        e_right = (lower_nodes[k + 1], upper_nodes[k + 1])
        for e in (e_left, e_right):
            edges.append(e)
            controlled_edges.append(e)
            edge_to_bays.setdefault(e, []).extend([k, k + 1])
            edge_drive_weight[e] = 1.35

    # Deduplicate while preserving order.
    seen = set()
    dedup_edges: List[Edge] = []
    for e in edges:
        key = tuple(sorted(e))
        if key in seen:
            continue
        seen.add(key)
        dedup_edges.append(e)
    edges = dedup_edges
    controlled_edges = [e for e in dedup_edges if e in edge_to_bays]

    edge_lengths0 = np.array([np.linalg.norm(x0[j] - x0[i]) for i, j in edges], dtype=float)
    masses = np.ones(len(nodes), dtype=float)
    masses[upper_nodes] = 0.80

    params = SimulationParameters()
    model = ArchBridgeCrawler(
        x0=x0,
        lower_nodes=lower_nodes,
        upper_nodes=upper_nodes,
        edges=edges,
        edge_lengths0=edge_lengths0,
        masses=masses,
        params=params,
    )

    base_arch = model.arch_heights(x0)
    base_length = {e: np.linalg.norm(x0[e[1]] - x0[e[0]]) for e in controlled_edges}
    control = ArchShapeTrackingControl(
        controlled_edges=controlled_edges,
        edge_to_bays=edge_to_bays,
        base_length=base_length,
        base_arch_profile=base_arch,
        edge_drive_weight=edge_drive_weight,
        wave_amplitude=0.36,
        wave_frequency=1.25,
        shape_gain=0.12,
        damping_gain=0.04,
        travel_gain=0.12,
        target_speed_x=0.12,
        bias_gain=0.45,
        wave_gain=0.145,
    )
    return model, control


def spring_forces(model: ArchBridgeCrawler, x: Array, v: Array, t: float, control: ArchShapeTrackingControl) -> Array:
    forces = np.zeros_like(x)
    k = model.params.spring_stiffness
    c = model.params.damping
    length_map = {e: L0 for e, L0 in zip(model.edges, model.edge_lengths0)}

    for idx, (i, j) in enumerate(model.edges):
        d = x[j] - x[i]
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            continue
        n = d / L
        ell = control.rest_length(t, (i, j), length_map[(i, j)])
        f_el = k * (L - ell) * n
        rel_v = v[j] - v[i]
        f_damp = c * np.dot(rel_v, n) * n
        f = f_el + f_damp
        forces[i] += f
        forces[j] -= f
    return forces


def ground_forces(model: ArchBridgeCrawler, x: Array, v: Array) -> Array:
    p = model.params.contact
    forces = np.zeros_like(x)
    for i in model.lower_nodes:
        y = x[i, 1]
        vy = v[i, 1]
        pen = max(0.0, -y)
        if pen <= 0.0:
            continue
        fn = p.k_n * pen + p.c_n * max(0.0, -vy)
        forces[i, 1] += fn
        vt = np.array([v[i, 0], 0.0], dtype=float)
        forces[i] += regularized_coulomb(vt, fn, p.mu, p.v_reg)
    return forces


def rhs_step(model: ArchBridgeCrawler, control: ArchShapeTrackingControl, x: Array, v: Array, t: float) -> Array:
    control.update_state(model, x, v, t)
    f_spr = spring_forces(model, x, v, t, control)
    f_ground = ground_forces(model, x, v)
    f_grav = np.zeros_like(x)
    f_grav[:, 1] -= model.masses * model.params.gravity
    total = f_spr + f_ground + f_grav - model.params.velocity_drag * v
    return total / model.masses[:, None]


def simulate(model: ArchBridgeCrawler, control: ArchShapeTrackingControl, T: float = 8.0, dt: float = 0.01, sample_every: int = 2) -> Tuple[Array, Array, Array]:
    n_steps = int(np.ceil(T / dt))
    x = model.x0.copy()
    v = np.zeros_like(x)
    xs: List[Array] = []
    vs: List[Array] = []
    ts: List[float] = []

    def save(t_now: float) -> None:
        ts.append(float(t_now))
        xs.append(x.copy())
        vs.append(v.copy())

    save(0.0)
    for k in range(1, n_steps + 1):
        t_prev = (k - 1) * dt
        a = rhs_step(model, control, x, v, t_prev)
        v = v + dt * a
        x = x + dt * v
        below = x[:, 1] < -0.04
        if np.any(below):
            x[below, 1] = -0.04
            v[below, 1] = np.maximum(v[below, 1], 0.0)
        if k % sample_every == 0 or k == n_steps:
            save(min(k * dt, T))

    return np.array(ts), np.array(xs), np.array(vs)


def center_of_mass_traj(model: ArchBridgeCrawler, xs: Array) -> Array:
    return np.array([model.center_of_mass(x) for x in xs], dtype=float)


def arch_shape_traj(model: ArchBridgeCrawler, xs: Array) -> Array:
    return np.array([model.arch_heights(x) for x in xs], dtype=float)


def shape_tracking_objective(shape: Array, target: Array) -> Array:
    return 0.5 * np.sum((shape - target) ** 2, axis=1)


def save_gif(
    model: ArchBridgeCrawler,
    ts: Array,
    xs: Array,
    com: Array,
    gif_name: str = "arch_bridge_crawler_animation.gif",
    n_frames: int = 18,
) -> str:
    ids = np.linspace(0, len(ts) - 1, n_frames, dtype=int)
    x_all = xs[:, :, 0]
    y_all = xs[:, :, 1]
    xlim = (float(x_all.min()) - 0.4, float(x_all.max()) + 0.4)
    ylim = (-0.25, float(y_all.max()) + 0.5)
    x_disp = com[:, 0] - com[0, 0]
    dx_lim = (
        float(np.min(x_disp)) - 0.002,
        float(np.max(x_disp)) + 0.002,
    )
    x0 = xs[0]

    frames: List[Image.Image] = []
    for sid in ids:
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 4.0))
        x = xs[sid]

        ax = axs[0]
        for i, j in model.edges:
            ax.plot([x0[i, 0], x0[j, 0]], [x0[i, 1], x0[j, 1]], color="#9ecae1", linewidth=1.0, alpha=0.7)
        for i, j in model.edges:
            ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], color="0.55", linewidth=1.0)
        ax.scatter(x0[model.lower_nodes, 0], x0[model.lower_nodes, 1], s=14, color="#6baed6", alpha=0.8)
        ax.scatter(x0[model.upper_nodes, 0], x0[model.upper_nodes, 1], s=18, color="#fcae91", alpha=0.8)
        ax.scatter(x[model.lower_nodes, 0], x[model.lower_nodes, 1], s=24, color="black")
        ax.scatter(x[model.upper_nodes, 0], x[model.upper_nodes, 1], s=30, color="crimson")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.plot(com[: sid + 1, 0], com[: sid + 1, 1], color="darkgreen", linewidth=1.6, alpha=0.9)
        ax.scatter(com[0, 0], com[0, 1], s=24, color="tab:blue", label="initial COM")
        ax.scatter(com[sid, 0], com[sid, 1], s=34, color="darkgreen", label="current COM")
        ax.annotate(
            "",
            xy=(com[sid, 0], ylim[0] + 0.12),
            xytext=(com[0, 0], ylim[0] + 0.12),
            arrowprops=dict(arrowstyle="->", color="darkgreen", linewidth=2.0),
        )
        ax.text(
            0.03,
            0.96,
            f"dx = {x_disp[sid]:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"World frame, t={ts[sid]:.2f}")
        ax.grid(alpha=0.2)
        if sid == ids[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axs[1]
        ax.plot(ts[: sid + 1], x_disp[: sid + 1], color="royalblue", linewidth=2.0)
        ax.scatter(ts[sid], x_disp[sid], s=50, color="crimson")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlim(float(ts[0]), float(ts[-1]))
        ax.set_ylim(*dx_lim)
        ax.set_title("Accumulated rightward progress")
        ax.set_xlabel("time")
        ax.set_ylabel("COM dx")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    gif_path = os.path.join(OUT_DIR, gif_name)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)
    return gif_path


def save_static_snapshots(
    model: ArchBridgeCrawler,
    ts: Array,
    xs: Array,
    com: Array,
    fig_name: str,
    n_snapshots: int = 6,
) -> str:
    ids = np.linspace(0, len(ts) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ids)))
    x0 = xs[0]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, j in model.edges:
        ax.plot([x0[i, 0], x0[j, 0]], [x0[i, 1], x0[j, 1]], color="#d0d8e8", linewidth=0.9, alpha=0.8)

    for c, sid in zip(colors, ids):
        x = xs[sid]
        for i, j in model.edges:
            ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], color=c, linewidth=1.15, alpha=0.65)
        ax.scatter(x[model.lower_nodes, 0], x[model.lower_nodes, 1], s=16, color=c, alpha=0.8)
        ax.scatter(x[model.upper_nodes, 0], x[model.upper_nodes, 1], s=20, color=c, alpha=0.8)
        ax.text(
            float(np.mean(x[:, 0])),
            float(np.max(x[:, 1]) + 0.18),
            f"t={ts[sid]:.1f}",
            color=c,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.plot(com[:, 0], com[:, 1], color="black", linewidth=2.0, label="COM trace")
    ax.scatter(com[ids, 0], com[ids, 1], s=34, color=colors, zorder=5)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title("Arch-bridge crawler at multiple times")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, fig_name)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def demo(T: float = 8.0, prefix: str = "arch_bridge_crawler", n_gif_frames: int = 18) -> None:
    model, control = build_arch_bridge_crawler()
    ts, xs, vs = simulate(model, control, T=T, dt=0.01, sample_every=2)
    com = center_of_mass_traj(model, xs)
    shape = arch_shape_traj(model, xs)
    targets = np.array([control.target_profile(t) for t in ts], dtype=float)
    Jshape = shape_tracking_objective(shape, targets)
    gif_path = save_gif(model, ts, xs, com, gif_name=f"{prefix}_animation.gif", n_frames=n_gif_frames)
    static_path = save_static_snapshots(model, ts, xs, com, fig_name=f"{prefix}_static_times.png")

    disp = com[-1] - com[0]
    avg_speed = disp[0] / max(ts[-1] - ts[0], 1e-12)

    print("Simulation finished: arch-bridge crawler")
    print(f"Nodes: {model.n_nodes}, edges: {model.n_edges}")
    print(f"Time horizon: T={T:.2f}")
    print(f"COM displacement: dx={disp[0]:.6f}, dy={disp[1]:.6f}")
    print(f"Average rightward speed: {avg_speed:.6f}")
    print(f"Final shape-tracking energy: {Jshape[-1]:.6f}")
    print(f"Animation saved in: {gif_path}")
    print(f"Static snapshot figure saved in: {static_path}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    x0 = model.x0
    ax = axs[0]
    for i, j in model.edges:
        ax.plot([x0[i, 0], x0[j, 0]], [x0[i, 1], x0[j, 1]], color="0.7", linewidth=1.0)
    ax.scatter(x0[model.lower_nodes, 0], x0[model.lower_nodes, 1], color="black", s=22)
    ax.scatter(x0[model.upper_nodes, 0], x0[model.upper_nodes, 1], color="crimson", s=26)
    ax.set_aspect("equal")
    ax.set_title("Initial triangulated arch-bridge graph")
    ax.grid(alpha=0.25)

    ax = axs[1]
    ax.plot(ts, com[:, 0], color="crimson", label="COM x")
    ax.plot(ts, com[:, 1], color="royalblue", label="COM y")
    ax.plot(ts, Jshape, color="darkgreen", label="shape objective")
    ax.set_title("Locomotion and shape tracking")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{prefix}_control_and_com.png"), dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax = axs[0]
    for node_id in range(model.n_nodes):
        ax.plot(xs[:, node_id, 0], xs[:, node_id, 1], color="0.8", linewidth=0.7)
    ax.plot(com[:, 0], com[:, 1], color="black", linewidth=2.0, label="COM")
    ax.set_aspect("equal")
    ax.set_title("Crawler footprint")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axs[1]
    ids = np.linspace(0, len(ts) - 1, 5, dtype=int)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ids)))
    for c, sid in zip(colors, ids):
        x = xs[sid]
        for i, j in model.edges:
            ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], color=c, linewidth=0.8, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title("Shape snapshots")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{prefix}_footprint_and_snapshots.png"), dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    ax = axs[0]
    for k in range(shape.shape[1]):
        ax.plot(ts, shape[:, k], linewidth=1.2, label=f"h{k}")
        ax.plot(ts, targets[:, k], linestyle="--", linewidth=0.9, color="0.6")
    ax.set_title("Arch-height tracking")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2)

    ax = axs[1]
    ax.plot(ts, Jshape, color="darkgreen", linewidth=2.0)
    ax.set_title("Shape-tracking objective")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{prefix}_shape_tracking.png"), dpi=180)
    plt.close(fig)

    print(f"Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    T = float(os.environ.get("ARCH_CRAWLER_T", "8.0"))
    prefix = os.environ.get("ARCH_CRAWLER_PREFIX", "arch_bridge_crawler")
    n_gif_frames = int(os.environ.get("ARCH_CRAWLER_GIF_FRAMES", "18"))
    demo(T=T, prefix=prefix, n_gif_frames=n_gif_frames)
