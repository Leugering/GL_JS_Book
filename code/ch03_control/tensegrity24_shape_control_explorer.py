from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results_tensegrity24_shape_control")
os.makedirs(OUT_DIR, exist_ok=True)

Edge = Tuple[int, int]
Array = np.ndarray


# ============================================================
# 6-bar / 24-string desk tensegrity
# ============================================================

BARS: List[Edge] = [(2 * k, 2 * k + 1) for k in range(6)]


def make_string_pattern() -> List[Edge]:
    offsets = (1, 2, 3, 5)
    strings: List[Edge] = []
    for k in range(6):
        top = 2 * k
        for s in offsets:
            bottom = 2 * ((k + s) % 6) + 1
            strings.append((top, bottom))
    return strings


STRINGS: List[Edge] = make_string_pattern()

assert len(STRINGS) == 24


def make_initial_positions(radius: float = 1.35, half_height: float = 0.95, twist: float = np.pi / 3.0) -> Array:
    X = np.zeros((12, 3), dtype=float)
    for k in range(6):
        theta = 2.0 * np.pi * k / 6.0
        X[2 * k] = np.array(
            [radius * np.cos(theta), radius * np.sin(theta), half_height],
            dtype=float,
        )
        X[2 * k + 1] = np.array(
            [
                radius * np.cos(theta + twist),
                radius * np.sin(theta + twist),
                -half_height,
            ],
            dtype=float,
        )
    return X


def edge_lengths(X: Array, edges: Sequence[Edge]) -> Array:
    return np.array([np.linalg.norm(X[j] - X[i]) for i, j in edges], dtype=float)


def center(X: Array) -> Array:
    return X - np.mean(X, axis=0, keepdims=True)


def segments(X: Array, edges: Sequence[Edge]) -> List[List[Array]]:
    return [[X[i], X[j]] for i, j in edges]


def midpoint_groups(X: Array, edges: Sequence[Edge]) -> Dict[str, Array]:
    mids = np.array([0.5 * (X[i] + X[j]) for i, j in edges], dtype=float)
    x = mids[:, 0]
    y = mids[:, 1]
    z = mids[:, 2]
    eps = 1.0e-8
    return {
        "all": np.ones(len(edges), dtype=bool),
        "x_pos": x > eps,
        "x_neg": x < -eps,
        "y_pos": y > eps,
        "y_neg": y < -eps,
        "z_pos": z > eps,
        "z_neg": z < -eps,
        "xy_pos": x * y > eps,
        "xy_neg": x * y < -eps,
    }


@dataclass
class ControlScenario:
    name: str
    description: str
    multiplicative_rest_factors: Array


class TensegritySimulator:
    def __init__(self, X0: Array) -> None:
        self.X = X0.copy()
        self.V = np.zeros_like(X0)

        self.dt = 0.002
        self.m = 1.0

        self.k_bar = 420.0
        self.k_string = 70.0
        self.damping = 4.0

        self.bar_L0 = edge_lengths(self.X, BARS)
        self.string_L0 = edge_lengths(self.X, STRINGS)
        self.string_rest = 0.90 * self.string_L0.copy()

    def forces(self) -> Array:
        F = np.zeros_like(self.X)

        for k, (i, j) in enumerate(BARS):
            d = self.X[j] - self.X[i]
            L = np.linalg.norm(d)
            if L < 1.0e-10:
                continue
            n = d / L
            fij = self.k_bar * (L - self.bar_L0[k]) * n
            F[i] += fij
            F[j] -= fij

        for k, (i, j) in enumerate(STRINGS):
            d = self.X[j] - self.X[i]
            L = np.linalg.norm(d)
            if L < 1.0e-10:
                continue
            n = d / L
            stretch = L - self.string_rest[k]
            if stretch > 0.0:
                fij = self.k_string * stretch * n
                F[i] += fij
                F[j] -= fij

        F -= self.damping * self.V
        return F

    def step(self) -> None:
        F = self.forces()
        self.V += self.dt * F / self.m
        self.X += self.dt * self.V
        self.X -= np.mean(self.X, axis=0, keepdims=True)
        self.V -= np.mean(self.V, axis=0, keepdims=True)

    def set_string_rest_factors(self, factors: Array) -> None:
        self.string_rest = np.clip(factors, 0.72, 1.18) * self.string_L0

    def run_to_equilibrium(self, steps: int = 9000, tol: float = 2.0e-4) -> Array:
        last_X = self.X.copy()
        for k in range(steps):
            self.step()
            if k % 200 == 0:
                drift = np.linalg.norm(self.X - last_X) / max(np.sqrt(self.X.size), 1.0)
                if drift < tol and np.linalg.norm(self.V) < 2.5e-3:
                    break
                last_X = self.X.copy()
        return self.X.copy()


def make_scenarios(X0: Array) -> List[ControlScenario]:
    groups = midpoint_groups(X0, STRINGS)
    base = 0.90 * np.ones(len(STRINGS), dtype=float)

    def factors(*ops: Tuple[str, float]) -> Array:
        vals = base.copy()
        for name, delta in ops:
            vals[groups[name]] += delta
        return vals

    return [
        ControlScenario("baseline", "uniform prestress", base.copy()),
        ControlScenario("uniform_contract", "all strings shortened", factors(("all", -0.08))),
        ControlScenario("z_lift", "upper strings contract, lower strings relax", factors(("z_pos", -0.08), ("z_neg", 0.05))),
        ControlScenario("x_bend", "right/left asymmetric x-bending", factors(("x_pos", -0.08), ("x_neg", 0.05))),
        ControlScenario("y_bend", "front/back asymmetric y-bending", factors(("y_pos", -0.08), ("y_neg", 0.05))),
        ControlScenario("twist", "opposite diagonal twist", factors(("xy_pos", -0.08), ("xy_neg", 0.05))),
        ControlScenario("saddle", "top contraction plus lateral skew", factors(("z_pos", -0.05), ("x_pos", -0.05), ("y_neg", 0.04))),
    ]


def shape_descriptors(X: Array) -> Dict[str, float]:
    Xc = center(X)
    span = np.max(Xc, axis=0) - np.min(Xc, axis=0)
    cov = Xc.T @ Xc / max(Xc.shape[0], 1)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return {
        "span_x": float(span[0]),
        "span_y": float(span[1]),
        "span_z": float(span[2]),
        "anisotropy": float(eig[0] / max(eig[-1], 1.0e-12)),
    }


def plot_structure(ax, X: Array, title: str, string_color: str = "tab:green") -> None:
    Xc = center(X)
    ax.scatter(Xc[:, 0], Xc[:, 1], Xc[:, 2], s=34, color="black")
    ax.add_collection3d(Line3DCollection(segments(Xc, STRINGS), colors=string_color, linewidths=1.0))
    ax.add_collection3d(Line3DCollection(segments(Xc, BARS), colors="tab:red", linewidths=3.0))
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_string_groups(X0: Array, scenarios: Sequence[ControlScenario]) -> str:
    Xc = center(X0)
    mids = np.array([0.5 * (Xc[i] + Xc[j]) for i, j in STRINGS], dtype=float)

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), subplot_kw={"projection": "3d"})
    axs = axs.ravel()
    for ax, scenario in zip(axs, scenarios[1:7]):
        colors = []
        for f in scenario.multiplicative_rest_factors:
            if f < 0.90 - 1.0e-8:
                colors.append("tab:blue")
            elif f > 0.90 + 1.0e-8:
                colors.append("tab:orange")
            else:
                colors.append("0.75")
        for idx, (i, j) in enumerate(STRINGS):
            ax.plot([Xc[i, 0], Xc[j, 0]], [Xc[i, 1], Xc[j, 1]], [Xc[i, 2], Xc[j, 2]], color=colors[idx], linewidth=1.5)
        for i, j in BARS:
            ax.plot([Xc[i, 0], Xc[j, 0]], [Xc[i, 1], Xc[j, 1]], [Xc[i, 2], Xc[j, 2]], color="tab:red", linewidth=3.0)
        ax.scatter(Xc[:, 0], Xc[:, 1], Xc[:, 2], s=28, color="black")
        ax.set_title(scenario.name, fontsize=10)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.suptitle("Controlled string groups: blue=contracted, orange=relaxed", y=0.98)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "tensegrity24_control_groups.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    X0 = make_initial_positions()
    scenarios = make_scenarios(X0)

    results: List[Tuple[ControlScenario, Array, Dict[str, float]]] = []
    for scenario in scenarios:
        sim = TensegritySimulator(X0)
        sim.set_string_rest_factors(scenario.multiplicative_rest_factors)
        Xeq = sim.run_to_equilibrium()
        results.append((scenario, Xeq, shape_descriptors(Xeq)))

    fig = plt.figure(figsize=(14, 9))
    for idx, (scenario, Xeq, desc) in enumerate(results):
        ax = fig.add_subplot(3, 3, idx + 1, projection="3d")
        title = (
            f"{scenario.name}\n"
            f"$s_x={desc['span_x']:.2f}$  $s_y={desc['span_y']:.2f}$\n"
            f"$s_z={desc['span_z']:.2f}$"
        )
        plot_structure(ax, Xeq, title=title)
    fig.suptitle("Possible shapes of the 6-bar / 24-string tensegrity under string rest-length control", y=0.98)
    fig.tight_layout()
    grid_path = os.path.join(OUT_DIR, "tensegrity24_shape_gallery.png")
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)

    group_path = plot_string_groups(X0, scenarios)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [sc.name for sc, _, _ in results]
    span_x = [desc["span_x"] for _, _, desc in results]
    span_y = [desc["span_y"] for _, _, desc in results]
    span_z = [desc["span_z"] for _, _, desc in results]
    anis = [desc["anisotropy"] for _, _, desc in results]

    axs[0].plot(labels, span_x, marker="o", label="span x")
    axs[0].plot(labels, span_y, marker="o", label="span y")
    axs[0].plot(labels, span_z, marker="o", label="span z")
    axs[0].set_title("Shape spans")
    axs[0].grid(alpha=0.25)
    axs[0].tick_params(axis="x", rotation=35)
    axs[0].legend(loc="best")

    axs[1].plot(labels, anis, marker="o", color="purple")
    axs[1].set_title("Anisotropy ratio")
    axs[1].grid(alpha=0.25)
    axs[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    diag_path = os.path.join(OUT_DIR, "tensegrity24_shape_descriptors.png")
    fig.savefig(diag_path, dpi=180)
    plt.close(fig)

    summary_path = os.path.join(OUT_DIR, "tensegrity24_shape_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("6-bar / 24-string tensegrity shape exploration\n")
        f.write("=============================================\n\n")
        for scenario, _, desc in results:
            f.write(f"{scenario.name}\n")
            f.write(f"  description : {scenario.description}\n")
            f.write(f"  span_x      : {desc['span_x']:.6f}\n")
            f.write(f"  span_y      : {desc['span_y']:.6f}\n")
            f.write(f"  span_z      : {desc['span_z']:.6f}\n")
            f.write(f"  anisotropy  : {desc['anisotropy']:.6f}\n\n")

    print("Shape exploration finished: 6-bar / 24-string tensegrity")
    print(f"Scenarios: {len(results)}")
    print(f"Gallery saved in: {grid_path}")
    print(f"Control groups:   {group_path}")
    print(f"Descriptors:      {diag_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
