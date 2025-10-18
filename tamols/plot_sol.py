import numpy as np
import matplotlib.pyplot as plt
from .tamols_dataclasses import *
from .problem_class import TAMOLS
from .trajectory import get_trajectory_function
from .helpers import euler_xyz_to_matrix  # if you want full frame plotting (optional)

def plot_all_iterations(sols, problem: TAMOLS, dt: float = 0.02):
    """
    Plot continuous 3D trajectories of the 4 feet plus the base trajectory & orientation.
    Orientation frames are drawn once per step (at the step start).
    """
    traj_fn = get_trajectory_function(sols, problem)
    times, feet_pos, base_pos, base_rot = traj_fn(dt)  # feet_pos: (N,4,3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    foot_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    foot_labels = ["Foot 1", "Foot 2", "Foot 3", "Foot 4"]

    # Plot feet trajectories
    for i in range(4):
        p = feet_pos[:, i, :]
        ax.plot(p[:, 0], p[:, 1], p[:, 2],
                color=foot_colors[i], linewidth=1.8, label=foot_labels[i])
        ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2],
                   color=foot_colors[i], edgecolors="k", s=55, marker="o", zorder=5)

    # ---- Base position trajectory ----
    ax.plot(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2],
            color="k", linewidth=2.2, label="Base")

    # Mark start/end of base
    ax.scatter(base_pos[0, 0], base_pos[0, 1], base_pos[0, 2],
               color="k", s=70, marker="^", edgecolors="w", zorder=6)
    ax.scatter(base_pos[-1, 0], base_pos[-1, 1], base_pos[-1, 2],
               color="k", s=70, marker="s", edgecolors="w", zorder=6)

    # ---- Orientation frames once per step ----
    step_duration = float(np.sum(np.asarray(problem.gait.tau_k, dtype=float)))
    total_time = times[-1]
    # Number of full steps (solutions length)
    n_steps = int(round(total_time / step_duration)) if step_duration > 1e-12 else 1
    frame_len = 0.08

    def closest_index(t_target: float):
        return int(np.clip(np.searchsorted(times, t_target, side="left"), 0, len(times)-1))

    for s in range(n_steps + 1):
        t_step = min(s * step_duration, total_time)
        idx = closest_index(t_step)
        p = base_pos[idx]
        phi = base_rot[idx]
        try:
            R = euler_xyz_to_matrix(phi)
            x_axis = p + frame_len * R[:, 0]
            y_axis = p + frame_len * R[:, 1]
            z_axis = p + frame_len * R[:, 2]
            ax.plot([p[0], x_axis[0]], [p[1], x_axis[1]], [p[2], x_axis[2]], color="r", linewidth=1.2)
            ax.plot([p[0], y_axis[0]], [p[1], y_axis[1]], [p[2], y_axis[2]], color="g", linewidth=1.2)
            ax.plot([p[0], z_axis[0]], [p[1], z_axis[1]], [p[2], z_axis[2]], color="b", linewidth=1.2)
        except Exception:
            yaw = phi[2]
            heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            h_end = p + frame_len * heading
            ax.plot([p[0], h_end[0]], [p[1], h_end[1]], [p[2], h_end[2]],
                    color="m", linewidth=1.2)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Foot and Base Trajectories")

    # Equal scaling
    all_pts = np.vstack([feet_pos.reshape(-1, 3), base_pos])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    spans = maxs - mins
    max_span = float(spans.max()) if spans.max() > 0 else 1.0
    center = (mins + maxs) / 2.0
    ax.set_xlim(center[0] - max_span / 2, center[0] + max_span / 2)
    ax.set_ylim(center[1] - max_span / 2, center[1] + max_span / 2)
    ax.set_zlim(center[2] - max_span / 2, center[2] + max_span / 2)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def compute_base_velocities(sols, problem: TAMOLS, dt: float = 0.02):
    """
    Sample the trajectory and return base linear velocities over time.
    Returns (times, base_vel) where base_vel has columns [vx, vy, vz].
    """
    traj_fn = get_trajectory_function(sols, problem)
    times, _, base_pos, _ = traj_fn(dt)
    base_vel = np.gradient(base_pos, dt, axis=0, edge_order=2)
    return times, base_vel

def plot_base(sols, problem: TAMOLS, dt: float = 0.02):
    """
    Plot base positional positions (x, y, z) and velocities (vx, vy, vz) over time.
    """
    traj_fn = get_trajectory_function(sols, problem)
    times, _, base_pos, _ = traj_fn(dt)
    base_vel = np.gradient(base_pos, dt, axis=0, edge_order=2)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Positions
    axs[0].plot(times, base_pos[:, 0], label="x", color="tab:blue")
    axs[0].plot(times, base_pos[:, 1], label="y", color="tab:orange")
    axs[0].plot(times, base_pos[:, 2], label="z", color="tab:green")
    axs[0].set_title("Base Position")
    axs[0].set_ylabel("Position [m]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    # Velocities
    axs[1].plot(times, base_vel[:, 0], label="vx", color="tab:blue")
    axs[1].plot(times, base_vel[:, 1], label="vy", color="tab:orange")
    axs[1].plot(times, base_vel[:, 2], label="vz", color="tab:green")
    axs[1].set_title("Base Positional Velocity")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    plt.tight_layout()
    plt.show()