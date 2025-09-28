import numpy as np
from scipy.interpolate import CubicSpline
from .helpers import *
from .tamols_dataclasses import *
from .problem_class import TAMOLS

def _build_leg_spline(p_start: np.ndarray, p_end: np.ndarray, apex_height: float):
    line_vec = p_end - p_start
    z_axis = np.array([0.0, 0.0, 1.0])
    # Handle degeneracy (zero or vertical line)
    lv_norm = np.linalg.norm(line_vec)
    if lv_norm < 1e-9 or np.linalg.norm(np.cross(line_vec, z_axis)) < 1e-9:
        normal = z_axis
    else:
        normal = np.cross(np.cross(line_vec, z_axis), line_vec)
        normal /= np.linalg.norm(normal)
    s = apex_height * normal + 0.5 * (p_start + p_end)
    t_knots = np.array([0.0, 0.5, 1.0])  # normalized time within phase
    pts = np.vstack([p_start, s, p_end])
    return CubicSpline(t_knots, pts, axis=0)

def get_trajectory_function(results: list, problem: TAMOLS):
    """
    Builds a trajectory object that can be sampled at a uniform timestep.
    Returns a function trajectory(dt) -> (times, feet_pos, base_pos, base_rot)
        times:    (N,)
        feet_pos: (N, 4, 3)
        base_pos: (N, 3)
         base_rot: (N, 3) Euler XYZ angles
    """

    if not results:
        raise ValueError("results list is empty.")

    gait = problem.gait
    tau_k = np.asarray(gait.tau_k, dtype=float)
    step_duration = float(np.sum(tau_k))
    apex_height = float(getattr(gait, "apex_height", 0.1))

    # Initial (previous) step foot positions
    feet_prev = np.array([
        np.asarray(problem.robot.p_1_start),
        np.asarray(problem.robot.p_2_start),
        np.asarray(problem.robot.p_3_start),
        np.asarray(problem.robot.p_4_start)
    ], dtype=float)

    step_defs = []
    t_cursor = 0.0

    # Unravel or pass-through each result
    for res in results:
        if isinstance(res, dict) and all(k in res for k in ("p_1","p_2","p_3","p_4")):
            sv = res
        else:
            vec = jnp.asarray(res)
            sv = problem._unravel(vec)

        feet_next = np.array([
            np.asarray(sv["p_1"]),
            np.asarray(sv["p_2"]),
            np.asarray(sv["p_3"]),
            np.asarray(sv["p_4"])
        ], dtype=float)

        a_pos = sv["a_pos"]  # (n_phases, 3, order+1)
        a_rot = sv["a_rot"]  # (n_phases, 3, order+1)

        # Swing splines (assumes 3-phase pattern: phase 0 legs (0,3), phase 2 legs (1,2))
        splines = {
            (0, 0): _build_leg_spline(feet_prev[0], feet_next[0], apex_height),
            (0, 3): _build_leg_spline(feet_prev[3], feet_next[3], apex_height),
            (2, 1): _build_leg_spline(feet_prev[1], feet_next[1], apex_height),
            (2, 2): _build_leg_spline(feet_prev[2], feet_next[2], apex_height),
        }

        step_defs.append({
            "t_start": t_cursor,
            "t_end": t_cursor + step_duration,
            "p_start": feet_prev.copy(),
            "p_end": feet_next.copy(),
            "splines": splines,
            "a_pos": a_pos,
            "a_rot": a_rot
        })

        feet_prev = feet_next
        t_cursor += step_duration

    total_time = step_defs[-1]["t_end"] if step_defs else 0.0

    def motion_trajectory(dt: float, include_endpoint: bool = True):
        """
        Sample the full feet trajectory at uniform timestep dt.
        Returns:
            times, feet_pos, base_pos, base_rot
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if total_time == 0.0:
            return (np.array([0.0]),
                    np.zeros((1, 4, 3)),
                    np.zeros((1, 3)),
                    np.zeros((1, 3)))

        end = total_time + (1e-12 if include_endpoint else 0.0)
        times = np.arange(0.0, end, dt, dtype=float)
        # If include_endpoint and last < total_time by tolerance, append
        if include_endpoint and (total_time - times[-1]) > 1e-9:
            times = np.append(times, total_time)

        feet_samples = []
        base_pos_samples = []
        base_rot_samples = []

        for t in times:
            # Reuse logic inside _feet_at_time but also evaluate base
            if t >= total_time:
                step = step_defs[-1]
            else:
                step = next(sd for sd in step_defs if sd["t_start"] <= t < sd["t_end"])
            local_t = t - step["t_start"]
            if local_t < tau_k[0]:
                phase = 0; phase_t = local_t; phase_dur = tau_k[0]
            elif local_t < tau_k[0] + tau_k[1]:
                phase = 1; phase_t = local_t - tau_k[0]; phase_dur = tau_k[1]
            else:
                phase = 2; phase_t = local_t - tau_k[0] - tau_k[1]; phase_dur = tau_k[2]

            # Feet (reuse existing logic)
            feet = step["p_start"].copy()
            if phase > 0:
                feet[0] = step["p_end"][0]
                feet[3] = step["p_end"][3]
            s_norm = (phase_t / phase_dur) if phase_dur > 0 else 0.0
            s_norm = float(min(max(s_norm, 0.0), 1.0))
            for k in [(phase, 0), (phase, 1), (phase, 2), (phase, 3)]:
                if k in step["splines"]:
                    feet[k[1]] = step["splines"][k](s_norm)
            if phase == 2 and s_norm >= 0.999:
                feet[1] = step["p_end"][1]
                feet[2] = step["p_end"][2]

            # Base position / rotation (evaluate spline coeffs)
            # phase_t is time within phase
            pB = evaluate_spline_position(step["a_pos"][phase], phase_t)
            rB = evaluate_spline_position(step["a_rot"][phase], phase_t)

            feet_samples.append(feet)
            base_pos_samples.append(np.asarray(pB))
            base_rot_samples.append(np.asarray(rB))

        feet_pos = np.stack(feet_samples, axis=0)
        base_pos = np.stack(base_pos_samples, axis=0)
        base_rot = np.stack(base_rot_samples, axis=0)
        return times, feet_pos, base_pos, base_rot

    # Metadata
    motion_trajectory.total_time = total_time
    motion_trajectory.step_duration = step_duration
    motion_trajectory.num_steps = len(step_defs)

    return motion_trajectory
