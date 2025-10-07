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
    n_phases = int(tau_k.shape[0])
    phase_ends = np.cumsum(tau_k)  # cumulative phase end times within a step
    step_duration = float(phase_ends[-1])
    apex_height = float(getattr(gait, "apex_height", 0.1))
    # Feet scheduled to land at desired in each phase (n_phases,4) with 0/1 flags
    at_des = np.asarray(getattr(gait, "at_des_position", np.zeros((n_phases, 4))), dtype=int)

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

        # Swing splines for any (phase, leg) where at_des[phase, leg] == 1
        splines = {}
        for ph in range(n_phases):
            for leg in range(4):
                if at_des[ph, leg] == 1:
                    splines[(ph, leg)] = _build_leg_spline(feet_prev[leg], feet_next[leg], apex_height)

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

            # Generic phase mapping for any number of phases using cumulative sums
            idx = int(np.searchsorted(phase_ends, local_t, side='right'))
            phase = min(idx, n_phases - 1)
            phase_start = 0.0 if phase == 0 else float(phase_ends[phase - 1])
            phase_dur = float(tau_k[phase])
            phase_t = float(local_t - phase_start)

            # Legs that already landed in any earlier phase of THIS step stay at p_end
            if phase > 0:
                landed_before = np.any(at_des[:phase, :] == 1, axis=0)  # (4,)
                for leg in range(4):
                    if landed_before[leg]:
                        feet[leg] = step["p_end"][leg]
            else:
                landed_before = np.zeros(4, dtype=bool)

            # Feet (reuse existing logic)
            feet = step["p_start"].copy()
            # Legs that landed in earlier phases stay at p_end
            if phase > 0:
                for leg in range(4):
                    if np.any(at_des[:phase, leg] == 1):
                        feet[leg] = step["p_end"][leg]
            s_norm = (phase_t / phase_dur) if phase_dur > 0 else 0.0
            s_norm = float(min(max(s_norm, 0.0), 1.0))
            for leg in range(4):
                key = (phase, leg)
                if key in step["splines"] and not landed_before[leg]:
                    feet[leg] = step["splines"][key](s_norm)
                    # Snap to end at the tail of the phase to avoid tiny drift
                    if s_norm >= 0.999:
                        feet[leg] = step["p_end"][leg]

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
