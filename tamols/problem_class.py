from .tamols_dataclasses import *
import numpy as np
import jax.numpy as jnp
from .cost_parts import *
from jax.flatten_util import ravel_pytree
from jax import vmap, jit, value_and_grad, jacfwd, jacrev
import cyipopt
from .helpers import (
    evaluate_spline_position,
    evaluate_spline_velocity,
    evaluate_spline_acceleration,
    euler_xyz_to_matrix,
    robust_norm,
    scalar_triple_product,
    angular_momentum_dot_world_from_euler_xyz
)
import time

class TAMOLS():
    def __init__(self, gait: Gait, terrain: Terrain, robot: Robot,
                 x_lb: np.ndarray = None, x_ub: np.ndarray = None):

        self.gait = gait
        self.terrain = terrain
        self.robot = robot
        self.current_state = CurrentState(
            p_1_meas=robot.p_1_start,
            p_2_meas=robot.p_2_start,
            p_3_meas=robot.p_3_start,
            p_4_meas=robot.p_4_start,
            current_base_pose=robot.initial_base_pose,
            current_base_velocity=robot.initial_base_velocity,
            use_virtual_floor=jnp.array(False),
        )

        # --- Assertions ---
        assert isinstance(self.gait.tau_k, (list, tuple, np.ndarray, jnp.ndarray)), \
            "gait.tau_k must be array-like"
        tau_k_arr = np.asarray(self.gait.tau_k)
        assert tau_k_arr.ndim == 1, \
            f"gait.tau_k must be 1D of length n_phases; got ndim={tau_k_arr.ndim}"
        assert tau_k_arr.shape[0] == self.gait.n_phases, \
            f"gait.tau_k length {tau_k_arr.shape[0]} != n_phases {self.gait.n_phases}"

        for name in ("contact_schedule", "at_des_position"):
            if not hasattr(self.gait, name):
                raise AttributeError(f"Gait missing required attribute '{name}'")
            arr = np.asarray(getattr(self.gait, name))
            assert arr.shape == (self.gait.n_phases, 4), \
                f"{name} shape {arr.shape} != ({self.gait.n_phases}, 4)"

        # Precompute phases with exactly two contacts
        cs_np = np.asarray(self.gait.contact_schedule, dtype=int)
        two_contact_phase_idx = np.where(cs_np.sum(axis=1) == 2)[0]
        two_contact_pairs = []
        for ph in two_contact_phase_idx:
            legs = np.where(cs_np[ph] == 1)[0]
            if legs.size != 2:
                continue  # safety
            two_contact_pairs.append(legs)
        self._two_contact_phases = jnp.asarray(two_contact_phase_idx, dtype=jnp.int32)
        self._two_contact_pairs  = jnp.asarray(two_contact_pairs, dtype=jnp.int32) if two_contact_pairs else jnp.zeros((0,2), dtype=jnp.int32)

        # Precompute (phase, leg_i, leg_j) for phases with >=3 contacts (for dyn_phase_ineq)
        multi_pairs = []
        multi_contact_phase_idx = np.where(cs_np.sum(axis=1) >= 3)[0]
        for ph in multi_contact_phase_idx:
            legs = np.where(cs_np[ph] == 1)[0]
            for a in range(len(legs)):
                for b in range(a + 1, len(legs)):
                    multi_pairs.append([ph, legs[a], legs[b]])
        self._multi_contact_phase_pairs = (
            jnp.asarray(multi_pairs, dtype=jnp.int32)
            if multi_pairs else jnp.zeros((0, 3), dtype=jnp.int32)
        )

        # Precompute heightmap gradients
        self.h_s1 = jnp.array(get_hs1(terrain.heightmap))
        self.h_s2 = jnp.array(get_hs2(terrain.heightmap))
        gxh, gyh = compute_heightmap_gradients(self.terrain.heightmap, self.terrain.grid_cell_length)
        gx1, gy1 = compute_heightmap_gradients(self.h_s1, self.terrain.grid_cell_length)
        self.grad_h_x = jnp.array(gxh)
        self.grad_h_y = jnp.array(gyh)
        self.grad_h_s1_x = jnp.array(gx1)
        self.grad_h_s1_y = jnp.array(gy1)

        # Virtual-floor gradients (for warm start)
        gx2, gy2 = compute_heightmap_gradients(self.h_s2, self.terrain.grid_cell_length)
        h_s1_vf = jnp.array(get_hs1(self.h_s2))
        gx1_vf, gy1_vf = compute_heightmap_gradients(h_s1_vf, self.terrain.grid_cell_length)
        self.grad_h2_x = jnp.array(gx2)
        self.grad_h2_y = jnp.array(gy2)
        self.grad_h_s1_vf_x = jnp.array(gx1_vf)
        self.grad_h_s1_vf_y = jnp.array(gy1_vf)

        # Decision vector and bounds
        x0 = build_initial_x0(gait, self.current_state)
        x0 = np.asarray(x0, dtype=float)
        self.n = int(x0.size)
        self.x0 = x0
        self.lb = np.full(self.n, -np.inf) if x_lb is None else np.asarray(x_lb, dtype=float)
        self.ub = np.full(self.n,  np.inf) if x_ub is None else np.asarray(x_ub, dtype=float)

        # Precompute limb centers and previous footholds
        self._limb_centers = jnp.stack([robot.r_1, robot.r_2, robot.r_3, robot.r_4])

        self.T_k = gait.tau_k / 6.0  # nominal timestep per phase (s)
        # Uniform per‑phase timestep grid (use a fixed samples-per-phase)
        N = 6
        t_idx = jnp.arange(1, N+1, dtype=jnp.float32)  # 1..6
        # Broadcast multiply: (n_phases, 1) * (1, 6) -> (n_phases, 6)
        self._timesteps = self.T_k[:, None] * t_idx[None, :]

        # Pair indices for leg collision cost
        pairs = np.array([(i, j) for i in range(4) for j in range(i + 1, 4)], dtype=np.int32)
        self._pairs = jnp.asarray(pairs)

        # Build a template to define x’s structure and get unravel_fn
        tpl = {
            "a_pos": jnp.zeros((self.gait.n_phases, 3, self.gait.spline_order + 1)),
            "a_rot": jnp.zeros((self.gait.n_phases, 3, self.gait.spline_order + 1)),
            "p_1": robot.p_1_start,
            "p_2": robot.p_2_start,
            "p_3": robot.p_3_start,
            "p_4": robot.p_4_start,
            "slack_var": jnp.ones(self.gait.n_phases)
        }
        _, self._unravel = ravel_pytree(tpl)

        # Param-aware (x, current_state) functions; no closure over current_state
        self._f_valgrad = jit(value_and_grad(lambda x, cs: self.compute_objective(x, cs)))
        self._c_eq      = jit(lambda x, cs: self.compute_eq_constraints(x, cs))
        self._c_ineq    = jit(lambda x, cs: self.compute_ineq_constraints(x, cs))
        self._c_all     = jit(lambda x, cs: jnp.concatenate([self._c_eq(x, cs), self._c_ineq(x, cs)]))
        self._J_all     = jit(jacfwd(lambda x, cs: self._c_all(x, cs), argnums=0))
        self._hess_L    = jit(jacfwd(jacrev(lambda x, lam, rho, cs:
                                             rho * self.compute_objective(x, cs)
                                             + jnp.dot(lam, self._c_all(x, cs)),
                                           argnums=0),
                                     argnums=0))

        x0_j = jnp.asarray(x0)
        c_eq0 = self._c_eq(x0_j, self.current_state)
        c_in0 = self._c_ineq(x0_j, self.current_state)
        m_eq   = int(np.asarray(c_eq0).size)
        m_ineq = int(np.asarray(c_in0).size)
        self.m = m_eq + m_ineq
        self.cl = np.concatenate([
            np.zeros(m_eq, dtype=float),
            np.zeros(m_ineq, dtype=float)
        ])
        self.cu = np.concatenate([
            np.zeros(m_eq, dtype=float),
            np.full(m_ineq, np.inf, dtype=float)
        ])

        # Dense Jacobian structure (row-major)
        if self.m > 0:
            rows = np.repeat(np.arange(self.m), self.n).astype(np.int64)
            cols = np.tile(np.arange(self.n), self.m).astype(np.int64)
        else:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
        self._jac_rows, self._jac_cols = rows, cols

        # Lower-triangular sparsity pattern (Ipopt expects symmetric lower triangle)
        tri = np.tril_indices(self.n)
        self._H_rows = tri[0].astype(np.int64)
        self._H_cols = tri[1].astype(np.int64)

     # -------- helpers used inside the objective --------
    def _cost_at_t(self, a_pos, a_rot, t, limb_center, T_k_phase):
        w = self.gait.weights
        p_B = evaluate_spline_position(a_pos, t)
        phi_B = evaluate_spline_position(a_rot, t)
        R_B = euler_xyz_to_matrix(phi_B)
        p_B_dot = evaluate_spline_velocity(a_pos, t)
        phi_B_dot = evaluate_spline_velocity(a_rot, t)
        phi_B_dotdot = evaluate_spline_acceleration(a_rot, t)

        c4 = w[3] * T_k_phase * base_pose_alignment_cost(
            p_B, self.gait.h_des, limb_center, R_B,
            self.h_s2, self.terrain.grid_cell_length
        )
        c7 = w[6] * T_k_phase * tracking_cost(
            p_B_dot, phi_B, phi_B_dot, self.gait.desired_base_velocity,
            self.gait.desired_base_angular_velocity, self.robot.mass,
            self.robot.inertia
        )
        c8 = w[7] * T_k_phase * smoothness_cost(phi_B, phi_B_dot, phi_B_dotdot, self.robot.inertia)
        return c4 + c7 + c8

    def _cost_of_phase(self, a_pos_phase, a_rot_phase, timesteps_phase, limb_center, T_k_phase):
        return jnp.sum(vmap(lambda t: self._cost_at_t(a_pos_phase, a_rot_phase, t, limb_center, T_k_phase))(timesteps_phase))

    # ----------------- x-only objective -----------------
    def compute_objective(self, x: jnp.ndarray, cs: CurrentState) -> jnp.ndarray:
        sv = self._unravel(x)

        # Coerce measured feet to JAX arrays once (fixed shapes/dtypes)
        m1 = jnp.asarray(cs.p_1_meas)
        m2 = jnp.asarray(cs.p_2_meas)
        m3 = jnp.asarray(cs.p_3_meas)
        m4 = jnp.asarray(cs.p_4_meas)

        # Decision variables
        a_pos = sv["a_pos"]  # (n_phases, 3, k+1)
        a_rot = sv["a_rot"]  # (n_phases, 3, k+1)
        feet = jnp.stack([sv["p_1"], sv["p_2"], sv["p_3"], sv["p_4"]])  # (4,3)
        feet_prev = jnp.stack([m1, m2, m3, m4])  # (4,3)
        w = self.gait.weights

        # Select foothold heightmap by cs.use_virtual_floor (no re-JIT needed)
        hmap_real = jnp.asarray(self.terrain.heightmap)
        hmap = jnp.where(cs.use_virtual_floor, self.h_s2, hmap_real)

        # Select gradient fields based on use_virtual_floor
        grad_h_x   = jnp.where(cs.use_virtual_floor, self.grad_h2_x,        self.grad_h_x)
        grad_h_y   = jnp.where(cs.use_virtual_floor, self.grad_h2_y,        self.grad_h_y)
        grad_h1_x  = jnp.where(cs.use_virtual_floor, self.grad_h_s1_vf_x,   self.grad_h_s1_x)
        grad_h1_y  = jnp.where(cs.use_virtual_floor, self.grad_h_s1_vf_y,   self.grad_h_s1_y)

        # Per-limb "static" foothold costs
        def limb_static_cost(foot, prev_foot):
            c1 = w[0] * foothold_on_ground_cost(foot, hmap, self.terrain.grid_cell_length)
            c5 = w[4] * edge_avoidance_cost(grad_h_x, grad_h_y,
                                            grad_h1_x, grad_h1_y,
                                            foot, self.terrain.grid_cell_length)
            c6 = w[5] * previous_solution_cost(foot, prev_foot)
            return c1 + c5 + c6

        static_costs = vmap(limb_static_cost)(feet, feet_prev)

        # Per-limb, per-phase spline costs
        phase_idx = jnp.arange(self.gait.n_phases)

        def limb_phase_sum(a_pos_all, a_rot_all, limb_center):
            return jnp.sum(vmap(lambda n: self._cost_of_phase(a_pos_all[n], a_rot_all[n],
                                                             self._timesteps[n], limb_center, self.T_k[n]))(phase_idx))
        
        spline_costs = vmap(lambda lc: limb_phase_sum(a_pos, a_rot, lc))(self._limb_centers)

        # Select feet per phase based on at_des_position (1 -> current decision foot, 0 -> previous/measured)
        at_des = jnp.asarray(self.gait.at_des_position, dtype=feet.dtype)  # (n_phases,4)
        mask = at_des[..., None]                       # (n_phases,4,1)
        feet_cur = feet[None, :, :]                    # (1,4,3)
        feet_prev_all = feet_prev[None, :, :]          # (1,4,3)
        feet_sel = mask * feet_cur + (1.0 - mask) * feet_prev_all  # (n_phases,4,3)

        # Nominal kinematics at mid‑phase for stance legs only
        contact = jnp.asarray(self.gait.contact_schedule, dtype=feet.dtype)  # (n_phases,4)

        def kin_mid_phase(a_pos_n, a_rot_n, timesteps_n, feet_n, contact_n):
            t_mid = 0.5 * timesteps_n[-1]  # tau_k/2
            p_B_mid = evaluate_spline_position(a_pos_n, t_mid)
            phi_mid = evaluate_spline_position(a_rot_n, t_mid)
            R_mid = euler_xyz_to_matrix(phi_mid)
            per_leg = vmap(lambda f, lc: nominal_kinematics_cost(
                p_B_mid, f, self.gait.h_des, lc, R_mid
            ))(feet_n, self._limb_centers)  # (4,)
            return w[2] * jnp.sum(per_leg * contact_n)

        kin_mid_total = jnp.sum(vmap(kin_mid_phase)(
            a_pos, a_rot, self._timesteps, feet_sel, contact
        ))

        def phase_pair_cost(feet_phase):
            return jnp.sum(vmap(
                lambda pair: w[1] * leg_collision_avoidance_cost(
                    feet_phase[pair[0]], feet_phase[pair[1]], self.gait.eps_min)
            )(self._pairs))

        pair_costs_all = vmap(phase_pair_cost)(feet_sel)  # (n_phases,)
        pair_cost_total = jnp.sum(pair_costs_all)

        return (jnp.sum(static_costs)
                + jnp.sum(spline_costs)
                + kin_mid_total
                + pair_cost_total
                + slack_variables_cost(sv["slack_var"]))
    
        # ----------------- x-only equality constraints -----------------
    def compute_eq_constraints(self, x: jnp.ndarray, cs: CurrentState) -> jnp.ndarray:
        sv = self._unravel(x)

        # Initial conditions (phase 0, t=0)
        p0_err   = evaluate_spline_position(sv["a_pos"][0], 0.0) - cs.current_base_pose[:3]
        r0_err   = evaluate_spline_position(sv["a_rot"][0], 0.0) - cs.current_base_pose[3:]
        v0_err   = evaluate_spline_velocity(sv["a_pos"][0], 0.0) - cs.current_base_velocity[:3]
        w0_err   = evaluate_spline_velocity(sv["a_rot"][0], 0.0) - cs.current_base_velocity[3:]

        # Junction continuity between consecutive phases (pos and vel for pos/rot)
        def junction(i):
            ti = self.gait.tau_k[i]
            return jnp.concatenate([
                evaluate_spline_position(sv["a_pos"][i], ti) - evaluate_spline_position(sv["a_pos"][i+1], 0.0),
                evaluate_spline_position(sv["a_rot"][i], ti) - evaluate_spline_position(sv["a_rot"][i+1], 0.0),
                evaluate_spline_velocity(sv["a_pos"][i], ti) - evaluate_spline_velocity(sv["a_pos"][i+1], 0.0),
                evaluate_spline_velocity(sv["a_rot"][i], ti) - evaluate_spline_velocity(sv["a_rot"][i+1], 0.0),
            ])

        if self.gait.n_phases > 1:
            jcs = vmap(junction)(jnp.arange(self.gait.n_phases - 1)).reshape(-1)
            eq = jnp.concatenate([p0_err, r0_err, v0_err, w0_err, jcs])
        else:
            eq = jnp.concatenate([p0_err, r0_err, v0_err, w0_err])

        return eq
    
    # ----------------- x-only inequality constraints -----------------
    def compute_ineq_constraints(self, x: jnp.ndarray, cs: CurrentState) -> jnp.ndarray:
        sv = self._unravel(x)
        g = self.terrain.gravity
        mass = self.robot.mass
        inertia = self.robot.inertia
        mu = self.terrain.mu
        e_z = jnp.array([0.0, 0.0, 1.0])
        l_min, l_max = self.robot.l_min, self.robot.l_max

        # Coerce measured feet once
        m1 = jnp.asarray(cs.p_1_meas)
        m2 = jnp.asarray(cs.p_2_meas)
        m3 = jnp.asarray(cs.p_3_meas)
        m4 = jnp.asarray(cs.p_4_meas)

        feet = jnp.stack([sv["p_1"], sv["p_2"], sv["p_3"], sv["p_4"]])  # (4,3)
        feet_prev = jnp.stack([m1, m2, m3, m4])  # (4,3)
        limb_centers = self._limb_centers

        # For each phase and leg: if at_des_position==1 use current decision foot,
        # otherwise use previous (measured) foot position.
        at_des = jnp.asarray(self.gait.at_des_position, dtype=feet.dtype)  # (n_phases,4)
        mask = at_des[..., None]  # (n_phases,4,1)
        feet_cur = feet[None, :, :]          # (1,4,3)
        feet_prev_all = feet_prev[None, :, :]  # (1,4,3)
        feet_sel = mask * feet_cur + (1.0 - mask) * feet_prev_all  # (n_phases,4,3)

        # Simple vertical-accel constraint: a_z(t) - g_z >= 0
        def az_minus_g_phase(phase):
            a_pos = sv["a_pos"][phase]
            ts = self._timesteps[phase]
            def at_t(t):
                acc = evaluate_spline_acceleration(a_pos, t)  # p¨_B
                return acc[2] - g[2]
            return vmap(at_t)(ts)
        az_g = jnp.concatenate([az_minus_g_phase(p) for p in range(self.gait.n_phases)])

        # Friction cone over all phases/timesteps: mu*(-a_Bz) - ||a_Bxy|| >= 0
        def friction_phase(phase):
            a_pos = sv["a_pos"][phase]
            ts = self._timesteps[phase]
            def at_t(t):
                p_B_dd = evaluate_spline_acceleration(a_pos, t)
                a_B = g - p_B_dd
                return mu * (-a_B[2]) - robust_norm(a_B[:2])
            return vmap(at_t)(ts)
        fric = jnp.concatenate([friction_phase(p) for p in range(self.gait.n_phases)])

        # Dynamic constraints (two-contact phases):
        g = self.terrain.gravity
        mass = self.robot.mass
        inertia = self.robot.inertia

        def dyn_phase_eq(phase, p_i, p_j):
            a_pos, a_rot = sv["a_pos"][phase], sv["a_rot"][phase]
            ts = jnp.array([self._timesteps[phase][-1]])
            p_ij = p_j - p_i
            eps = sv["slack_var"][phase]

            def at_t(t):
                # Base kinematics
                p_B    = evaluate_spline_position(a_pos, t)
                p_B_dd = evaluate_spline_acceleration(a_pos, t)
                a_B    = g - p_B_dd  # base acceleration in world

                # Orientation & angular momentum rate
                phi_B  = evaluate_spline_position(a_rot, t)
                phi_d  = evaluate_spline_velocity(a_rot, t)
                phi_dd = evaluate_spline_acceleration(a_rot, t)
                L_B_dot = angular_momentum_dot_world_from_euler_xyz(phi_B, phi_d, phi_dd, inertia)

                # Original dynamic line constraint (intended ~ 0)
                con_dyn = jnp.dot(p_ij, L_B_dot) - mass * scalar_triple_product(p_ij, p_B - p_i, a_B)

                # Inlined former extra_phase (moment) constraint
                # M_i = ( (p_B - p_i) x a_B ) - L_B_dot / mass
                M_i = jnp.cross((p_B - p_i), a_B) - L_B_dot / mass
                moment_val = scalar_triple_product(e_z, p_ij, M_i)  # want >= -eps

                # Pack: [con_dyn + eps >= 0, -con_dyn + eps >= 0, moment_val + eps >= 0]
                return jnp.array([con_dyn + eps, -con_dyn + eps, moment_val + eps])

            return vmap(at_t)(ts)

        dyn_eq_list = []
        for k in range(int(self._two_contact_phases.shape[0])):
            ph = self._two_contact_phases[k].astype(int)
            legs = self._two_contact_pairs[k]
            i, j = legs[0].astype(int), legs[1].astype(int)
            p_i = feet_sel[ph, i]
            p_j = feet_sel[ph, j]
            dyn_eq_list.append(dyn_phase_eq(ph, p_i, p_j).reshape(-1))
        dyn_eq = jnp.concatenate(dyn_eq_list) if dyn_eq_list else jnp.zeros((0,), dtype=feet.dtype)

        def dyn_phase_ineq(phase, p_i, p_j):
            a_pos, a_rot = sv["a_pos"][phase], sv["a_rot"][phase]
            ts = jnp.array([self._timesteps[phase][-1]])
            p_ij = p_j - p_i
            eps = sv["slack_var"][phase]

            def at_t(t):
                p_B    = evaluate_spline_position(a_pos, t)
                p_B_dd = evaluate_spline_acceleration(a_pos, t)
                a_B = g - p_B_dd

                phi_B  = evaluate_spline_position(a_rot, t)
                phi_d = evaluate_spline_velocity(a_rot, t)
                phi_dd = evaluate_spline_acceleration(a_rot, t)
                L_B_dot = angular_momentum_dot_world_from_euler_xyz(phi_B, phi_d, phi_dd, inertia)

                return jnp.array(jnp.dot(p_ij, L_B_dot) - mass * scalar_triple_product(p_ij, p_B - p_i, a_B) + eps)

            return vmap(at_t)(ts)
        
        dyn_ineq_list = []
        # Apply dyn_phase_ineq ONLY to phases with >=3 contacts, for every contact leg pair
        for idx in range(int(self._multi_contact_phase_pairs.shape[0])):
            ph = self._multi_contact_phase_pairs[idx, 0].astype(int)    
            i  = self._multi_contact_phase_pairs[idx, 1].astype(int)
            j  = self._multi_contact_phase_pairs[idx, 2].astype(int)
            p_i = feet_sel[ph, i]
            p_j = feet_sel[ph, j]
            dyn_ineq_list.append(dyn_phase_ineq(ph, p_i, p_j).reshape(-1))
        dyn_ineq = (jnp.concatenate(dyn_ineq_list)
                    if dyn_ineq_list else jnp.zeros((0,), dtype=feet.dtype))

        # Convex GIAC (per phase, using at_des-based selected feet)
        def giac_block(p1, p2, p3, p4):
            p_12, p_13, p_14 = p2 - p1, p3 - p1, p4 - p1
            p_21, p_23, p_24 = p1 - p2, p3 - p2, p4 - p2
            # Enforce consistent winding: orient_sign=+1 for CCW, -1 for CW
            return  jnp.array([
                jnp.dot(jnp.cross(p_13, p_12), e_z),
                jnp.dot(jnp.cross(p_14, p_13), e_z),
                jnp.dot(jnp.cross(p_24, p_23), e_z),
                jnp.dot(jnp.cross(p_21, p_24), e_z),
            ])

        giac_per_phase = vmap(lambda f: giac_block(f[0], f[1], f[3], f[2]))(feet_sel)  # (n_phases,4)
        giac = giac_per_phase.reshape(-1)

        # Kinematic leg-length bounds per phase/timestep/leg
        def kin_phase(phase):
            a_pos = sv["a_pos"][phase]
            a_rot = sv["a_rot"][phase]
            ts = self._timesteps[phase]
            def at_t(t):
                p_B  = evaluate_spline_position(a_pos, t)
                phi  = evaluate_spline_position(a_rot, t)
                R_B  = euler_xyz_to_matrix(phi)
                def per_leg(foot, limb_center):
                    leg_vec = p_B + R_B @ limb_center - foot
                    ll2 = robust_norm(leg_vec)**2
                    return jnp.array([ll2 - l_min**2, l_max**2 - ll2])
                return vmap(per_leg)(feet, limb_centers).reshape(-1)
            return vmap(at_t)(ts).reshape(-1)
        kin = jnp.concatenate([kin_phase(p) for p in range(self.gait.n_phases)])

        # Slack vars >= 0
        slack_var_constr = sv["slack_var"]  # (n_phases,)

        return jnp.concatenate([fric, kin, az_g, slack_var_constr, giac, dyn_ineq])
    
    # ============== Ipopt callbacks ==============
    def objective(self, x):
        v, _ = self._f_valgrad(jnp.asarray(x), self.current_state)
        return float(v)

    def gradient(self, x):
        _, g = self._f_valgrad(jnp.asarray(x), self.current_state)
        return np.asarray(g, dtype=float)

    def constraints(self, x):
        c = self._c_all(jnp.asarray(x), self.current_state)
        return np.asarray(c, dtype=float)

    def jacobian(self, x):
        if self.m == 0:
            return np.array([], dtype=float)
        J = self._J_all(jnp.asarray(x), self.current_state)
        return np.asarray(J, dtype=float).ravel(order="C")

    def jacobianstructure(self):
        if self.m == 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return (self._jac_rows, self._jac_cols)

    def hessianstructure(self):
        # Lower-triangular structure
        return (self._H_rows, self._H_cols)

    def hessian(self, x, lagrange, obj_factor):
        H = self._hess_L(jnp.asarray(x),
                        jnp.asarray(lagrange),
                        jnp.asarray(obj_factor),
                        self.current_state)
        H = np.asarray(H, dtype=float)
        return H[(self._H_rows, self._H_cols)]

    # Solve helper
    def run_single_optimization(self, options: dict | None = None):
        x0 = self.x0
        nlp = cyipopt.Problem(
            n=self.n, m=self.m, problem_obj=self,
            lb=self.lb, ub=self.ub, cl=self.cl, cu=self.cu
        )
        ipopt_opts = {
            "hessian_approximation": "exact",
            "print_level": 0,      # silence IPOPT
            "sb": "yes",           # (small banner) further reduce output
            "max_iter": 300,
            "tol": 1e-8,
            "acceptable_tol": 1e-4,
            "print_timing_statistics": "no"
        }
        if options:
            ipopt_opts.update(options)
        for k, v in ipopt_opts.items():
            nlp.add_option(k, v)
        x_sol, info = nlp.solve(np.asarray(x0, dtype=float))
        return x_sol, info

    def run_repeated_optimizations(self, warm_start: bool = True, options: dict | None = None):
        n_steps = int(self.gait.n_steps)
        sols, infos = [], []
        x_curr = np.asarray(self.x0, dtype=float)
        print("Setting up optimization...")

        t0_total = time.perf_counter()
        opt_times = []

        for k in range(n_steps):
            self.x0 = x_curr
            step_opt_time = 0.0

            if warm_start:
                # Virtual-floor warm start (no re-JIT needed)
                self.current_state.use_virtual_floor = jnp.array(False)  # set True if you want warm-start on h_s2
                t_ws0 = time.perf_counter()
                x_ws, info_ws = self.run_single_optimization(options)
                step_opt_time += time.perf_counter() - t_ws0
                self.x0 = np.asarray(x_ws, dtype=float)

            # Main solve on real terrain
            self.current_state.use_virtual_floor = jnp.array(False)
            t_main0 = time.perf_counter()
            x_sol, info = self.run_single_optimization(options)
            step_opt_time += time.perf_counter() - t_main0
            opt_times.append(step_opt_time)

            sols.append(x_sol); infos.append(info)
            final_obj = self.objective(x_sol)
            print(f"Optimization {k+1}/{n_steps}: objective = {final_obj:.6f}")

            sv_sol = self._unravel(jnp.asarray(x_sol))
            next_state = update_state_from_solution(self, sv_sol)
            self.current_state = next_state  # just update; no re-JIT needed
            x_curr = np.asarray(build_initial_x0(self.gait, next_state), dtype=float)

        # Exclude the very first timing entry from totals/averaging
        timed = opt_times[1:] if len(opt_times) > 1 else []
        total_opt = float(sum(timed))
        steps_counted = len(timed)
        avg_opt = (total_opt / steps_counted) if steps_counted > 0 else 0.0
        wall = time.perf_counter() - t0_total

        print("Note: excluded the first optimization time from totals/average.")
        print(f"Total optimization time: {total_opt:.3f} s (avg per step: {avg_opt:.3f} s over {steps_counted} steps)")
        print(f"Total wall-clock time (incl. setup): {wall:.3f} s")

        return sols, infos
 
def update_state_from_solution(problem: TAMOLS, sv_sol):
    # End of last phase
    p_last = int(problem.gait.n_phases) - 1
    t_end = float(problem._timesteps[p_last, -1])
    a_pos_last = sv_sol["a_pos"][p_last]
    a_rot_last = sv_sol["a_rot"][p_last]
    p_end = evaluate_spline_position(a_pos_last, t_end)
    r_end = evaluate_spline_position(a_rot_last, t_end)
    v_end = evaluate_spline_velocity(a_pos_last, t_end)
    w_end = evaluate_spline_velocity(a_rot_last, t_end)

    p1m, p2m, p3m, p4m = sv_sol["p_1"], sv_sol["p_2"], sv_sol["p_3"], sv_sol["p_4"]

    return CurrentState(
        p_1_meas=p1m,
        p_2_meas=p2m,
        p_3_meas=p3m,
        p_4_meas=p4m, 
        current_base_pose=jnp.concatenate([p_end, r_end]),
        current_base_velocity=jnp.concatenate([v_end, w_end]),
        use_virtual_floor=jnp.array(False),
    )


def build_initial_x0(gait: Gait, state: CurrentState) -> np.ndarray:
    a_pos = jnp.zeros((gait.n_phases, 3, gait.spline_order + 1))
    a_rot = jnp.zeros((gait.n_phases, 3, gait.spline_order + 1))
    # Seed constant terms with initial base pose/orientation for every phase
    a_pos = a_pos.at[:, :, 0].set(state.current_base_pose[:3])
    a_rot = a_rot.at[:, :, 0].set(state.current_base_pose[3:])
    a_pos = a_pos.at[:, :, 1].set(state.current_base_velocity[:3])
    a_rot = a_rot.at[:, :, 1].set(state.current_base_velocity[3:])

    tpl = {
        "a_pos": a_pos,
        "a_rot": a_rot,
        "p_1": state.p_1_meas,
        "p_2": state.p_2_meas,
        "p_3": state.p_3_meas,
        "p_4": state.p_4_meas,
        "slack_var": jnp.zeros(gait.n_phases),
    }
    x0, _ = ravel_pytree(tpl)
    return np.asarray(x0, dtype=float)