from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util

@dataclass
class Gait:
    n_steps: int       # Number of steps to optimize
    n_phases: int      # Number of phases in each step
    spline_order: int  # Spline order for trajectory generation
    tau_k: jnp.ndarray  # Time duration of phases [s]
    h_des: float    # Desired height above ground
    eps_min: float  # Minimum distance between feet
    weights: jnp.ndarray  # Weights for the cost function
    desired_base_velocity: jnp.ndarray  # Desired base velocity in world frame
    desired_base_angular_velocity: jnp.ndarray  # Desired base angular velocity in world frame
    apex_height: float # Apex height for swing leg trajectory

@dataclass
class Terrain:
    heightmap: jnp.ndarray # Original heightmap
    grid_cell_length: float # Length of grid cells in heightmap
    mu: float # Friction coefficient
    gravity: jnp.ndarray # Gravity vector

@dataclass
class Robot:
    mass: float
    inertia: jnp.ndarray
    l_min: float # Minimum leg length
    l_max: float # Maximum leg length
    r_1: jnp.ndarray # Position of limb 1 center in base frame
    r_2: jnp.ndarray # Position of limb 2 center in base frame
    r_3: jnp.ndarray # Position of limb 3 center in base frame
    r_4: jnp.ndarray # Position of limb 4 center in base frame
    p_1_start: jnp.ndarray # Start foot position of limb 1
    p_2_start: jnp.ndarray # Start foot position of limb 2
    p_3_start: jnp.ndarray # Start foot position of limb 3
    p_4_start: jnp.ndarray # Start foot position of limb 4

@dataclass
class CurrentState: 
    p_1_meas: jnp.ndarray  # Measured foot position of limb 1
    p_2_meas: jnp.ndarray  # Measured foot position of limb 2
    p_3_meas: jnp.ndarray  # Measured foot position of limb 3
    p_4_meas: jnp.ndarray  # Measured foot position of limb 4
    initial_base_pose: jnp.ndarray  # Initial base pose (x, y, z, roll, pitch, yaw)
    initial_base_velocity: jnp.ndarray  # Initial base velocity (vx, vy, vz, wx, wy, wz)

# Register CurrentState as a pytree (all leaves are arrays)
def _cs_flatten(cs: CurrentState):
    leaves = (cs.p_1_meas, cs.p_2_meas, cs.p_3_meas, cs.p_4_meas,
              cs.initial_base_pose, cs.initial_base_velocity)
    return leaves, None
def _cs_unflatten(aux, leaves):
    p1, p2, p3, p4, pose, vel = leaves
    return CurrentState(p_1_meas=p1, p_2_meas=p2, p_3_meas=p3, p_4_meas=p4,
                        initial_base_pose=pose, initial_base_velocity=vel)
tree_util.register_pytree_node(CurrentState, _cs_flatten, _cs_unflatten)

try:
    tree_util.register_pytree_node(CurrentState, _cs_flatten, _cs_unflatten)
except Exception:
    pass