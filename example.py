from problem_class import TAMOLS, build_initial_x0
from tamols_dataclasses import *
import jax.numpy as jnp
from helpers import transform_inertia
from manual_heightmaps import *
from map_processing import *
from plot_sol import plot_all_iterations
from trajectory import get_trajectory_function

gait = Gait(
    n_steps=6,
    n_phases=3,
    spline_order=5,
    tau_k=jnp.array([1.0, 0.4, 1.0])/2.0,
    h_des=0.445,
    eps_min=0.2,
    weights = jnp.array([10.0e4, 0.001, 7.0, 100.0, 3.0, 0.01, 2.0, 0.001]),
    desired_base_velocity = jnp.array([0.1, 0.0, 0.0]),
    desired_base_angular_velocity = jnp.array([0.0, 0.0, 0.0]),
    apex_height=0.1,
)

# h = jnp.array(get_flat_heightmap(a=300, b=300, height=0.0))  # Flat heightmap for testing
h = jnp.array(get_rough_terrain_heightmap(a=350, b=350, sigma=0.05, platform_height=0.0, platform_size=5, smooth_sigma=3, seed=42)) # Heightmap with platforms
# h = jnp.array(get_stairs_heightmap(a=300, b=300, start_col=200, step_depth=100, step_height=0.1))  # Heightmap with stairs

grid_cell_length = 0.01  # Length of grid cells in heightmap

h_s1 = jnp.array(get_hs1(h))  # Smoothed heightmap
h_s2 = jnp.array(get_hs2(h))  # Virtual floor heightmap

terrain = Terrain(
    heightmap = h,
    h_s1 = h_s1,
    h_s2 = h_s2,
    grid_cell_length = grid_cell_length,
    grad_h_x = jnp.array(compute_heightmap_gradients(h, grid_cell_length)[0]),
    grad_h_y = jnp.array(compute_heightmap_gradients(h, grid_cell_length)[1]),
    grad_h_s1_x = jnp.array(compute_heightmap_gradients(h_s1, grid_cell_length)[0]),
    grad_h_s1_y = jnp.array(compute_heightmap_gradients(h_s1, grid_cell_length)[1]),
    mu = 0.6,
    gravity = jnp.array([0.0, 0.0, -9.81]),
)

I_A = jnp.diag(jnp.array([0.107027, 0.0980771, 0.0244531]))  # Inertia matrix of the robot base (diagonal)
r = jnp.array([0.021112, 0, -0.005366])
q = jnp.array([-0.000543471, 0.713435, -0.00173769, 0.700719])
mass = 6.921

robot = Robot(
    mass=mass,
    inertia=transform_inertia(I_A, mass, r, q),
    l_min=0.1,
    l_max=0.5,
    r_1=jnp.array([0.1934,0.0465+0.0955,0.0]),
    r_2=jnp.array([0.1934,-0.0465-0.0955,0.0]),
    r_3=jnp.array([-0.1934,0.0465+0.0955,0.0]),
    r_4=jnp.array([-0.1934,-0.0465-0.0955,0.0]),
    p_1_start=jnp.array([0.1934,0.0465+0.0955,0.0]),
    p_2_start=jnp.array([0.1934,-0.0465-0.0955,0.0]),
    p_3_start=jnp.array([-0.1934,0.0465+0.0955,0.0]),
    p_4_start=jnp.array([-0.1934,-0.0465-0.0955,0.0]),
)

currentstate = CurrentState(
    p_1_meas=robot.p_1_start,
    p_2_meas=robot.p_2_start,
    p_3_meas=robot.p_3_start,
    p_4_meas=robot.p_4_start,
    initial_base_pose=jnp.array([0.0, 0.0, 0.445, 0.0, 0.0, 0.0]),
    initial_base_velocity=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

x0 = build_initial_x0(gait, currentstate)

problem = TAMOLS(x0, gait, terrain, robot, currentstate)

sols, infos = problem.run_repeated_optimizations({
            "max_iter": 300,
            "acceptable_tol": 1e-3
        })

plot_all_iterations(sols, problem, dt=0.02)

trajectory_fn = get_trajectory_function(sols, problem)

time, feet_positions, base_positions, base_rotations = trajectory_fn(0.02)  # positions shape: (N, 4, 3)

