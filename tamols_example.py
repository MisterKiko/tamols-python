import jax.numpy as jnp
from tamols.tamols_dataclasses import Gait, Terrain, Robot, CurrentState
from tamols import build_initial_x0, TAMOLS, plot_all_iterations, get_trajectory_function
from tamols.helpers import transform_inertia
from tamols.manual_heightmaps import get_flat_heightmap, get_rough_terrain_heightmap
from tamols.map_test import show_map
from tamols.map_processing import save_heightmap_to_png

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

terrain = Terrain(
    heightmap = h,
    grid_cell_length = 0.01,  # Length of grid cells in heightmap
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
    initial_base_pose=jnp.array([0.0, 0.0, 0.445, 0.0, 0.0, 0.0]),
    initial_base_velocity=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

problem = TAMOLS(gait, terrain, robot)

sols, infos = problem.run_repeated_optimizations({
            "max_iter": 300,
            "acceptable_tol": 1e-3
        })

plot_all_iterations(sols, problem, dt=0.02)

save_heightmap_to_png(terrain.heightmap, "heightmap.png")

trajectory_fn = get_trajectory_function(sols, problem)

time, feet_positions, base_positions, base_rotations = trajectory_fn(0.02)  # positions shape: (N, 4, 3)

