import jax.numpy as jnp
from tamols.tamols_dataclasses import Gait, Terrain, Robot
from tamols import TAMOLS, plot_all_iterations, get_trajectory_function
from tamols.helpers import transform_inertia
from tamols.manual_heightmaps import get_flat_heightmap, get_rough_terrain_heightmap, get_stairs_heightmap
from tamols.map_processing import save_heightmap_to_png, show_map

gait = Gait(
    n_steps=6,
    n_phases=2,
    spline_order=5,
    tau_k=jnp.array([0.5, 0.5]), # Time duration of phases [s]
    h_des=0.445,
    eps_min=0.1,
    weights = jnp.array([10.0e4, 0.001, 7.0, 100.0, 3.0, 0.01, 2.0, 0.001]),
    desired_base_velocity = jnp.array([0.3, 0.0, 0.0]),
    desired_base_angular_velocity = jnp.array([0.0, 0.0, 0.0]),
    apex_height=0.1,
    contact_schedule= jnp.array([
        [1, 0, 0, 1],  # Phase 0 FL and RR in swing
        [0, 1, 1, 0],  # Phase 1 FR and RL in swing
    ]),
    at_des_position= jnp.array([
        [0, 1, 1, 0],  # Phase 0 FL and RR should be at desired pos
        [1, 1, 1, 1],  # Phase 1 All feet should be at desired pos
    ])
)

# h = jnp.array(get_flat_heightmap(a=300, b=300, height=0.0))  # Flat heightmap for testing
# h = jnp.array(get_rough_terrain_heightmap(a=400, b=400, sigma=0.04, platform_height=0.0, platform_size=5, smooth_sigma=3, seed=917)) # Heightmap with platforms
h = jnp.array(get_stairs_heightmap(a=300, b=300, start_col=200, step_depth=30, step_height=0.05))  # Heightmap with stairs


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
    r_1=jnp.array([0.1934,0.0465+0.0955,0.0]), # Hip joint offset positions in base frame
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

sols, infos = problem.run_repeated_optimizations(warm_start=True, options=
        {
            "max_iter": 300,
            "acceptable_tol": 1e-3
        })

plot_all_iterations(sols, problem, dt=0.02)

save_heightmap_to_png(terrain.heightmap, "outputs/heightmap.png")

trajectory_fn = get_trajectory_function(sols, problem)

time, feet_positions, base_positions, base_rotations = trajectory_fn(0.02)  # positions shape: (N, 4, 3)

