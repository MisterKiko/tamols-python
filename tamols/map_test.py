from manual_heightmaps import *
from map_processing import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # heightmap = get_heightmap_with_platforms_and_holes(a=100,b=100,hole_size=10,hole_number=2,platform_size=10,platform_height=0.1)
    heightmap = get_stairs_heightmap(a=100, b=100, start_col=60, step_depth=10, step_height=0.10) # Heightmap with stairs
    # heightmap = get_rough_terrain_heightmap(a=350, b=350, sigma=0.05, platform_height=0.0, platform_size=5, smooth_sigma=3, seed=42) # Heightmap with platforms
    # heightmap = get_heightmap_ramp(a=200, b=400, ramp_height=0.2, ramp_depth=100, start_col=250)  # Heightmap with a ramp

    save_heightmap_to_png(heightmap, "outputs/heightmap.png")

    show_map(h=heightmap)