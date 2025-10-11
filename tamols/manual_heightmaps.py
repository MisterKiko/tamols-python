import numpy as np
from scipy import ndimage

def get_flat_heightmap(a=50, b=50, height=0.0):
    """
    Generate a flat heightmap with a constant height.
    a: number of rows
    b: number of columns
    height: constant height value

    Returns a 2D numpy array representing the heightmap.
    """
    return np.full((a, b), height)

def get_rough_terrain_heightmap(a=50, b=50, sigma=0.01, platform_height=0.0, platform_size=5, smooth_sigma=1.0, seed=None):
    """
    Generate a heightmap divided into platform-sized parcels, each with a random height,
    but with smooth transitions so that nearby platforms have similar heights.
    a: number of rows
    b: number of columns
    sigma: standard deviation of the Gaussian noise for each parcel
    platform_height: mean height of the platforms
    platform_size: size of the platform (number of rows/columns)
    smooth_sigma: standard deviation for Gaussian smoothing of the platform heights
    """
    rows = a // platform_size
    cols = b // platform_size
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    # Generate random heights for each platform parcel
    platform_heights = np.random.normal(loc=platform_height, scale=sigma, size=(rows, cols))
    # Smooth the platform heights so neighbors are similar
    platform_heights = ndimage.gaussian_filter(platform_heights, sigma=smooth_sigma, mode='nearest')
    # Expand to full heightmap
    heightmap = np.zeros((a, b))
    for i in range(rows):
        for j in range(cols):
            row_start = i * platform_size
            row_end = min((i + 1) * platform_size, a)
            col_start = j * platform_size
            col_end = min((j + 1) * platform_size, b)
            heightmap[row_start:row_end, col_start:col_end] = platform_heights[i, j]
    return heightmap

def get_stairs_heightmap(a=150, b=150, step_height=0.05, step_depth=25, step_width=None, start_col=0):
    """
    Generate a heightmap representing stairs rising in the x direction (columns).
    a: number of rows
    b: number of columns
    step_height: height increment for each step
    step_depth: depth (number of columns) per step
    step_width: width of the stairs (number of rows), defaults to full height
    start_col: the column index where the first step starts

    Returns a 2D numpy array representing the heightmap.
    """
    if step_width is None:
        step_width = a  # Full height by default

    heightmap = np.zeros((a, b))
    # Adjust the number of steps to fit within the available columns
    num_steps = (b - start_col) // step_depth

    for step in range(num_steps):
        col_start = start_col + step * step_depth
        col_end = min(start_col + (step + 1) * step_depth, b)
        row_start = (a - step_width) // 2
        row_end = row_start + step_width
        heightmap[row_start:row_end, col_start:col_end] = step_height * (step + 1)

    return heightmap

def get_heightmap_ramp(a=150, b=150, ramp_height=0.5, ramp_depth=100, ramp_width=None, start_col=0):
    """
    Generate a heightmap representing a linear ramp rising in the x direction (columns).
    a: number of rows
    b: number of columns
    ramp_height: final height at the end of the ramp (meters)
    ramp_depth: number of columns over which the ramp rises
    ramp_width: width of the ramp in rows; defaults to full height (a)
    start_col: column index where the ramp starts
    Returns a 2D numpy array (a, b).
    """
    if ramp_width is None:
        ramp_width = a  # Full height by default

    heightmap = np.zeros((a, b), dtype=float)

    # Clamp ramp region to map bounds
    col_start = max(0, int(start_col))
    col_end = min(int(start_col + ramp_depth), b)
    if col_start >= b or col_start >= col_end:
        return heightmap  # nothing to draw

    # Center the ramp in rows
    row_start = max(0, (a - int(ramp_width)) // 2)
    row_end = min(a, row_start + int(ramp_width))

    # Linear rise from 0 to ramp_height across [col_start, col_end)
    n_cols = col_end - col_start
    ramp_profile = np.linspace(0.0, float(ramp_height), n_cols, endpoint=True)

    for j in range(col_start, col_end):
        heightmap[row_start:row_end, j] = ramp_profile[j - col_start]

    return heightmap