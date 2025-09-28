from scipy import ndimage
import numpy as np

def compute_heightmap_gradients(height_map,grid_cell_length):
    """
    Compute gradients using 1D 5-point central finite difference kernel
    Args:
        height_map: 2D height map array
    Returns:
        gradient_x: Gradient in x direction
        gradient_y: Gradient in y direction
    """
    kernel = np.array([-1, 8, 0, -8, 1]) / (12 * grid_cell_length)
    
    # Compute first derivatives
    gradient_x = ndimage.convolve1d(height_map, kernel, axis=1)
    gradient_y = ndimage.convolve1d(height_map, kernel, axis=0)
    
    return gradient_x, gradient_y

def get_hs1(h_raw,sigma=0.5):
    """
    Compute the Gaussian filtered heightmap.
    Args:
        h_raw: Raw heightmap data
    Returns:
        h_s1: Smoothed heightmap
    """
    h_s1 = ndimage.gaussian_filter(h_raw, sigma)
    return h_s1

def get_hs2(h_raw, median_kernel=25, dilation_kernel=25, sigma=5.0, limit=0.05):
    """
    Compute the virtual floor heightmap.
    Args:
        h_s1: Smoothed heightmap from get_hs1
    Returns:
        h_s2: Virtual floor heightmap
    """
    h_median= ndimage.median_filter(h_raw, median_kernel)

    delta_h=h_raw-h_median

    mask=np.zeros_like(h_raw)
    mask[delta_h>limit] = 1
    mask[delta_h<-limit] = 1

    h_dilated=ndimage.maximum_filter(h_raw,dilation_kernel)
    h_s2 = np.where(mask==1, h_dilated, h_raw)
    h_s2 = ndimage.gaussian_filter(h_s2, sigma)

    return h_s2