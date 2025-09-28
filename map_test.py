from manual_heightmaps import *
from map_processing import *
import matplotlib.pyplot as plt

def show_map(h):

    h_s1 = get_hs1(h)
    h_s2 = get_hs2(h)

    y = np.arange(h.shape[0])
    x = np.arange(h.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 5))

    zmin,zmax= -0.1, 0.5

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, h, cmap='terrain')
    ax1.set_title("Original h")
    ax1.set_zlim(zmin, zmax)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Height (Z)")

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, h_s1, cmap='terrain')
    ax2.set_title("h_s1")
    ax2.set_zlim(zmin, zmax)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Height (Z)")

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, h_s2, cmap='terrain')
    ax3.set_title("h_s2")
    ax3.set_zlim(zmin, zmax)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Height (Z)")

    # Plot gradients on a separate 2D figure
    grad_x, grad_y = compute_heightmap_gradients(h_s1, grid_cell_length=0.05)
    skip = 1

    fig2, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        h_s1, origin='lower', cmap='terrain',
        extent=[0, h.shape[1], 0, h.shape[0]]
    )
    ax.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        grad_x[::skip, ::skip], grad_y[::skip, ::skip],
        color='red', scale=10, width=0.003
    )

    ax.set_title("Gradient field on h_s1")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    #heightmap = get_heightmap_with_platforms_and_holes(a=100,b=100,hole_size=10,hole_number=2,platform_size=10,platform_height=0.1)
    #heightmap = get_stairs_heightmap(a=300, b=300, start_col=200, step_depth=30, step_height=0.05)  # Heightmap with stairs
    heightmap = get_rough_terrain_heightmap(a=350, b=350, sigma=0.02, platform_height=0.0, platform_size=5, smooth_sigma=2, seed=42) # Heightmap with platforms

    save_heightmap_to_png(heightmap, "outputs/heightmap.png")

    show_map(h=heightmap)