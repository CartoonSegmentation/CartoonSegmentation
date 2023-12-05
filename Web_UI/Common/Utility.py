import numpy as Number


# Pending
def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """

    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = Number.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = Number.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05

    return mask