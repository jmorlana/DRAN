"""Helper functions to perform exhaustive search."""
from dran.utils.colmap import Camera
import gin
import torch
import numpy as np
from typing import Union, Tuple, List, Dict, NamedTuple
import matplotlib.pyplot as plt
import time


def retrieve_argmax(correspondence_map, factor,ratio=0.9):
    """Use a modified ratio test to obtain correspondences."""
    #TODO: optimize this fking function
    channels, width, height = correspondence_map.shape
    indices = torch.zeros((channels, 2))
    mask = torch.zeros((channels,))
    top_k = int(factor * width * height)
    for i, window in enumerate(correspondence_map):
        with torch.no_grad():
            dist_nn, ind = window.view(-1).topk(top_k, dim=-1, largest=True)
            match_ok = ((ratio ** 2) * dist_nn[0] >= dist_nn[-1])
            a_y, a_x = np.unravel_index(ind[0].item(), (width, height))
            indices[i,:] = torch.FloatTensor((a_x, a_y))
            mask[i] = match_ok.item()
    return indices, mask.cpu().numpy().astype(bool)

@gin.configurable
def exhaustive_search(dense_descriptor_map: torch.FloatTensor,
                      sparse_descriptor_map: torch.FloatTensor,
                      image_shape: List[int],
                      map_size: List[int],
                      cell_size: List[int],
                      factor: float):
    """ Perform exhaustive dense matching.

    Args:
        dense_descriptor_map: The dense query hypercolumn feature map.
        sparse_descriptor_map: The sparse reference set of hypercolumn.
        image_shape: The original image width and height.
        cell_size: The cell size in the reference image.
        factor: The thresholding factor for the ratio test.
    Returns:
        argmax_in_query_space: The set of corresponding points in query space.
        query_cell_sizes: The cell size in the query image.
        mask: The mask of valid correspondences after the ratio test.
    """
    # Compute correspondence maps map
    width, height = list(map(int, map_size))

    correspondence_map = torch.mm(
        sparse_descriptor_map, dense_descriptor_map).view(-1, width, height)

    # Find 2D maximums in correspondence map coordinates
    cell_sizes = torch.DoubleTensor(cell_size)
    indices, mask = retrieve_argmax(correspondence_map, factor)

    # Compute query argmax coordinates in image space
    map_size = torch.FloatTensor(list(correspondence_map.shape[1:]))
    #image_shape = torch.FloatTensor(image_shape)
    image_shape_tensor = torch.FloatTensor(map_size.shape)
    image_shape_tensor[0] = image_shape[0]
    image_shape_tensor[1] = image_shape[1]

    query_cell_sizes = image_shape_tensor / map_size
    argmax_in_query_space = (indices * query_cell_sizes) + (query_cell_sizes / 2)
    # for i in range(correspondence_map.shape[0]):
    #     p1 = correspondence_map[i,:,:].unsqueeze(0)
    #     plt.imshow(p1.permute(1, 2, 0).cpu()  )
    #     plt.show()
    #     print('done')
    #     i += 1 
    return argmax_in_query_space, mask

def pack_to_s2dhm(dense_query_map: List,
                  sparse_ref_map: dict):
    
    # Hypercolumns already normalized!
    channels, width, height = dense_query_map.shape[1:]
    dense_query_feats = dense_query_map.squeeze().view((channels, -1))

    sparse_ref_feats = torch.zeros(len(sparse_ref_map), channels)
    idx = 0
    for key, value in sparse_ref_map.items():
        sparse_ref_feats[idx] = value[0]
        idx += 1

    return sparse_ref_feats, dense_query_feats, width, height

def scale_to_image(ref2D_points: torch.FloatTensor, image_shape: List[int], map_size: List[int]):

    width, height = list(map(int, map_size))
    map_size = torch.FloatTensor([width, height])
    image_shape_tensor = torch.FloatTensor([image_shape[0], image_shape[1]])

    query_cell_sizes = image_shape_tensor / map_size
    points2D_ref_sc = (ref2D_points * query_cell_sizes) + (query_cell_sizes / 2)

    return points2D_ref_sc

def get_camera(q_camera: Camera, scales_query: Union[float, int, Tuple[Union[float, int]]]):
    new_camera = q_camera.scale(scales_query)

    distortion = np.zeros(4)
    distortion[0] = new_camera.dist[0]
    distortion[1] = new_camera.dist[1]

    intrinsics = np.zeros(shape=(3,3))
    intrinsics[0,0] = new_camera.f[0]
    intrinsics[1,1] = new_camera.f[1]
    intrinsics[0,2] = new_camera.c[0]
    intrinsics[1,2] = new_camera.c[1]
    intrinsics[2,2] = 1.0
    return distortion, intrinsics