import logging
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

import vod_converter.velo_to_bev.config as cnf
from vod_converter.velo_to_bev.kitti_utils import Object3d

logger = logging.getLogger(__name__)


def removePoints(PointCloud, BoundaryCond, reduce_resolution=0):
    """
    reduce_resolution (float): The percentage of the point cloud to remove regularly (0 to 1).
    """

    # Calculate the step size based on the reduce_resolution
    if reduce_resolution < 0 or reduce_resolution > 1:
        raise ValueError("reduce_resolution should be between 0 and 1")
    elif reduce_resolution == 1:
        raise ValueError("reduce_resolution should not be maximal = 1")

    if reduce_resolution > 0:
        step = int(1 / (1 - reduce_resolution))
        PointCloud = PointCloud[::step]

    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) &
                    (PointCloud[:, 1] >= minY) & (PointCloud[:, 1] <= maxY) &
                    (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] - minZ

    return PointCloud


def makeBVFeature(PointCloud_, Discretization, bc, projection_plane='xy'):
    """
    Create a BEV feature map from a point cloud, projected onto the specified plane.
    
    Parameters:
    PointCloud_ (numpy.ndarray): The input point cloud.
    Discretization (float): The discretization factor.
    bc (dict): Boundary conditions with 'maxX', 'minX', 'maxZ', and 'minZ'.
    projection_plane (str): The plane to project onto. 
                            'xy' for the ground plane (default), 'yz' for the image plane.
    
    Returns:
    numpy.ndarray: The RGB BEV feature map.
    """
    PointCloud = np.copy(PointCloud_)

    # Set up the height and width for the BEV feature map
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1
    max_counts = cnf.MAX_COUNTS

    # Dynamically change axes based on projection plane
    if projection_plane == 'xy':
        # Ground plane projection (z = 0)
        x_idx, y_idx, height_idx = 0, 1, 2  # x and y as axes, z as "height"
        max_height = float(np.abs(bc['maxZ'] - bc['minZ']))  # Use Z-range for normalization
    elif projection_plane == 'yz':
        # Image plane projection (x = 0)
        x_idx, y_idx, height_idx = 2, 1, 0  # z and y as axes, x as "height"
        max_height = float(np.abs(bc['maxX'] - bc['minX']))  # Use X-range for normalization
    else:
        raise ValueError("Invalid projection_plane. Use 'xy' or 'yz'.")

    # Discretize the feature map
    PointCloud[:, y_idx] = np.int_(np.floor(PointCloud[:, y_idx] / Discretization) + Width / 2)
    PointCloud[:, x_idx] = np.int_(np.floor(PointCloud[:, x_idx] / Discretization))

    # Sort points (sort by height, then by y, then by x)
    indices = np.lexsort((-PointCloud[:, height_idx], PointCloud[:, y_idx], PointCloud[:, x_idx]))
    PointCloud = PointCloud[indices]

    # Height Map: normalize based on height range
    heightMap = np.zeros((Height, Width))

    _, unique_indices, counts = np.unique(PointCloud[:, [x_idx, y_idx]],
                                          axis=0,
                                          return_index=True,
                                          return_counts=True)

    PointCloud_top = PointCloud[unique_indices]
    heightMap[np.int_(PointCloud_top[:, x_idx]),
              np.int_(PointCloud_top[:, y_idx])] = PointCloud_top[:, height_idx] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(max_counts))

    intensityMap[np.int_(PointCloud_top[:, x_idx]),
                 np.int_(PointCloud_top[:, y_idx])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, x_idx]),
               np.int_(PointCloud_top[:, y_idx])] = normalizedCounts

    # Create the RGB map (r = density, g = height, b = intensity)
    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map (density)
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map (height)
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map (intensity)

    return RGB_Map


def read_labels_for_bevbox(objects: List[Object3d]) -> Tuple[np.ndarray, bool]:
    bbox_selected = []
    for obj in objects:
        logger.debug(f"obj.cls_id:{obj.cls_id}")
        if obj.cls_id != -1:
            bbox = [obj.cls_id, obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry]
            bbox_selected.append(bbox)

    if (len(bbox_selected) == 0):
        return np.zeros((1, 8), dtype=np.float32), True
    else:
        bbox_selected = np.array(bbox_selected).astype(np.float32)
        return bbox_selected, False


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    # Ensure width and length are within BEV_WIDTH and BEV_HEIGHT
    assert w < cnf.BEV_WIDTH, f"Error: Width {w} exceeds BEV_WIDTH {cnf.BEV_WIDTH}"
    assert l < cnf.BEV_HEIGHT, f"Error: Length {l} exceeds BEV_HEIGHT {cnf.BEV_HEIGHT}"

    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    return bev_corners


def build_yolo_target(labels: np.ndarray, projection_plane='xy') -> np.ndarray:
    """
    Build YOLO targets based on the given labels and projection plane.
    
    Parameters:
    labels (np.ndarray): Array of object labels with [class, x, y, z, h, w, l, yaw].
    projection_plane (str): The projection plane, 'xy' for ground plane or 'yz' for image plane.
    
    Returns:
    np.ndarray: Target array with bounding box and orientation information.
    """
    bc = cnf.boundary
    target = np.zeros([50, 7], dtype=np.float32)

    index = 0

    # Define axis mapping based on the projection plane
    if projection_plane == 'xy':
        x_idx, y_idx = 0, 1  # Project onto (x, y) and adjust w, l
        minX, maxX, minY, maxY = bc['minX'], bc['maxX'], bc['minY'], bc['maxY']
    elif projection_plane == 'yz':
        x_idx, y_idx = 2, 1  # Project onto (y, z) and adjust l, h
        minX, maxX, minY, maxY = bc['minZ'], bc['maxZ'], bc['minY'], bc['maxY']
    else:
        raise ValueError("Invalid projection_plane. Use 'xy' for ground or 'yz' for image plane.")

    logger.debug(f"labels:{labels.shape}")

    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]

        # ped and cyc labels are very small, so lets add some factor to height/width
        l = l + 0.3
        w = w + 0.3

        pos_data = [x, y, z]
        dis_data = [l, w, h]
        x, y = pos_data[x_idx], pos_data[y_idx]
        l, w = dis_data[x_idx], dis_data[y_idx]

        # Adjust yaw to be relative to full circle
        yaw = np.pi * 2 - yaw

        # Check if the object is within the boundary conditions
        if (x > minX) and (x < maxX) and (y > minY) and (y < maxY):
            # Normalize the coordinates to be in the range [0, 1]
            y1 = (y - minY) / (maxY - minY)
            x1 = (x - minX) / (maxX - minX)

            # Normalize width and length according to the boundary conditions
            w1 = w / (maxY - minY)
            l1 = l / (maxX - minX)

            # Fill in the target array with class, position, size, and orientation
            target[index][0] = cl
            target[index][1] = y1
            target[index][2] = x1
            target[index][3] = w1
            target[index][4] = l1
            target[index][5] = math.sin(float(yaw))
            target[index][6] = math.cos(float(yaw))

            index += 1

    return target


def inverse_yolo_target(targets: np.ndarray, bc: Dict[str, float], add_conf: bool = False):
    nb_target_features = 9
    ntargets = 0
    for i, t in enumerate(targets):
        if t.sum(0):
            ntargets += 1

    if not add_conf:
        assert targets.shape[1] == nb_target_features - 1
        # add a col filled with 0
        targets = np.column_stack((targets, np.zeros(targets.shape[0])))

    assert targets.shape[1] == nb_target_features

    labels = np.zeros([ntargets, nb_target_features + 1], dtype=np.float32)

    n = 0
    for t in targets:
        if t.sum(0) == 0:
            continue

        c, y, x, w, l, im, re, conf, cls_conf = t
        z, h = -1.55, 1.5
        if c == 1:
            h = 1.8
        elif c == 2:
            h = 1.4

        y = y * (bc["maxY"] - bc["minY"]) + bc["minY"]
        x = x * (bc["maxX"] - bc["minX"]) + bc["minX"]
        w = w * (bc["maxY"] - bc["minY"])
        l = l * (bc["maxX"] - bc["minX"])

        w -= 0.3
        l -= 0.3

        labels[n, :] = c, x, y, z, h, w, l, -np.arctan2(im, re) - 2 * np.pi, conf, cls_conf
        n += 1

    return labels


# send parameters in bev image coordinates format
def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.astype(int)
    cv2.polylines(img, [corners_int.reshape(-1, 1, 2)], True, color, 2)
    logger.debug(f"corners_int = {corners_int}")
    return corners_int
