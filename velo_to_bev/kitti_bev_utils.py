from typing import Dict
import numpy as np
import math
import config as cnf


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


def makeBVFeature(PointCloud_, Discretization, bc):
    """
    Create a BEV feature map from a point cloud.
    
    Parameters:
    PointCloud_ (numpy.ndarray): The input point cloud.
    Discretization (float): The discretization factor.
    bc (dict): Boundary conditions with 'maxZ' and 'minZ'.
    
    Returns:
    numpy.ndarray: The RGB BEV feature map.
    """
    PointCloud = np.copy(PointCloud_)

    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1
    max_counts = cnf.MAX_COUNTS

    # Discretize Feature Map
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2],
                                   axis=0,
                                   return_index=True,
                                   return_counts=True)

    PointCloud_top = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]),
              np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(max_counts))

    intensityMap[np.int_(PointCloud_top[:, 0]),
                 np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    return RGB_Map


def read_labels_for_bevbox(objects):
    bbox_selected = []
    for obj in objects:
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


def build_yolo_target(labels):
    bc = cnf.boundary
    target = np.zeros([50, 7], dtype=np.float32)

    index = 0
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]

        # ped and cyc labels are very small, so lets add some factor to height/width
        l = l + 0.3
        w = w + 0.3

        yaw = np.pi * 2 - yaw
        if (x > bc["minX"]) and (x < bc["maxX"]) and (y > bc["minY"]) and (y < bc["maxY"]):
            y1 = (y - bc["minY"]) / (bc["maxY"] - bc["minY"]
                                    )  # we should put this in [0,1], so divide max_size  80 m
            x1 = (x - bc["minX"]) / (bc["maxX"] - bc["minX"]
                                    )  # we should put this in [0,1], so divide max_size  40 m
            w1 = w / (bc["maxY"] - bc["minY"])
            l1 = l / (bc["maxX"] - bc["minX"])

            target[index][0] = cl
            target[index][1] = y1
            target[index][2] = x1
            target[index][3] = w1
            target[index][4] = l1
            target[index][5] = math.sin(float(yaw))
            target[index][6] = math.cos(float(yaw))

            index = index + 1

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

