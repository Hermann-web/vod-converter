import glob
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import velo_to_bev.config as cnf
import velo_to_bev.kitti_aug_utils as aug_utils
import velo_to_bev.kitti_bev_utils as bev_utils
import velo_to_bev.kitti_utils as kitti_utils

logger = logging.getLogger(__name__)


class KittiYOLODataset:

    def __init__(self, imageset_dir: os.PathLike):
        self.lidar_path = Path(imageset_dir) / "velodyne"
        self.label_path = Path(imageset_dir) / "label_2"
        self.calib_path = Path(imageset_dir) / "calib"
        self.files = sorted(glob.glob("%s/*.bin" % self.lidar_path))
        self.image_idx_list = [os.path.split(x)[1].split(".")[0].strip() for x in self.files]
        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

    def __len__(self):
        return len(self.sample_id_list)

    def parse_idx(self, idx):
        return int(idx) + 0

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, '%06d.bin' % self.parse_idx(idx))
        assert os.path.exists(lidar_file), f"{lidar_file} not found"
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx) -> List[kitti_utils.Object3d]:
        label_file = os.path.join(self.label_path, '%06d.txt' % self.parse_idx(idx))
        assert os.path.exists(label_file), f"label_file {label_file} not found"
        return kitti_utils.read_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%06d.txt' % self.parse_idx(idx))
        assert os.path.exists(calib_file)
        return kitti_utils.Calibration(calib_file)

    def get_bev_map(self, sample_id):
        lidarData = self.get_lidar(sample_id)

        b = bev_utils.removePoints(lidarData, cnf.boundary, reduce_resolution=0)
        rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

        return rgb_map

    def get_bev_targets(self, sample_id):
        # get label and calibration data
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)

        labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)

        if not noObjectLabels:
            labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                          calib.P)  # convert rect cam to velo cord
        target = bev_utils.build_yolo_target(labels)
        logger.debug(f"target:{target.shape}")
        ntargets = 0
        for i, t in enumerate(target):
            if t.sum(0):
                ntargets += 1
        logger.debug(f"ntargets:{ntargets}")
        targets = torch.zeros((ntargets, 8))
        for i, t in enumerate(target):
            if t.sum(0):
                targets[i, 1:] = torch.from_numpy(t)

        return targets

    def __getitem__(self, index):
        sample_id = int(self.sample_id_list[index])
        # get bev features
        rgb_map = self.get_bev_map(sample_id)
        # Projected targets
        targets = self.get_bev_targets(sample_id)
        return rgb_map, targets


def get_bev_dataset(data_folder: Path, bev_folder: Path):
    dataset = KittiYOLODataset(imageset_dir=data_folder)

    img_size = cnf.BEV_WIDTH

    bev_dataset: List[Dict] = []

    for batch_i, (bev_map, targets) in enumerate(dataset):

        print(f">>> batch_i={batch_i} targets:{targets.shape}")

        # Rescale target
        targets[:, 2:6] *= cnf.BEV_WIDTH
        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        bev_map = torch.from_numpy(bev_map).type(torch.FloatTensor)
        bev_maps = torch.squeeze(bev_map).numpy()

        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map

        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)

        bev_fname = str(bev_folder / str(batch_i).zfill(6)) + ".png"
        cv2.imwrite(bev_fname, RGB_Map)

        img_display = RGB_Map

        bounding_boxes = []
        for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
            # Draw rotated box
            logger.debug(f"---c = {c}, x, y = {x, y}, w, l = {w, l}, yaw = {yaw}")
            c = int(c)  # float to int
            corners_int = bev_utils.drawRotatedBox(img_display, x, y, w, l, yaw, cnf.colors[c])
            assert corners_int.ndim == 2 and corners_int.shape[1] == 2
            # Extract x and y coordinates
            x_coords = corners_int[:, 0]
            y_coords = corners_int[:, 1]

            # Calculate the bounding box
            bounding_box = {
                'label': cnf.class_list[c],
                'left': np.min(x_coords),
                'right': np.max(x_coords),
                'top': np.min(y_coords),
                'bottom': np.max(y_coords)
            }
            bounding_boxes.append(bounding_box)

        # cv2.imshow('img-kitti-bev', img_display)

        # if cv2.waitKey(0) & 0xff == 27:
        #     break

        bev_dataset.append({"bev_fname": bev_fname, "bounding_boxes": bounding_boxes})

    return bev_dataset


if __name__ == "__main__":
    data_folder = Path(r"/home/ubuntu/Documents/share_remote/kitti-min")
    data_folder = data_folder / "training"
    bev_folder = data_folder / "bev"
    bev_folder.mkdir(exist_ok=True)
    bev_dataset = get_bev_dataset(data_folder, bev_folder)
