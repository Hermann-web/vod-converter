import os 
import numpy as np
import kitti_bev_utils as bev_utils
import config as cnf
import kitti_utils
from PIL import Image
import glob
from pathlib import Path

class KittiYOLODataset:

    def __init__(self, imageset_dir:os.PathLike):
        self.lidar_path = Path(imageset_dir) /  "velodyne"
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
    
    def get_label(self, idx):
        label_file = os.path.join(self.label_path, '%06d.txt' % self.parse_idx(idx))
        assert os.path.exists(label_file), f"label_file {label_file} not found"
        return kitti_utils.read_label(label_file)

    def __getitem__(self, index):
        sample_id = int(self.sample_id_list[index])

        lidarData = self.get_lidar(sample_id)

        b = bev_utils.removePoints(lidarData,
                                       cnf.boundary,
                                       reduce_resolution=0)
        rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

        # Projected targets
        
        labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)

        if not noObjectLabels:
            labels[:,
                    1:] = augUtils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                          calib.P)  # convert rect cam to velo cord
        target = bev_utils.build_yolo_target(labels)
        ntargets = 0
        for i, t in enumerate(target):
            if t.sum(0):
                ntargets += 1
        targets = torch.zeros((ntargets, 8))
        for i, t in enumerate(target):
            if t.sum(0):
                targets[i, 1:] = torch.from_numpy(t)

        return rgb_map
    
data_folder = Path(r"C:\Users\hermann.agossou\Documents\Data\KITTI\data_object_velodyne")
data_folder = data_folder / "testing"
bev_folder = data_folder / "bev"
bev_folder.mkdir(exist_ok=True)

dataset = KittiYOLODataset(imageset_dir=data_folder)

for index, bev_map in enumerate(dataset):
    index_str = str(index).zfill(6)
    bev_image_path = Path(bev_folder) / f"{index_str}.jpg"
    # Assuming 'bev_map' is a NumPy array representing the image
    # print("bev_map = ",bev_map.shape, bev_map.max())
    print('bev_image_path = ',bev_image_path)
    # Transpose the array from (3, 608, 608) to (608, 608, 3)
    bev_map_transposed = np.transpose(bev_map, (1, 2, 0))
    bev_map_uint8 = (bev_map_transposed * 255).astype(np.uint8)
    img = Image.fromarray(bev_map_uint8)
    # Display the image
    # img.show()
    # Save the image to the given path
    img.save(str(bev_image_path))
