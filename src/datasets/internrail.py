import laspy
import numpy as np
import os.path as osp
import torch
from src.data import Data
from src.datasets import BaseDataset
from src.datasets.internrail_config import *


########################################################################
#                                 Utils                                #
########################################################################

def read_internrail_tile(
        file_path, xyz=True, intensity=False, semantic=True):
    """PLACEHOLDER."""
    data = Data()

    # Extract coordinates [x, y, z], strength [0.->1.] & label [uint]
    las = laspy.read(file_path)

    if xyz:
        pos = torch.tensor([las.x, las.y, las.z], dtype=torch.float).T
        offset = torch.FloatTensor(las.header.offset)

        if "CSX" in file_path:
            FEET_TO_METERS = 1200/3937
            pos = pos*FEET_TO_METERS
            offset = offset*FEET_TO_METERS

        data.pos = pos - offset
        data.pos_offset = offset

    if intensity:
        intensity = np.array(las.intensity)
        intensity = intensity/np.iinfo(intensity.dtype).max
        data.intensity = torch.FloatTensor(intensity)

    if semantic:
        data.y = torch.LongTensor(las.classification)

    return data


########################################################################
#                              InternRail                              #
########################################################################

class InternRail(BaseDataset):
    """
    PLACEHOLDER

    PLACEHOLDER.
    """
    @property
    def raw_dir(self):
        return self.root

    @property
    def class_names(self):
        """List of class names of the InternRail dataset."""
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes of the InternRail dataset."""
        return INTERNRAIL_NUM_CLASSES

    @property
    def stuff_classes(self):
        """Labels of classes that are considered objects."""
        return STUFF_CLASSES

    @property
    def class_colors(self):
        """Colors of each class for the visualisation."""
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self):
        """The unique file names of each tile."""
        return TILES

    def download(self):
        pass

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        if id in self.all_cloud_ids['train']:
            stage = 'train'
        elif id in self.all_cloud_ids['val']:
            stage = 'val'
        elif id in self.all_cloud_ids['test']:
            stage = 'test'
        else:
            raise ValueError(f"Unknown tile id '{id}'")
        return osp.join(stage, self.id_to_base_id(id) + '.laz')

    def processed_to_raw_path(self, processed_path):
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]

        # Raw 'val' and 'trainval' tiles are all located in the
        # 'raw/train/' directory
        stage = 'train' if stage in ['trainval'] else stage

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = osp.join(self.raw_dir, stage, base_cloud_id + '.laz')

        return raw_path

    def read_single_raw_cloud(self, raw_cloud_path):
        """Returns one tile as a PyGData object with pos, intensity and label attrs."""
        return read_internrail_tile(
            raw_cloud_path, xyz=True, intensity=False, semantic=True)
