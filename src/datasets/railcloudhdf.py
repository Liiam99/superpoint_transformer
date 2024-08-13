import laspy
import numpy as np
import os.path as osp
import torch
from src.data import Data
from src.datasets import BaseDataset
from src.datasets.railcloudhdf_config import *


########################################################################
#                                 Utils                                #
########################################################################

def read_railcloudhdf_tile(
        file_path, xyz=True, rgb=True, intensity=True, semantic=True):
    """PLACEHOLDER."""
    data = Data()

    # Extract coordinates [x, y, z], strength [0.->1.] & label [uint]
    las = laspy.read(file_path)

    if xyz:
        pos = torch.tensor([las.x, las.y, las.z], dtype=torch.float).T
        offset = torch.FloatTensor(las.header.offset)
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
#                             RailCloudHdF                             #
########################################################################

class RailCloudHdF(BaseDataset):
    """
    PLACEHOLDER

    PLACEHOLDER.
    """
    @property
    def raw_dir(self):
        return self.root

    @property
    def data_subdir_name(self):
        return self.__class__.__name__.lower() + f"/{self.stage}"

    @property
    def class_names(self):
        """List of class names of the RailCloud-HdF dataset."""
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes of the RailCloud-HdF dataset."""
        return RAILCLOUDHDF_NUM_CLASSES

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
        return f"{self.id_to_base_id(id)}.laz"

    def read_single_raw_cloud(self, raw_cloud_path):
        """Returns one tile as a PyGData object with pos, intensity and label attrs."""
        return read_railcloudhdf_tile(
            raw_cloud_path, xyz=True, rgb=True, intensity=True, semantic=True)
