import laspy
import numpy as np
import torch
from src.data import Data
from src.datasets import BaseDataset
from src.datasets.railcloudhdf_config import *


########################################################################
#                                 Utils                                #
########################################################################

def read_railcloudhdf_tile(
        file_path, xyz=True, intensity=True, semantic=True, instance=True):
    """PLACEHOLDER."""
    data = Data()

    # Extract coordinates [x, y, z], strength [0.->1.] & label [uint]
    las = laspy.read(file_path)

    if xyz:
        pos = torch.tensor([las.x, las.y, las.z], dtype=torch.float).T
        data.pos = pos - las.header.offset
        data.pos_offset = las.header.offset

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
        """PLACEHOLDER."""
        return TILES

    def download_dataset(self):
        pass

    def read_single_raw_cloud(self, raw_cloud_path):
        """PLACEHOLDER."""
        return read_railcloudhdf_tile(
            raw_cloud_path, intensity=True, semantic=True, instance=False)
