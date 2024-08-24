import numpy as np
import os
from posixpath import splitext
import pyrootutils

########################################################################
#                              Data splits                             #
########################################################################
root = str(pyrootutils.setup_root(__file__, pythonpath=True))
DATASET_DIR = root + "/data/InternRail/"

TILES = {
    "train": [splitext(file_name)[0] for file_name in os.listdir(DATASET_DIR + "train") if file_name.startswith("HdF")],
    "val": [splitext(file_name)[0] for file_name in os.listdir(DATASET_DIR + "val") if file_name.startswith("HdF")],
    "test": [splitext(file_name)[0] for file_name in os.listdir(DATASET_DIR + "test")[:10]]
}

########################################################################
#                                Labels                                #
########################################################################

CLASS_NAMES = [
    "Unlabelled",
    "Installation",
    "Crossing",
    "Switch box",
    "Signalling",
    "Unknown",
]

INTERNRAIL_NUM_CLASSES = len(CLASS_NAMES) - 1

CLASS_COLORS = np.asarray([
    [0, 0, 0],
    [54, 90, 175],
    [255, 126, 52],
    [255, 54, 37],
    [251, 232, 83]
])

MIN_OBJECT_SIZE = 100
THING_CLASSES = [1, 2, 3, 4]
STUFF_CLASSES = [i for i in range(INTERNRAIL_NUM_CLASSES) if not i in THING_CLASSES]
