import numpy as np

########################################################################
#                              Data splits                             #
########################################################################

TILES = {
    "train": [
        "train_tile_1",
        "train_tile_2",
        "train_tile_3",
        "train_tile_4",
        "train_tile_5",
        "train_tile_6",
        "train_tile_7",
        "train_tile_8",
        "train_tile_9",
        "train_tile_10",
        "train_tile_11",
        "train_tile_12",
        "train_tile_13",
        "train_tile_14",
        "train_tile_15",
        "train_tile_16",
        "train_tile_17",
        "train_tile_18",
        "train_tile_19",
        "train_tile_20",
        "train_tile_21",
        "train_tile_22",
        "train_tile_23",
        "train_tile_24",
        "train_tile_25",
        "train_tile_26",
        "train_tile_27",
        "train_tile_28",
        "train_tile_29",
        "train_tile_30",
    ],
    "val": [
        "val_tile_1",
        "val_tile_2",
        "val_tile_3",
        "val_tile_4",
        "val_tile_5",
        "val_tile_6",
        "val_tile_7",
        "val_tile_8",
        "val_tile_9",
        "val_tile_10",
    ],
    "test": [
        "test_tile_1",
        "test_tile_2",
        "test_tile_3",
        "test_tile_4",
        "test_tile_5",
        "test_tile_6",
        "test_tile_7",
        "test_tile_9",
        "test_tile_11",
        "test_tile_12",
    ]
}

########################################################################
#                                Labels                                #
########################################################################

CLASS_NAMES = [
    "Unclassified",
    "Rail",
    "Wiring",
    "Catenary pole",
    "Installation",
    "Crossing",
    "Switch box",
    "Signalling",
    "Unknown"
]

RAILCLOUDHDF_NUM_CLASSES = len(CLASS_NAMES) - 1

CLASS_COLORS = np.asarray([
    [255, 168, 130],
    [147, 219, 197],
    [225, 157, 204],
    [55, 192, 200],
    [54, 90, 175],
    [255, 126, 52],
    [255, 54, 37],
    [251, 232, 83]
])

MIN_OBJECT_SIZE = 100
THING_CLASSES = [3, 4, 5, 6, 7]
STUFF_CLASSES = [i for i in range(RAILCLOUDHDF_NUM_CLASSES) if not i in THING_CLASSES]
