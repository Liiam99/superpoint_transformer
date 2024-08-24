import argparse
import hydra
import laspy
import numpy as np
import torch
from pathlib import Path

from src.utils import init_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory where the results should be exported to."
    )
    config = parser.parse_args()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse the configs using hydra
    cfg = init_config(overrides=[
        "experiment=semantic/internrail",
        "ckpt_path=./logs/train/runs/2024-08-16_16-59-59/checkpoints/epoch_259.ckpt",
        "datamodule.load_full_res_idx=True"
    ])

    # Instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()

    dataset = datamodule.test_dataset
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(cfg.ckpt_path)
    model = model.eval().to("cuda:0")

    raw_dir = Path(dataset.raw_dir)
    for idx, tile in enumerate(dataset):
        nag = dataset.on_device_transform(tile.to("cuda:0"))

        with torch.no_grad():
            output = model(nag)

        raw_semseg_y = output.full_res_semantic_pred(
            super_index_level0_to_level1=nag[0].super_index,
            sub_level0_to_raw=nag[0].sub)

        las_path = raw_dir / dataset.id_to_relative_raw_path(dataset.cloud_ids[idx])
        las = laspy.read(las_path)

        # Add the predicted labels to the point cloud.
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="prediction",
            type=np.uint8,
        ))
        las.prediction = raw_semseg_y.cpu()

        save_path = output_dir / las_path.name
        las.write(save_path)
