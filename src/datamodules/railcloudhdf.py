import logging
from src.datamodules.base import BaseDataModule
from src.datasets import RailCloudHdF


log = logging.getLogger(__name__)


class RailCloudHdFDataModule(BaseDataModule):
    """PLACEHOLDER."""
    _DATASET_CLASS = RailCloudHdF


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/semantic/railcloudhdf.yaml")
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
