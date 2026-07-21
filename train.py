# import os

# from lightning import Trainer, LightningModule, LightningDataModule
# from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import Logger, WandbLogger

import os

from lightning.pytorch import Trainer, LightningModule, LightningDataModule  # ← lightning.pytorch に変更
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger

# from brain_mri.data import MRIDataModule
from hyda.utils import namespace_to_dict


class SaveConfigWandB(SaveConfigCallback):
    def __init__( self, *args,  **kwargs) -> None:
        super().__init__(save_to_log_dir=False, *args, **kwargs)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            logger : WandbLogger = trainer.logger
            # 変更: PACS用のプロジェクト名 'pacs_uda' の場合も、データを綺麗にシリアライズ（dict化）するように条件を追加します
            is_pacs_or_mri = (logger.name == 'mri_uda' or logger.name == 'pacs_uda')
            data = namespace_to_dict(self.config.data) if is_pacs_or_mri else self.config.data
            logger.experiment.config.update(dict(data=data, model=namespace_to_dict(self.config.model)))

class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__( self, *args,  **kwargs) -> None:
        super().__init__(save_to_log_dir=False, *args, **kwargs)
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})

class CLI(LightningCLI):
    def add_default_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--seed_everything', type=int, help='Seed for reproducibility')

        parser.add_argument("experiment_name", type=str, help="Name of the experiment")
        parser.add_argument('--trainer.logger.init_args.name', type=str, help='W&B experiment name')
        parser.link_arguments(source="experiment_name", target="trainer.logger.init_args.name", apply_on="parse")
        parser.add_lightning_class_args(ModelCheckpoint, "chkpt_callback")
        parser.link_arguments(source="experiment_name", target="chkpt_callback.filename", apply_on="parse",
                              compute_fn=lambda x: x + "_{epoch:03d}")
        parser.link_arguments(source="experiment_name", target="chkpt_callback.dirpath", apply_on="parse",
                              compute_fn=lambda x: f"{os.environ['HOME']}/ckpts/{x}")

# def cli_main():
#     cli = CLI(model_class=LightningModule, subclass_mode_model=True, save_config_callback=SaveConfigWandB)

def cli_main():
    # model_class=LightningModule を削除します
    cli = CLI(
        subclass_mode_model=True,
        save_config_callback=SaveConfigWandB,
    )

if __name__ == "__main__":
    cli_main()