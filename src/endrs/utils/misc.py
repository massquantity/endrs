from typing import Any, override

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm import tqdm


class LightningProgressBar(TQDMProgressBar):
    """A customized progress bar for PyTorch Lightning training and validation.
    
    Parameters
    ----------
    model_name : str
        Name of the model being trained, used for identification purposes.
    leave : bool
        Whether to leave the progress bar after completion of each epoch.
        If False, progress bars are cleared between epochs and a separator is printed.
    
    Notes
    -----
    This progress bar handles both training and validation phases separately,
    reinitializing for each epoch to ensure accurate progress tracking.
    """

    def __init__(self, model_name: str, leave: bool):
        super().__init__(leave=False)
        self.model_name = model_name
        self.leave = leave

    @override
    def on_train_start(self, *_: Any) -> None:
        pass

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, *_: Any):
        self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(self.total_train_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch + 1}")
        self.train_progress_bar.leave = self.leave
        # self.train_progress_bar.pos = 10

    @override
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.train_progress_bar.close()
        if not self.leave:
            tqdm.write("=" * 30)

    @override
    def on_train_end(self, *_: Any) -> None:
        pass

    @override
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass

    @override
    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not trainer.sanity_checking:
            self.val_progress_bar = self.init_validation_tqdm()
            # self.val_progress_bar.pos = -20
            self.val_progress_bar.leave = self.leave

    @override
    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        self.val_progress_bar.set_description(f"Eval Epoch {trainer.current_epoch + 1}")

    @override
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
        if self._train_progress_bar is not None and trainer.state.fn == "fit":
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    @override
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson

    Original source from openAI `gym`:
    https://github.com/openai/gym/blob/master/gym/utils/colorize.py
    """

    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return "\x1b[%sm%s\x1b[0m" % (attrs, string)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)
