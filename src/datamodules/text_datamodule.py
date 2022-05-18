import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset  # , SequentialSampler, RandomSampler

from src.datamodules.components.text_dataset import TextDataset
from transformers import RobertaTokenizer


class TextDataModule(LightningDataModule):
    """Example of LightningDataModule for text dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        tokenizer_name: str,
        block_size: int,
        data_dir: str = "data/roberta",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: Optional[int] = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        if self.hparams.num_classes is not None:
            return self.hparams.num_classes
        else:
            return 2

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        tokenizer = RobertaTokenizer.from_pretrained(self.hparams.tokenizer_name)

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = TextDataset(tokenizer, os.path.join(self.hparams.data_dir, 'train.jsonl'), self.hparams.block_size)
            self.data_val = TextDataset(tokenizer, os.path.join(self.hparams.data_dir, 'valid.jsonl'), self.hparams.block_size)
            self.data_test = TextDataset(tokenizer, os.path.join(self.hparams.data_dir, 'test.jsonl'), self.hparams.block_size)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            # sampler=RandomSampler(self.data_train),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            # sampler=SequentialSampler(self.data_val),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            # sampler=SequentialSampler(self.data_test),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
