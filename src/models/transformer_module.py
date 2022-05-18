from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification)


class RoBERTaClassifier(LightningModule):
    """Example of LightningModule for text classification using Transformers (RoBERTa).

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
        https://huggingface.co/docs/transformers/model_doc/roberta
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lr: float = 0.0001,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        max_steps: int = 10000  # len(train_dataloader) * num_train_epochs
    ):
        super().__init__()

        self.config = RobertaConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.encoder = RobertaForSequenceClassification.from_pretrained(model_name,config=self.config)
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.max_steps = max_steps

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, input_ids: torch.Tensor):
        logits=self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        # prob=torch.softmax(logits,-1)
        return logits

    def step(self, batch: Any):
        input_ids, labels = batch
        logits = self.forward(input_ids)
        loss = self.criterion(logits, labels)  # NOTE: No need to apply softmax beforehand
        preds = torch.argmax(torch.softmax(logits,-1), dim=1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon)
        # max_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.max_steps*0.1,
                                                    num_training_steps=self.max_steps)
        return [optimizer], [scheduler]
