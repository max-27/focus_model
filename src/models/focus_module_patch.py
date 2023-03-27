from typing import Any, List
import csv
import pickle

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics import MeanAbsoluteError

class FocusModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # loss function
        self.criterion = torch.nn.SmoothL1Loss()
        # metric objects for calculating and averaging mean absolute error across batches
        self.train_focus_error = MeanAbsoluteError()
        self.val_focus_error = MeanAbsoluteError()
        self.test_focus_error = MeanAbsoluteError()
        self.test_target_dict = {}
        self.test_prediction_dict = {}

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_focus_error_best = MinMetric()
        self.test_focus_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        #self.val_acc_best.reset()
        pass

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        loss, preds, targets = self.step((x, y))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.train_loss(loss)
        self.train_focus_error(preds, targets)
        return loss

    def training_epoch_end(self, outs: List[Any]) -> None:
        self.log("train_loss_epoch", self.train_loss.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.log("train_focus_error_epoch", self.train_focus_error.compute(), prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, _ = batch
        loss, preds, targets = self.step((x, y))

        # update and log metrics
        self.val_loss(loss)
        self.val_focus_error(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/focus_error", self.val_focus_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        focus_error = self.val_focus_error.compute()
        self.val_focus_error_best(focus_error)
        self.log("val/focus_error_best", self.val_focus_error_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        x, y, ids = batch
        patch_ids = [ids[0][i] + "_" + ids[1][i] for i in range(len(x))]
            
        loss, preds, targets = self.step((x, y))

        self.test_focus_error(preds, targets)

        for idx, id in enumerate(patch_ids):
            if id not in self.test_prediction_dict:
                self.test_prediction_dict[id] = torch.Tensor()
                self.test_target_dict[id] = targets[idx].to('cpu')
            self.test_prediction_dict[id] = torch.cat((self.test_prediction_dict[id], preds[idx].to('cpu')))
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/focus_error", self.test_focus_error, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        with open('/home/maf4031/focus_model/output/test_prediction.pkl', 'wb') as f:
            pickle.dump(self.test_prediction_dict, f)
        
        errors = []
        for sample_id in self.test_prediction_dict:
            self.test_prediction_dict[sample_id] = torch.mean(self.test_prediction_dict[sample_id])
            errors.append(abs(self.test_target_dict[sample_id]-self.test_prediction_dict[sample_id]))
        final_error = torch.mean(torch.Tensor(errors))
        final_std = torch.std(torch.Tensor(errors))
        self.log("test/focus_error", final_error, prog_bar=True)
        self.log("test/focus_std", final_std, prog_bar=True)
        self.log("test/focus_error_best", self.test_focus_err_best.compute(), prog_bar=True)

    def configure_optimizers(self):
        #optimizer = self.hparams.optimizer(params=self.parameters())
        optimizer = self.hparams.optimizer(filter(lambda p: p.requires_grad, self.parameters()))
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def compute_focus_error(self, preds, targets):
        return torch.abs(preds - targets)