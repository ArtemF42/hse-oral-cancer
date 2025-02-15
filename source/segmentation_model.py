from typing import Tuple

import torch
import torch.nn as nn

from lightning import LightningModule

from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index


class SegmentationModel(LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(dim=1)

    def training_step(self, batch: dict) -> torch.Tensor:
        images, true_masks = batch['image'], batch['mask']
        pred_masks = self(images)
        loss = self.loss_fn(pred_masks, true_masks)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict) -> None:
        images, true_masks = batch['image'], batch['mask']
        pred_masks = self(images)
        loss = self.loss_fn(pred_masks, true_masks)
        self.log('val_loss', loss)
        self._compute_metrics(pred_masks, true_masks)

    def test_step(self, batch: dict) -> None:
        images, true_masks = batch['image'], batch['mask']
        pred_masks = self(images)
        self._compute_metrics(pred_masks, true_masks, mode='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _compute_metrics(
        self,
        pred_masks: torch.Tensor,
        true_masks: torch.Tensor,
        mode: str = 'validation',
    ) -> None:
        for metric_name, metric in zip(
            ('jaccard', 'dice'), (binary_jaccard_index, dice)
        ):
            metric_name = f'{mode}_{metric_name}'
            self.log(metric_name, metric(pred_masks, true_masks.int()), on_epoch=True)
