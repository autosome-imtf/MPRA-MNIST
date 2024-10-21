import pytorch_lightning as pl
import torch
import torch.nn as nn


from model import initialize_weights
from torchmetrics import PearsonCorrCoef

from training_config import TrainingConfig

class LitModel(pl.LightningModule):
    def __init__(self, tr_cfg: TrainingConfig):
        super().__init__()
        
        self.tr_cfg = tr_cfg
        self.model = self.tr_cfg.get_model()
        self.model.apply(initialize_weights)
        self.loss = nn.MSELoss() 
        self.val_pearson = PearsonCorrCoef()
        
    def training_step(self, batch, _):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True,  on_step=False, on_epoch=True,  logger=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        #self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar = True)
        self.val_pearson(y_hat, y)
        self.log("val_pearson", self.val_pearson, on_epoch=True, logger=True, prog_bar = True)
    
    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True, logger=True)
        corr = self.val_pearson(y_hat, y)
        self.log("test_pearson", 
                 corr ,
                 on_epoch=True,
                 prog_bar=True,
                 on_step=False, logger=True)
        
    def predict_step(self, batch, _):
        if isinstance(batch, (tuple, list)):
            x, _ = batch 
        else:
            x = batch
        y_hat = self.model(x)
        return y_hat
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.tr_cfg.max_lr  / 25,
                                      weight_decay=self.tr_cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, # type: ignore
                                                        max_lr=self.tr_cfg.max_lr ,
                                                        three_phase=False, 
                                                        total_steps=self.trainer.estimated_stepping_batches, # type: ignore
                                                        pct_start=0.3,
                                                        cycle_momentum =False)
        
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "cycle_lr"
        }
        return [optimizer], [lr_scheduler_config]