import lightning.pytorch as pl
import torch
import torch.nn as nn

from torchmetrics import PearsonCorrCoef
from model import LinearNN, small_cnn


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = small_cnn(230)
        self.loss = nn.MSELoss() 
        self.val_pearson = PearsonCorrCoef()
        
    def training_step(self, batch, _):
        X, y = batch
        y_hat = self.model(X)
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True,  on_step=False, on_epoch=True,  logger=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_pearson(y_hat, y)
        self.log("val_pearson", self.val_pearson, on_epoch=True)


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=0.001,
                                      weight_decay=0.001)
        return optimizer


