import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import torch.utils.data as data

from torchmetrics import MeanSquaredError
from torchmetrics import PearsonCorrCoef

from mpramnist.models import HumanLegNet
from mpramnist.models import initialize_weights


class LitModel(L.LightningModule):
    def __init__(self, model, loss, print_each, weight_decay = 1e-2, lr = 3e-4):
        super().__init__()
        
        self.model = model
        
        self.loss = loss
        self.print_each = print_each
        self.weight_decay = weight_decay
        
        self.lr = lr
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True,  on_step=False, on_epoch=True, logger = True)
        self.train_pearson.update(y_hat, y)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()
        
        self.log("val_pearson", val_pearson, prog_bar=True)
        self.log("train_pearson", train_pearson)
        
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            res_str += f'| Val Pearson: {val_pearson:.5f} '
    
            res_str += f'| Train Loss: {self.trainer.callback_metrics['train_loss']:.5f} '
            res_str += f'| Train Pearson: {train_pearson:.5f} '
            border = '-'*len(res_str)
            print("\n".join(['', border, res_str, border, '']))
        
        self.train_pearson.reset()
        self.val_pearson.reset()
        
    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)
        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        self.log('test_pearson', test_pearson, prog_bar=True)
        self.test_pearson.reset()
        
    def predict_step(self, batch, _):
        x, y = batch 
        pred = self.model(x)
        
        return {"pred": y_pred.cpu().detach().float(),"y": labels.cpu().detach().float()}
    
    def configure_optimizers(self):
        
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                               lr=self.lr,
                                               weight_decay = self.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                        max_lr=self.lr,
                                                        three_phase=False, 
                                                        total_steps=self.trainer.estimated_stepping_batches, # type: ignore
                                                        pct_start=0.3,
                                                        cycle_momentum =False)
        lr_scheduler_config = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "cycle_lr"
            }
            
        return [self.optimizer], [lr_scheduler_config]
        
class LitModel_Kircher(LitModel):
    def __init__(self, weight_decay, lr, 
                 model = None, in_ch = 4, out_ch = 1, loss = nn.MSELoss(), print_each = 1):
        super().__init__(model, loss, print_each, weight_decay, lr)
        
        if model is None:
            self.model = HumanLegNet(
                in_ch=in_ch,
                output_dim=out_ch,
                stem_ch=64,
                stem_ks=11,
                ef_ks=9,
                ef_block_sizes=[80, 96, 112, 128],
                pool_sizes=[2, 2, 2, 2],
                resize_factor=4
            )
            self.model.apply(initialize_weights)
        else:
            self.model = model

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        seqs, labels = batch
        
        if isinstance(seqs, dict):
            seq_x = seqs.get("seq")
            seq_alt_x = seqs.get("seq_alt")
            
            ref_pred = self.model(seq_x) 
            
            alt_pred = self.model(seq_alt_x) 
        else:
            ref_pred = self.model(seqs)
            alt_pred = None
    
        result = {
            "ref_predicted": ref_pred.cpu().detach().float(),
            "target": labels.cpu().detach().float()
        }
        
        if alt_pred is not None:
            result["alt_predicted"] = alt_pred.cpu().detach().float()
    
        return result

class LitModel_Dream(LitModel):
    def __init__(self, weight_decay, lr, 
                 model = None, in_ch = 4, out_ch = 1, loss = nn.MSELoss(), print_each = 1):
        
        super().__init__(model, loss, print_each, weight_decay, lr)
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seqs, labels = batch
        
        if isinstance(seqs, dict):
            seq_x = seqs.get("seq")
            seq_alt_x = seqs.get("seq_alt")
            
            ref_pred = self.model(seq_x) 
            
            alt_pred = self.model(seq_alt_x) 
        else:
            ref_pred = self.model(seqs)
            alt_pred = None
    
        result = {
            "ref_predicted": ref_pred.cpu().detach().float(),
            "target": labels.cpu().detach().float()
        }
        
        if alt_pred is not None:
            result["alt_predicted"] = alt_pred.cpu().detach().float()
    
        return result