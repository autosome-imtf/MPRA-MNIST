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
        
        return {"predicted": pred.cpu().detach().float(),"target": y.cpu().detach().float()}
    
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

class LitModel_AgarwalJoint(LitModel):
    def __init__(self, weight_decay, lr, 
                 num_outputs = 3,
                 model = None, loss = nn.MSELoss(), print_each = 1):
        
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.train_pearson = PearsonCorrCoef(num_outputs = num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs = num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs = num_outputs)

    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()
        
        self.log("val_pearson", val_pearson.mean(), prog_bar=True)
        self.log("train_pearson", train_pearson.mean())
        
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            res_str += f'| Val Pearson: {val_pearson.mean():.5f} '
    
            res_str += f'| Train Pearson: {train_pearson.mean():.5f} '
            border = '-'*len(res_str)
            print("\n".join(['', border, res_str, border, '']))
        
        self.train_pearson.reset()
        self.val_pearson.reset()

class LitModel_Dream(LitModel):
    def __init__(self, weight_decay, lr, 
                 model = None, loss = nn.MSELoss(), print_each = 1):
        
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
        
class LitModel_Kircher(LitModel_Dream):
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
        
class LitModel_Vaishnav(LitModel_Dream):
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        seqs, labels = batch

        if isinstance(seqs, dict):
            seq_x = seqs.get("seq")
            seq_alt_x = seqs.get("seq_alt")
            
            ref_pred = self.model(seq_x) 
            alt_pred = self.model(seq_alt_x) 
        else:
            ref_pred = self.model(seqs)
            alt_pred = None
            
        y_hat = ref_pred
        
        if alt_pred is not None:
            y_hat = alt_pred - ref_pred
        
        loss = self.loss(y_hat, labels)
        
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)
        self.test_pearson.update(y_hat, labels)

class LitModel_StarrSeq(L.LightningModule):
    
    def __init__(self, model, loss, lr=3e-4, weight_decay = 0.01, print_each = 1):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.print_each = print_each
        self.weight_decay = weight_decay
        self.val_loss = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X).squeeze()
        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True,  on_step=True, on_epoch=True, logger = True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.append(loss)
        
    def on_validation_epoch_end(self):

        val_loss = torch.stack(self.val_loss, dim = 0).mean()
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = '|' + ' {}: {:.5f} |'.format("current_epoch", self.current_epoch) 
            res_str += ' {}: {:.5f} |'.format("val_loss", val_loss)
            border = '-'*len(res_str)
            print("\n".join(['',border, res_str, border,'']))
        self.val_loss.clear()
        return None
        
    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss(y_hat, y)
        
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, tuple) or isinstance(batch, list):
            x, _ = batch
        else:
            x = batch
        return self(x)

    def configure_optimizers(self):
        
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                               lr=self.lr,
                                               weight_decay = self.weight_decay)
        
        return self.optimizer
        
class LitModel_StarrSeqBinary(L.LightningModule):
    def __init__(
        self,
        model,
        loss,
        batch_size,
        seq_len,
        lr=3e-4,
        weight_decay=0.01,
        print_each=1,
        out_bs=32,
        dropout=0.3
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_each = print_each
        self.batch_size = batch_size
        
        # Initialize last block
        final_feature_size = seq_len
        self.last_block = nn.Sequential(
            nn.Linear(model.block_sizes[-1] * final_feature_size * 2, out_bs),
            nn.BatchNorm1d(out_bs),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_bs, 1)
        )
        
        # For tracking validation loss
        self.val_loss = []

    def forward(self, x):
        return self.model(x)

    def _process_batch(self, batch):
        seqs, labels = batch
        enhancer = self(seqs["seq_enh"])
        promoter = self(seqs["seq"])
        concat = torch.cat([enhancer, promoter], dim=1)
        out = self.last_block(concat).squeeze()
        return out, labels

    def training_step(self, batch, batch_nb):
        out, labels = self._process_batch(batch)
        loss = self.loss(out, labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        out, labels = self._process_batch(batch)
        loss = self.loss(out, labels)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        self.val_loss.append(loss)

    def on_validation_epoch_end(self):
        if not self.val_loss:
            return
            
        val_loss = torch.stack(self.val_loss).mean()
        
        if (self.current_epoch + 1) % self.print_each == 0:
            border = "-" * 50
            res_str = (
                f"| current_epoch: {self.current_epoch} | "
                f"val_loss: {val_loss:.5f} |"
            )
            print(f"\n{border}\n{res_str}\n{border}\n")
            
        self.val_loss.clear()

    def test_step(self, batch, batch_idx):
        out, labels = self._process_batch(batch)
        loss = self.loss(out, labels)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out, _ = self._process_batch(batch)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )