import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import torch.utils.data as data
import numpy as np

from torchmetrics import MeanSquaredError
from torchmetrics import PearsonCorrCoef
from torchmetrics import (
    Accuracy,
    AUROC,
    AveragePrecision,
    Precision,
    Recall,
    F1Score,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score, auc, roc_curve
)
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

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

class LitModel_DeepPromoter(LitModel):
    ...
        
class LitModel_Agarwal(LitModel):
    ...
    
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
        
    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        self.log('test_pearson', test_pearson.mean(), prog_bar=True)
        self.test_pearson.reset()

class LitModel_DeepStarr(LitModel_AgarwalJoint):
    ...

class LitModel_Sharpr(LitModel):
    
    def __init__(self, weight_decay, lr, 
                 activity_columns,
                 num_outputs = 12,
                 model = None, loss = nn.MSELoss(), print_each = 1):
        
        super().__init__(model, loss, print_each, weight_decay, lr)
        self.activity_columns = activity_columns
        self.num_outputs = num_outputs
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
            res_str += f'| Val Pearson: '
            for i in range(self.num_outputs):
                res_str += f'{self.activity_columns[i]} : {val_pearson[i]}, '
            res_str += f'| Train Pearson: {train_pearson.mean():.5f} '
            border = '-'*100
            print("\n".join(['', border, res_str, border, '']))
        
        self.train_pearson.reset()
        self.val_pearson.reset()
        
    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        for i in range(self.num_outputs):
            self.log(f'test_pearson_{self.activity_columns[i]}', test_pearson[i], prog_bar=True)
        self.test_pearson.reset()

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

class LitModel_Evfratov(LitModel):

    def __init__(self, 
                 weight_decay, lr, 
                 n_classes,
                 show_figure = True,
                 model = None, loss = nn.CrossEntropyLoss(), print_each = 1):
        
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.val_aupr = AveragePrecision(task="multiclass", num_classes=n_classes)
        self.val_precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_aupr = AveragePrecision(task="multiclass", num_classes=n_classes)
        self.test_precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=n_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")

        #for plotting
        self.n_classes = n_classes
        self.show_figure = show_figure
        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)  
        self.y_true = self.y_true.to(self.device) 

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)
        y = y.long()

        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_aupr(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {self.val_acc.compute()} "
            res_str += f'| Val AUROC: {self.val_auroc.compute()} '
            res_str += f'| Val AUPR: {self.val_aupr.compute()} |'
            res_str += f'\n| Val Precision: {self.val_precision.compute()} '
            res_str += f'| Val Recall: {self.val_recall.compute()} '
            res_str += f'| Val F1: {self.val_f1.compute()} '
            border = '-'*100
            print("\n".join(['', border, res_str, border, '']))
        
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.long()

        loss = self.loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_aupr(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        # for plotting
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        
        res_str = f"| Test Acc: {self.test_acc.compute()} "
        res_str += f'| Test AUROC: {self.test_auroc.compute()} '
        res_str += f'| Test AUPR: {self.test_aupr.compute()} |'
        res_str += f'\n| Test Precision: {self.test_precision.compute()} '
        res_str += f'| Test Recall: {self.test_recall.compute()} '
        res_str += f'| Test F1: {self.test_f1.compute()} '
        border = '-'*100
        print("\n".join(['', border, res_str, border, '']))
        
        if self.show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.calculate_auroc(self.y_score, self.y_true, self.n_classes, ax1 if self.show_figure else None) 
        self.plot_hist(self.y_score, self.y_true, self.n_classes, ax2 if self.show_figure else None) 
        
        if self.show_figure:
            plt.tight_layout()
            plt.show()
            
        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.y_score = torch.tensor([], device = self.device)
        self.y_true = torch.tensor([], device = self.device)

    def calculate_auroc(self, y_score, y_true, n_classes, ax = None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        fpr, tpr, roc_auc = dict(), dict(), dict()
        
        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        if ax is not None:
            colors = cycle(['orange', 'green', 'red', 'purple', 'blue', 'yellow', 'cyan', 'brown'])
            
            # Plot ROC curves for each class
            for i, color in zip(range(n_classes), colors):
                ax.plot(
                    fpr[i], tpr[i], 
                    color=color, 
                    lw=1,
                    label=f'Class {i} (AUC = {roc_auc[i]:0.2f})'
                )
            
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('multi-class ROC Curves')
            ax.legend(loc="lower right")

    def plot_hist(self, y_score, y_true, n_classes, ax = None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        y_true = y_true.cpu().numpy()
        
        # Plot histogram if axis is provided
        if ax is not None:
            counts = np.bincount(y_pred, minlength=n_classes)
            ax.bar(np.arange(n_classes), counts, color='skyblue', edgecolor='black')
            
            for i, count in enumerate(counts):
                ax.text(i, count, str(count), ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Count')
            ax.set_title('Predicted Class Distribution')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return {"y": y.squeeze().long().cpu().detach().float(), "pred": y_hat.cpu().detach().float()}

class LitModel_Fluorescence_Reg(LitModel_AgarwalJoint):
    ...

class LitModel_Fluorescence_Clas(LitModel_Evfratov):
    ...

class LitModel_Malinois(LitModel_AgarwalJoint):
    ...

class LitModel_Sure_Clas(LitModel):
    def __init__(self, 
                 weight_decay, lr, 
                 activity_columns = ["K562", "HepG2"],
                 n_classes = 10,
                 model = None, loss = nn.CrossEntropyLoss(), print_each = 1
                ):
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.activity_columns = activity_columns
        self.loss = torch.nn.CrossEntropyLoss()
        self.print_each = print_each
        self.n_classes = n_classes
        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)  
        self.y_true = self.y_true.to(self.device)  
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X.permute(0,2,1))
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:,0])
        loss += self.loss(y_hat[:, 5:10], y[:,1])
        
        self.log("train_loss", loss, prog_bar=True,  on_step=True, on_epoch=True, logger = True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.permute(0,2,1).permute(0,2,1))
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:,0])
        loss += self.loss(y_hat[:, 5:10], y[:,1])
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])
        
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            print('| {}: {:.5f} |\n'.format("Current_epoch", self.current_epoch))
            self.shared_test_val_epoch_end()
        self.y_score = torch.tensor([], device = self.device)
        self.y_true = torch.tensor([], device = self.device)
        
        return None

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x.permute(0,2,1))
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:,0])
        loss += self.loss(y_hat[:, 5:10], y[:,1])
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])
        
    def on_test_epoch_end(self):
        self.shared_test_val_epoch_end(show_figure = True)
        
        self.y_score = torch.tensor([], device = self.device)
        self.y_true = torch.tensor([], device = self.device)

    def shared_test_val_epoch_end(self, show_figure = False):
        border = '-'*100
        
        if show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('ROC Curves Comparison', fontsize=14)
            plt_index = 0
            
        for i in range(len(self.activity_columns)):

            # i*5 : i*5 + 5 will create intervals like [0:5] and [5:10]
            auroc = self.calculate_auroc(y_score=self.y_score[:, i*5:i*5 + 5], y_true=self.y_true[:,i],
                                         n_classes=self.n_classes//2, 
                                         show_figure = show_figure, 
                                         name = self.activity_columns[i],
                                         ax=ax1 if plt_index == 0 else ax2 if show_figure else None)
            precision, recall, accuracy, f1, aupr = self.calculate_aupr(y_score=self.y_score[:, i*5:i*5 + 5], y_true=self.y_true[:,i], n_classes=self.n_classes//2) 
            
            class_str = f"| {self.activity_columns[i]}: |"
            class_str += '| {}: {:.5f} |'.format("Precision", precision)
            class_str += ' {}: {:.5f} |'.format("Recall", recall)
            class_str += ' {}: {:.5f} |'.format("Accuracy", accuracy)
            class_str += ' {}: {:.5f} |'.format("F1", f1)
            class_str += ' {}: {:.5f} |'.format("Val_AUCROC", auroc)
            class_str += ' {}: {:.5f} |'.format("Val_AUPR", aupr)
            
            print("\n".join(['', border, class_str, border, '']))
            
            if show_figure:
                plt_index += 1
                if plt_index == 2: 
                    plt.tight_layout()
                    plt.show()
                    plt_index = 0

    def calculate_auroc(self, y_score, y_true, n_classes, name, show_figure = False, ax = None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        fpr, tpr, roc_auc = dict(), dict(), dict()
        
        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        colors = cycle(['orange', 'green', 'red', 'purple', 'blue'])
        
        # Plot ROC curves for each class
        for i, color in zip(range(n_classes), colors):
            ax.plot(
                fpr[i], tpr[i], 
                color=color, 
                lw=1,
                label=f'Class {i} (AUC = {roc_auc[i]:0.2f})'
            )
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} multi-class ROC Curves')
        ax.legend(loc="lower right")

        return roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")

    def calculate_aupr(self, y_score, y_true, n_classes):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        y_true = y_true.cpu().numpy()
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division = 0)  
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        pr_auc = average_precision_score(y_true_bin, y_score, average="macro")
        return precision, recall, accuracy, f1, pr_auc
    
class LitModel_Sure_Reg(LitModel):

    def __init__(self, weight_decay, lr, 
                 num_outputs = 2,
                 activity_columns = ["K562", "HepG2"],
                 model = None, loss = nn.MSELoss(), print_each = 1):
        
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.activity_columns = activity_columns
        
        self.train_pearson = PearsonCorrCoef(num_outputs = num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs = num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs = num_outputs)

        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)  
        self.y_true = self.y_true.to(self.device)  

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X.permute(0,2,1))
        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True,  on_step=False, on_epoch=True, logger = True)
        self.train_pearson.update(y_hat, y)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.permute(0,2,1))
        loss = self.loss(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])
    
    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson(self.y_score, self.y_true)
        
        self.log("val_pearson", val_pearson.mean(), prog_bar=True)
        self.log("train_pearson", train_pearson.mean())
        
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            for i in range(len(self.activity_columns)):
                res_str += f'| Val Pearson {self.activity_columns[i]}: {val_pearson[i]:.5f} '
    
            res_str += f'| Train Pearson: {train_pearson.mean():.5f} '
            border = '-'*len(res_str)
            print("\n".join(['', border, res_str, border, '']))
        
        self.train_pearson.reset()
        self.y_score = torch.tensor([], device = self.device)
        self.y_true = torch.tensor([], device = self.device)

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x.permute(0,2,1))
        loss = self.loss(y_hat, y)
        
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])
        
    def on_test_epoch_end(self):
        test_pearson =  self.test_pearson(self.y_score, self.y_true)
        for i in range(len(self.activity_columns)):
            self.log(f'test_pearson {self.activity_columns[i]}', test_pearson[i], prog_bar=True)
        self.y_score = torch.tensor([], device = self.device)
        self.y_true = torch.tensor([], device = self.device)

class LitModel_StarrSeq(LitModel):
    
    def __init__(self, 
                 weight_decay, lr, 
                 out_bs=32,
                 dropout=0.3,
                 is_binary = False,
                 model = None, loss = nn.CrossEntropyLoss(), print_each = 1
                ):
        
        super().__init__(model, loss, print_each, weight_decay, lr)

        # Validation metrics for binary classification
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_aupr = AveragePrecision(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
    
        # Test metrics for binary classification
        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_aupr = AveragePrecision(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.is_binary = is_binary
        if self.is_binary or model.is_binary:
        # Initialize last block for binary task
            final_feature_size = model.seq_len
            
            self.last_block = nn.Sequential(
                nn.Linear(model.block_sizes[-1] * final_feature_size * 2, out_bs),
                nn.BatchNorm1d(out_bs),
                nn.SiLU(),
                nn.Linear(out_bs, out_bs),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(out_bs),
                nn.Linear(out_bs, 1)
            )

    def _process_batch(self, batch):
        seqs, labels = batch
        enhancer = self(seqs["seq_enh"])
        promoter = self(seqs["seq"])
        concat = torch.cat([enhancer, promoter], dim=1)
        out = self.last_block(concat).squeeze()
        return out, labels

    def training_step(self, batch, batch_nb):
        if self.is_binary or model.is_binary:
            y_hat, y = self._process_batch(batch)
        else:
            X, y = batch
            y_hat = self.model(X).squeeze()

        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_binary or model.is_binary:
            y_hat, y = self._process_batch(batch)
        else:
            X, y = batch
            y_hat = self.model(X).squeeze()
        loss = self.loss(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_aupr(y_hat, y.long())
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {self.val_acc.compute()} "
            res_str += f'| Val AUROC: {self.val_auroc.compute()} '
            res_str += f'| Val AUPR: {self.val_aupr.compute()} |'
            res_str += f'\n| Val Precision: {self.val_precision.compute()} '
            res_str += f'| Val Recall: {self.val_recall.compute()} '
            res_str += f'| Val F1: {self.val_f1.compute()} '
            border = '-'*100
            print("\n".join(['', border, res_str, border, '']))
        
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        if self.is_binary or model.is_binary:
            y_hat, y = self._process_batch(batch)
        else:
            X, y = batch
            y_hat = self.model(X).squeeze()

        loss = self.loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_aupr(y_hat, y.long())
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        
        res_str = f"| Test Acc: {self.test_acc.compute()} "
        res_str += f'| Test AUROC: {self.test_auroc.compute()} '
        res_str += f'| Test AUPR: {self.test_aupr.compute()} |'
        res_str += f'\n| Test Precision: {self.test_precision.compute()} '
        res_str += f'| Test Recall: {self.test_recall.compute()} '
        res_str += f'| Test F1: {self.test_f1.compute()} '
        border = '-'*100
        print("\n".join(['', border, res_str, border, '']))
            
        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.is_binary or model.is_binary:
            y_hat, y = self._process_batch(batch)
        else:
            x, y = batch
            y_hat = self.model(x)
        return {"predicted": y_hat.squeeze().long().cpu().detach().float(), "target" : y.cpu().detach().float()}