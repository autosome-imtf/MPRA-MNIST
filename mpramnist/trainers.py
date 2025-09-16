import torch
import torch.nn as nn
import lightning.pytorch as L
import numpy as np

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
    roc_auc_score,
    auc,
    roc_curve,
)
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle


class LitModel(L.LightningModule):
    def __init__(self, model, loss, print_each, weight_decay=1e-2, lr=3e-4):
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
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        self.log("val_pearson", val_pearson, prog_bar=True)
        self.log("train_pearson", train_pearson)

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            res_str += f"| Val Pearson: {val_pearson:.5f} "

            res_str += f"| Train Pearson: {train_pearson:.5f} "
            border = "-" * len(res_str)
            print("\n".join(["", border, res_str, border, ""]))

        self.train_pearson.reset()
        self.val_pearson.reset()

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        self.log("test_pearson", test_pearson, prog_bar=True)
        self.test_pearson.reset()

    def predict_step(self, batch, _):
        x, y = batch
        pred = self.forward(x)

        return {
            "predicted": pred.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            cycle_momentum=False,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "cycle_lr",
        }

        return [self.optimizer], [lr_scheduler_config]


class LitModel_DeepPromoter(LitModel): ...


class LitModel_Agarwal(LitModel): ...


class LitModel_AgarwalJoint(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        num_outputs=3,
        activity_columns=["HepG2", "K562", "WTC11"],
        model=None,
        loss=nn.MSELoss(),
        print_each=1,
    ):
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.num_outputs = num_outputs
        self.activity_columns = activity_columns

        self.train_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=num_outputs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.forward(X)

        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(-1)  # [1076] -> [1076, 1]
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        self.train_pearson.update(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(-1)  # [1076] -> [1076, 1]
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )

        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        val_str = ""
        train_str = ""

        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        # num_outputs = 1
        if self.num_outputs == 1:
            name_train_metric = "train_" + self.activity_columns[0] + "_pearson"
            name_val_metric = "val_" + self.activity_columns[0] + "_pearson"

            self.log(
                name_train_metric,
                train_pearson,
                prog_bar=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                name_val_metric, val_pearson, prog_bar=True, on_epoch=True, logger=True
            )

            val_str += f"| Val Pearson {self.activity_columns[0]}: {val_pearson:.5f} "
            train_str += (
                f"| Train Pearson {self.activity_columns[0]}: {train_pearson:.5f} "
            )

            mean_val_pearson = val_pearson
            mean_train_pearson = train_pearson
        else:
            # num_outputs > 1
            for i in range(self.num_outputs):
                name_train_metric = "train_" + self.activity_columns[i] + "_pearson"
                name_val_metric = "val_" + self.activity_columns[i] + "_pearson"

                self.log(
                    name_train_metric,
                    train_pearson[i],
                    prog_bar=False,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    name_val_metric,
                    val_pearson[i],
                    prog_bar=True,
                    on_epoch=True,
                    logger=True,
                )

                val_str += (
                    f"| Val Pearson {self.activity_columns[i]}: {val_pearson[i]:.5f} "
                )
                train_str += f"| Train Pearson {self.activity_columns[i]}: {train_pearson[i]:.5f} "

            mean_val_pearson = val_pearson.mean()
            mean_train_pearson = train_pearson.mean()

        self.log(
            "val_pearson", mean_val_pearson, prog_bar=True, on_epoch=True, logger=True
        )

        self.train_pearson.reset()
        self.val_pearson.reset()

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "

            if self.num_outputs > 1:
                val_str += f"| Mean Val Pearson: {mean_val_pearson:.5f} "
                train_str += f"| Mean Train Pearson: {mean_train_pearson:.5f} "

            border = "-" * max(len(res_str), len(val_str), len(train_str))
            print(
                "\n".join(
                    ["", border, res_str, val_str + "|", train_str + "|", border, ""]
                )
            )

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)

        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(-1)  # [1076] -> [1076, 1]
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)
        self.log(
            "test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )

        self.test_pearson.update(y_hat, y)

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()

        if self.num_outputs == 1:
            # num_outputs = 1
            name_of_metric = "test_" + self.activity_columns[0] + "_pearson"
            self.log(name_of_metric, test_pearson, prog_bar=True)
        else:
            # num_outputs > 1
            for i in range(self.num_outputs):
                name_of_metric = "test_" + self.activity_columns[i] + "_pearson"
                self.log(name_of_metric, test_pearson[i], prog_bar=True)

        self.test_pearson.reset()

    def predict_step(self, batch, _):
        x, y = batch
        pred = self.forward(x)

        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # [1076] -> [1076, 1]
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # [1076] -> [1076, 1]

        return {
            "predicted": pred.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }


class LitModel_DeepStarr(LitModel_AgarwalJoint):
    def __init__(
        self,
        weight_decay,
        lr,
        num_outputs=3,
        activity_columns=["Developmental", "HouseKeeping"],
        model=None,
        loss=nn.MSELoss(),
        print_each=1,
    ):
        super().__init__(
            model=model,
            loss=loss,
            num_outputs=num_outputs,
            activity_columns=activity_columns,
            print_each=print_each,
            weight_decay=weight_decay,
            lr=lr,
        )


class LitModel_Sharpr(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        activity_columns,
        num_outputs=12,
        model=None,
        loss=nn.MSELoss(),
        print_each=1,
    ):
        super().__init__(model, loss, print_each, weight_decay, lr)
        self.activity_columns = activity_columns
        self.num_outputs = num_outputs
        self.train_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=num_outputs)

    def on_validation_epoch_end(self):
        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        self.log("val_pearson", val_pearson.mean(), prog_bar=True)
        self.log("train_pearson", train_pearson.mean())

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            res_str += "| Val Pearson: "
            for i in range(self.num_outputs):
                res_str += f"{self.activity_columns[i]} : {val_pearson[i]}, "
                self.log(f"{self.activity_columns[i]}", val_pearson[i])
            res_str += f"| Train Pearson: {train_pearson.mean():.5f} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))

        self.train_pearson.reset()
        self.val_pearson.reset()

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson.compute()
        for i in range(self.num_outputs):
            self.log(
                f"test_pearson_{self.activity_columns[i]}",
                test_pearson[i],
                prog_bar=True,
            )
        self.test_pearson.reset()


class LitModel_Dream(LitModel):
    def __init__(self, weight_decay, lr, model=None, loss=nn.MSELoss(), print_each=1):
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
            "target": labels.cpu().detach().float(),
        }

        if alt_pred is not None:
            result["alt_predicted"] = alt_pred.cpu().detach().float()

        return result


class LitModel_Kircher(LitModel_Dream): ...


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

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.test_pearson.update(y_hat, labels)


class LitModel_Evfratov(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        n_classes,
        show_figure=True,
        model=None,
        loss=nn.CrossEntropyLoss(),
        print_each=1,
    ):
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.val_aupr = AveragePrecision(task="multiclass", num_classes=n_classes)
        self.val_precision = Precision(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_aupr = AveragePrecision(task="multiclass", num_classes=n_classes)
        self.test_precision = Precision(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )

        # for plotting
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
            res_str += f"| Val AUROC: {self.val_auroc.compute()} "
            res_str += f"| Val AUPR: {self.val_aupr.compute()} |"
            res_str += f"\n| Val Precision: {self.val_precision.compute()} "
            res_str += f"| Val Recall: {self.val_recall.compute()} "
            res_str += f"| Val F1: {self.val_f1.compute()} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))

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
        res_str += f"| Test AUROC: {self.test_auroc.compute()} "
        res_str += f"| Test AUPR: {self.test_aupr.compute()} |"
        res_str += f"\n| Test Precision: {self.test_precision.compute()} "
        res_str += f"| Test Recall: {self.test_recall.compute()} "
        res_str += f"| Test F1: {self.test_f1.compute()} "
        border = "-" * 100
        print("\n".join(["", border, res_str, border, ""]))

        if self.show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.calculate_auroc(
            self.y_score, self.y_true, self.n_classes, ax1 if self.show_figure else None
        )
        self.plot_hist(
            self.y_score, self.y_true, self.n_classes, ax2 if self.show_figure else None
        )

        if self.show_figure:
            plt.tight_layout()
            plt.show()

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def calculate_auroc(self, y_score, y_true, n_classes, ax=None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if ax is not None:
            colors = cycle(
                ["orange", "green", "red", "purple", "blue", "yellow", "cyan", "brown"]
            )

            # Plot ROC curves for each class
            for i, color in zip(range(n_classes), colors):
                ax.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=1,
                    label=f"Class {i} (AUC = {roc_auc[i]:0.2f})",
                )

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("multi-class ROC Curves")
            ax.legend(loc="lower right")

    def plot_hist(self, y_score, y_true, n_classes, ax=None):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        y_true = y_true.cpu().numpy()

        # Plot histogram if axis is provided
        if ax is not None:
            counts = np.bincount(y_pred, minlength=n_classes)
            ax.bar(np.arange(n_classes), counts, color="skyblue", edgecolor="black")

            for i, count in enumerate(counts):
                ax.text(
                    i,
                    count,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_xlabel("Class Label")
            ax.set_ylabel("Count")
            ax.set_title("Predicted Class Distribution")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return {
            "y": y.squeeze().long().cpu().detach().float(),
            "pred": y_hat.cpu().detach().float(),
        }


class LitModel_Fluorescence_Reg(LitModel_AgarwalJoint): ...


class LitModel_Fluorescence_Clas(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        n_labels=3,
        show_figure=True,
        model=None,
        loss=nn.BCEWithLogitsLoss(),
        print_each=1,
    ):
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.val_acc = Accuracy(task="multilabel", num_labels=n_labels)
        self.val_auroc = AUROC(task="multilabel", num_labels=n_labels)
        self.val_aupr = AveragePrecision(task="multilabel", num_labels=n_labels)
        self.val_precision = Precision(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.val_recall = Recall(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.val_f1 = F1Score(task="multilabel", num_labels=n_labels, average="macro")

        self.test_acc = Accuracy(task="multilabel", num_labels=n_labels)
        self.test_auroc = AUROC(task="multilabel", num_labels=n_labels)
        self.test_aupr = AveragePrecision(task="multilabel", num_labels=n_labels)
        self.test_precision = Precision(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.test_recall = Recall(
            task="multilabel", num_labels=n_labels, average="macro"
        )
        self.test_f1 = F1Score(task="multilabel", num_labels=n_labels, average="macro")

        # for plotting
        self.n_labels = n_labels
        self.show_figure = show_figure
        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)
        self.y_true = self.y_true.to(self.device)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)
        y = y.float()

        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        y_metrics = y.long()

        self.val_acc(y_hat, y_metrics)
        self.val_auroc(y_hat, y_metrics)
        self.val_aupr(y_hat, y_metrics)
        self.val_precision(y_hat, y_metrics)
        self.val_recall(y_hat, y_metrics)
        self.val_f1(y_hat, y_metrics)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {self.val_acc.compute()} "
            res_str += f"| Val AUROC: {self.val_auroc.compute()} "
            res_str += f"| Val AUPR: {self.val_aupr.compute()} |"
            res_str += f"\n| Val Precision: {self.val_precision.compute()} "
            res_str += f"| Val Recall: {self.val_recall.compute()} "
            res_str += f"| Val F1: {self.val_f1.compute()} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.float()

        loss = self.loss(y_hat, y)

        y_metrics = y.long()

        self.test_acc(y_hat, y_metrics)
        self.test_auroc(y_hat, y_metrics)
        self.test_aupr(y_hat, y_metrics)
        self.test_precision(y_hat, y_metrics)
        self.test_recall(y_hat, y_metrics)
        self.test_f1(y_hat, y_metrics)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        # for plotting
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        res_str = f"| Test Acc: {self.test_acc.compute()} "
        res_str += f"| Test AUROC: {self.test_auroc.compute()} "
        res_str += f"| Test AUPR: {self.test_aupr.compute()} |"
        res_str += f"\n| Test Precision: {self.test_precision.compute()} "
        res_str += f"| Test Recall: {self.test_recall.compute()} "
        res_str += f"| Test F1: {self.test_f1.compute()} "
        border = "-" * 100
        print("\n".join(["", border, res_str, border, ""]))

        if self.show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.calculate_auroc(
            self.y_score, self.y_true, self.n_labels, ax1 if self.show_figure else None
        )
        self.plot_hist(
            self.y_score, self.y_true, self.n_labels, ax2 if self.show_figure else None
        )

        if self.show_figure:
            plt.tight_layout()
            plt.show()

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def calculate_auroc(self, y_score, y_true, n_labels, ax=None):
        y_score = torch.sigmoid(y_score.float()).cpu().numpy()
        y_true = y_true.cpu().numpy()

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and AUC for each label
        for i in range(n_labels):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if ax is not None:
            colors = cycle(
                ["orange", "green", "red", "purple", "blue", "yellow", "cyan", "brown"]
            )

            # Plot ROC curves for each label
            for i, color in zip(range(n_labels), colors):
                ax.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=1,
                    label=f"Label {i} (AUC = {roc_auc[i]:0.2f})",
                )

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves for each label")
            ax.legend(loc="lower right")

    def plot_hist(self, y_score, y_true, n_labels, ax=None):
        y_score = torch.sigmoid(y_score.float()).cpu().numpy()
        y_pred = (y_score > 0.5).astype(int)
        y_true = y_true.cpu().numpy()

        # Plot histogram if axis is provided
        if ax is not None:
            pos_counts = np.sum(y_pred, axis=0)

            ax.bar(np.arange(n_labels), pos_counts, color="skyblue", edgecolor="black")

            for i, count in enumerate(pos_counts):
                ax.text(
                    i,
                    count,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_xlabel("Label")
            ax.set_ylabel("Positive predictions count")
            ax.set_title("Positive predictions per label")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return {"y": y.squeeze().float().cpu().detach(), "pred": y_hat.cpu().detach()}


def shannon_entropy(x):
    p_c = nn.Softmax(dim=1)(x)
    return torch.sum(-p_c * torch.log(p_c), axis=1)


def _get_ranks(x):
    tmp = x.argsort(dim=0)
    ranks = torch.zeros_like(tmp)
    if len(x.shape) > 1:
        dims = x.shape[1]
        for dim in range(dims):
            ranks[tmp[:, dim], dim] = torch.arange(
                x.shape[0], layout=x.layout, device=x.device
            )
    else:
        ranks[tmp] = torch.arange(x.shape[0], layout=x.layout, device=x.device)
    return ranks


def spearman_correlation(x, y):
    x_rank = _get_ranks(x).float()
    y_rank = _get_ranks(y).float()
    vx = x_rank - torch.mean(x_rank, dim=0)
    vy = y_rank - torch.mean(y_rank, dim=0)
    pearsons = torch.sum(vx * vy, dim=0) / (
        torch.sqrt(torch.sum(vx**2, dim=0)) * torch.sqrt(torch.sum(vy**2, dim=0))
        + 1e-10
    )
    return pearsons, torch.mean(pearsons)


class LitModel_Malinois(LitModel_AgarwalJoint):
    def __init__(
        self,
        weight_decay,
        lr,
        num_outputs=3,
        activity_columns=["K562", "HepG2", "SKNSH"],
        model=None,
        loss=nn.MSELoss(),
        print_each=1,
        use_one_cycle=False,
    ):
        super().__init__(
            model=model,
            loss=loss,
            print_each=print_each,
            weight_decay=weight_decay,
            lr=lr,
            activity_columns=activity_columns,
            num_outputs=num_outputs,
        )

        self.use_one_cycle = use_one_cycle

        self.validation_step_outputs = []

    def categorical_mse(self, x, y):
        return (x - y).pow(2).mean(dim=0)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(-1)  # [1076] -> [1076, 1]
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # [1076] -> [1076, 1]

        loss = self.loss(y_hat, y)
        metric = self.categorical_mse(y_hat, y)

        self.validation_step_outputs.append(
            {"loss": loss, "metric": metric, "preds": y_hat, "labels": y}
        )

        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        self.val_pearson.update(y_hat, y)

    def on_validation_epoch_end(self):
        val_str = ""
        train_str = ""

        train_pearson = self.train_pearson.compute()
        val_pearson = self.val_pearson.compute()

        # num_outputs = 1
        if self.num_outputs == 1:
            name_train_metric = "train_" + self.activity_columns[0] + "_pearson"
            name_val_metric = "val_" + self.activity_columns[0] + "_pearson"

            self.log(
                name_train_metric,
                train_pearson,
                prog_bar=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                name_val_metric, val_pearson, prog_bar=True, on_epoch=True, logger=True
            )

            val_str += f"| Val Pearson {self.activity_columns[0]}: {val_pearson:.5f} "
            train_str += (
                f"| Train Pearson {self.activity_columns[0]}: {train_pearson:.5f} "
            )

            mean_val_pearson = val_pearson
            mean_train_pearson = train_pearson
        else:
            # num_outputs > 1
            for i in range(self.num_outputs):
                name_train_metric = "train_" + self.activity_columns[i] + "_pearson"
                name_val_metric = "val_" + self.activity_columns[i] + "_pearson"

                self.log(
                    name_train_metric,
                    train_pearson[i],
                    prog_bar=False,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    name_val_metric,
                    val_pearson[i],
                    prog_bar=True,
                    on_epoch=True,
                    logger=True,
                )

                val_str += (
                    f"| Val Pearson {self.activity_columns[i]}: {val_pearson[i]:.5f} "
                )
                train_str += f"| Train Pearson {self.activity_columns[i]}: {train_pearson[i]:.5f} "

            mean_val_pearson = val_pearson.mean()
            mean_train_pearson = train_pearson.mean()

        self.log(
            "val_pearson", mean_val_pearson, prog_bar=True, on_epoch=True, logger=True
        )

        harm_mean = (
            torch.stack(
                [batch["metric"] for batch in self.validation_step_outputs], dim=0
            )
            .mean(dim=0)
            .pow(-1)
            .mean()
            .pow(-1)
        )

        epoch_preds = torch.cat(
            [batch["preds"] for batch in self.validation_step_outputs], dim=0
        )
        epoch_labels = torch.cat(
            [batch["labels"] for batch in self.validation_step_outputs], dim=0
        )

        spearman, mean_spearman = spearman_correlation(epoch_preds, epoch_labels)
        shannon_pred, shannon_label = (
            shannon_entropy(epoch_preds),
            shannon_entropy(epoch_labels),
        )
        specificity_spearman, specificity_mean_spearman = spearman_correlation(
            shannon_pred, shannon_label
        )

        self.log(
            "enthropy_spearman",
            specificity_mean_spearman.item(),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "prediction_mean_spearman",
            mean_spearman.item(),
            prog_bar=False,
            on_epoch=True,
            logger=True,
        )

        self.validation_step_outputs.clear()
        self.train_pearson.reset()
        self.val_pearson.reset()

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Loss: {self.trainer.callback_metrics['val_loss']:.5f} "
            res_str += f"| Harm Mean Loss: {harm_mean:.5f} "
            res_str += f"| Enthropy Spearman: {specificity_mean_spearman.item():.5f} |"
            if self.num_outputs > 1:
                val_str += f"| Mean Val Pearson: {mean_val_pearson:.5f} "
                train_str += f"| Mean Train Pearson: {mean_train_pearson:.5f} "

            border = "-" * max(len(res_str), len(val_str), len(train_str))
            print(
                "\n".join(
                    ["", border, res_str, val_str + "|", train_str + "|", border, ""]
                )
            )

    def configure_optimizers(self):
        if self.use_one_cycle:
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                three_phase=False,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                cycle_momentum=False,
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cycle_lr",
            }
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                betas=(0.8661062881299633, 0.879223105336538),
                eps=1e-08,
                weight_decay=self.weight_decay,
                lr=self.lr,
                amsgrad=True,
            )

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer, T_0=4096, T_mult=1, eta_min=0.0, last_epoch=-1
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "learning_rate",
            }
        return [self.optimizer], [lr_scheduler_config]


class LitModel_Sure_Clas(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        activity_columns=["K562", "HepG2"],
        n_classes=10,
        model=None,
        loss=nn.CrossEntropyLoss(),
        print_each=1,
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
        y_hat = self.model(X.permute(0, 2, 1))
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:, 0])
        loss += self.loss(y_hat[:, 5:10], y[:, 1])

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.permute(0, 2, 1))
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:, 0])
        loss += self.loss(y_hat[:, 5:10], y[:, 1])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.print_each == 0:
            print("\n| {}: {:.5f} |\n".format("Current_epoch", self.current_epoch))
            self.shared_test_val_epoch_end()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

        return None

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x.permute(0, 2, 1))
        y = y.long()

        loss = self.loss(y_hat[:, 0:5], y[:, 0])
        loss += self.loss(y_hat[:, 5:10], y[:, 1])
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        self.shared_test_val_epoch_end(show_figure=True)

        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def shared_test_val_epoch_end(self, show_figure=False):
        border = "-" * 100
        plt_index = 0

        if show_figure:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("ROC Curves Comparison", fontsize=14)
        else:
            fig, ax1, ax2 = None, None, None

        for i in range(len(self.activity_columns)):
            auroc = self.calculate_auroc(
                y_score=self.y_score[:, i * 5 : i * 5 + 5],
                y_true=self.y_true[:, i],
                n_classes=self.n_classes // 2,
                show_figure=show_figure,
                name=self.activity_columns[i],
                ax=ax1 if plt_index == 0 else ax2 if show_figure else None,
            )
            precision, recall, accuracy, f1, aupr = self.calculate_aupr(
                y_score=self.y_score[:, i * 5 : i * 5 + 5],
                y_true=self.y_true[:, i],
                n_classes=self.n_classes // 2,
            )

            class_str = f"| {self.activity_columns[i]}: |"
            class_str += f"| Precision: {precision:.5f} |"
            class_str += f" Recall: {recall:.5f} |"
            class_str += f" Accuracy: {accuracy:.5f} |"
            class_str += f" F1: {f1:.5f} |"
            class_str += f" Val_AUCROC: {auroc:.5f} |"
            class_str += f" Val_AUPR: {aupr:.5f} |"

            print("\n".join(["", border, class_str, border, ""]))

            if show_figure and plt_index == 1:
                plt.tight_layout()
                plt.show()
                plt.close(fig)
                plt_index = 0
            elif show_figure:
                plt_index += 1

    def calculate_auroc(
        self, y_score, y_true, n_classes, name, show_figure=False, ax=None
    ):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if show_figure and ax is not None:
            colors = cycle(["orange", "green", "red", "purple", "blue"])

            # Plot ROC curves for each class
            for i, color in zip(range(n_classes), colors):
                ax.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=1,
                    label=f"Class {i} (AUC = {roc_auc[i]:0.2f})",
                )

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{name} multi-class ROC Curves")
            ax.legend(loc="lower right")

        return roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")

    def calculate_aupr(self, y_score, y_true, n_classes):
        y_score = F.softmax(y_score.float(), dim=1).cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        y_true = y_true.cpu().numpy()

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        pr_auc = average_precision_score(y_true_bin, y_score, average="macro")
        return precision, recall, accuracy, f1, pr_auc


class LitModel_Sure_Reg(LitModel):
    def __init__(
        self,
        weight_decay,
        lr,
        num_outputs=2,
        activity_columns=["K562", "HepG2"],
        model=None,
        loss=nn.MSELoss(),
        print_each=1,
    ):
        super().__init__(model, loss, print_each, weight_decay, lr)

        self.activity_columns = activity_columns

        self.train_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=num_outputs)

        self.y_score = torch.tensor([])
        self.y_true = torch.tensor([])

    def setup(self, stage=None):
        self.y_score = self.y_score.to(self.device)
        self.y_true = self.y_true.to(self.device)

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X.permute(0, 2, 1))
        loss = self.loss(y_hat, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        self.train_pearson.update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.permute(0, 2, 1))
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
                res_str += (
                    f"| Val Pearson {self.activity_columns[i]}: {val_pearson[i]:.5f} "
                )

            res_str += f"| Train Pearson: {train_pearson.mean():.5f} "
            border = "-" * len(res_str)
            print("\n".join(["", border, res_str, border, ""]))

        self.train_pearson.reset()
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x.permute(0, 2, 1))
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.y_score = torch.cat([self.y_score, y_hat])
        self.y_true = torch.cat([self.y_true, y])

    def on_test_epoch_end(self):
        test_pearson = self.test_pearson(self.y_score, self.y_true)
        for i in range(len(self.activity_columns)):
            self.log(
                f"test_pearson {self.activity_columns[i]}",
                test_pearson[i],
                prog_bar=True,
            )
        self.y_score = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def predict_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x.permute(0, 2, 1))

        return {
            "predicted": y_hat.cpu().detach().float(),
            "target": y.cpu().detach().float(),
        }


class LitModel_StarrSeq(LitModel):
    def __init__(
        self, weight_decay, lr, model=None, loss=nn.BCEWithLogitsLoss(), print_each=1
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

    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_aupr(y_hat, y.long())
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_auroc = self.val_auroc.compute()
        val_aupr = self.val_aupr.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1.compute()

        self.log("val_aupr", val_aupr, on_epoch=True, prog_bar=True)
        self.log("val_auroc", val_auroc, on_epoch=True, prog_bar=True)

        if (self.current_epoch + 1) % self.print_each == 0:
            res_str = f"| Epoch: {self.current_epoch} "
            res_str += f"| Val Acc: {val_acc} "
            res_str += f"| Val AUROC: {val_auroc} "
            res_str += f"| Val AUPR: {val_aupr} |"
            res_str += f"\n| Val Precision: {val_prec} "
            res_str += f"| Val Recall: {val_rec} "
            res_str += f"| Val F1: {val_f1} "
            border = "-" * 100
            print("\n".join(["", border, res_str, border, ""]))

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)

        loss = self.loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_aupr(y_hat, y.long())
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        test_aupr = self.test_aupr.compute()
        test_auroc = self.test_auroc.compute()

        self.log("test_aupr", test_aupr, on_epoch=True, prog_bar=True)
        self.log("test_auroc", test_auroc, on_epoch=True, prog_bar=True)

        res_str = f"| Test Acc: {self.test_acc.compute()} "
        res_str += f"| Test AUROC: {test_auroc} "
        res_str += f"| Test AUPR: {test_aupr} |"
        res_str += f"\n| Test Precision: {self.test_precision.compute()} "
        res_str += f"| Test Recall: {self.test_recall.compute()} "
        res_str += f"| Test F1: {self.test_f1.compute()} "
        border = "-" * 100
        print("\n".join(["", border, res_str, border, ""]))

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_aupr.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)

        return {
            "predicted": y_hat.squeeze().cpu().detach(),
            "target": y.cpu().detach().float(),
        }
