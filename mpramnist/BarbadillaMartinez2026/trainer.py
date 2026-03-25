import torch.nn as nn
from mpramnist.trainers import LitModel
from torchmetrics import PearsonCorrCoef

class LitModel_BarbadillaMartinez(LitModel):
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