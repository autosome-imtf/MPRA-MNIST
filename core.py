import torch 
import pandas as pd
import numpy as np
import lightning.pytorch as pl


from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path 
from pytorch_lightning import loggers as pl_loggers
from torch import nn

train_batch_size = 1024
valid_batch_size = 1024
num_workers = 1

data_path = './datasets/From_Legnet_git_data/HepG2.tsv'

from datamodule import SeqDataModule
data = SeqDataModule(num_workers, train_batch_size, valid_batch_size, data_path)
train_dl = data.train_dataloader()
valid_dl = data.val_dataloader()


logger = pl_loggers.TensorBoardLogger("./logs", name = "cnn_test_like_mnist")

from trainer import LitModel
model = LitModel()
trainer = pl.Trainer(accelerator='gpu', 
                            devices = [1],
                            max_epochs=5,
                            logger = logger)

trainer.fit(model, datamodule=data)