import torch

import pandas as pd
import pytorch_lightning as pl

from datamodule import SeqDataModule
from pathlib import Path
import scipy
from scipy import stats
from training_config import TrainingConfig


def save_predict(trainer: pl.Trainer,
                 model: pl.LightningModule, 
                 data: SeqDataModule,
                 fold: int,
                 cfg: TrainingConfig,
                 to_mean_score = False,
                 data_path_averaged = "datasets/VikramDataset/HepG2_averaged.tsv"
):
    df_mean = pd.read_csv(data_path_averaged, 
                 sep='\t')
    df_mean =  df_mean[df_mean.fold == fold]
    df = pd.DataFrame() 
    if cfg.reverse_augment and to_mean_score:
        print("====================")
        print("без усреднения скора")
        print("====================")
        for pred_name, dl in data.dls_for_predictions():
            
            y_preds =  trainer.predict(model,
                                       dataloaders=dl)
            print("====================")
            print(pred_name, "score")
            print("====================")
            trainer.test(model,
                                       dataloaders=dl)
            y_preds = torch.concat(y_preds).cpu().numpy() #type: ignore
            df[pred_name] = y_preds
    else:
        for pred_name, dl in data.dls_for_predictions():
            trainer.test(model,
                                       dataloaders=dl)
    if cfg.reverse_augment and to_mean_score:
        print("====================")
        print("с усреднением скора")
        print("====================")
        print(stats.pearsonr(df_mean.mean_value,df.mean(axis = 1)))
    else:
        pass
    return None
