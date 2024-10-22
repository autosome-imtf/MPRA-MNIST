import torch

import pandas as pd
import pytorch_lightning as pl

from datamodule import SeqDataModule
from pathlib import Path
import scipy
from scipy import stats
from training_config import TrainingConfig
def get_name(full_name: str) -> str:
        return full_name.split("_", maxsplit=1)[0]

def save_predict(trainer: pl.Trainer,
                 model: pl.LightningModule, 
                 data: SeqDataModule,
                 fold: int,
                 cfg: TrainingConfig,
                 save_dir: Path,
                 to_mean_score = False,
                 data_path_averaged = "datasets/VikramDataset/HepG2_averaged.tsv",
                 pref: str = "",
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
        print("с усреднением скора по аугментациям")
        print("====================")
        df_mean['pred'] = ((df['forw_pred'] + df['rev_pred']) / 2).values
        print(stats.pearsonr(df_mean.mean_value,df_mean.pred))
        
        print("====================")
        print("с усреднением скора по цепям")
        print("====================")
        names = df_mean['seq_id'].apply(lambda x: get_name(x))
        sm = df_mean.groupby(names).mean(numeric_only=True)
        print(stats.pearsonr(sm['pred'], sm['mean_value']))
    else:
        pass
    if pref != "":
        df.to_csv(save_dir / f"predictions_{pref}.tsv", 
                  sep='\t', 
                  index=False)
    else:
        df.to_csv(save_dir / f"predictions.tsv", 
                  sep='\t', 
                  index=False)
    return None
    
    
