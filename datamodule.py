import lightning.pytorch as pl
import pandas as pd
import transforms as t
from torch.utils.data import DataLoader, random_split
from dataset import TrainSeqDatasetProb #, TestSeqDatasetProb

class SeqDataModule(pl.LightningDataModule):
    def __init__(self, 
                 num_workers,
                 train_batch_size,
                 valid_batch_size,
                 data_path: pd.DataFrame):
        super().__init__()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        df = pd.read_csv(data_path, 
                 sep='\t')
        df.columns = ['seq_id', 'seq', 'mean_value', 'fold_num', 'rev'][0:len(df.columns)]
        if "rev" in df.columns:
            df = df[df.rev == 0] # delete all lines, where rev = 1
        df = df.drop(columns=['fold_num', 'rev'])
        
        df_dataset = TrainSeqDatasetProb(df, transform = t.Compose([t.Seq2Tensor(),t.UseReverse()]))
        
        self.train, self.valid, self.test = random_split(df_dataset,(0.6,0.2,0.2))

    def train_dataloader(self):
        
        return DataLoader(self.train, 
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle = True) 
    
    def val_dataloader(self):

        return DataLoader(self.valid, 
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    def dls_for_predictions(self):
        
        return DataLoader(self.test,
                              batch_size=self.valid_batch_size,
                              num_workers=self.num_workers,
                              shuffle=False)