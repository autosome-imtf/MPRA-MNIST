import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import DataLoader

from training_config import TrainingConfig

from dataset import VikramDataset, VikramTestDataset
import transforms as t

class SeqDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_fold:int,
                 val_fold: int,
                 test_fold: int,
                 cfg: TrainingConfig,
                 cell_type: str,
                 train_transform = None,
                 test_transform = None):
        super().__init__()
        self.cfg = cfg
        self.cell_type = cell_type
        self.train_ds = VikramDataset(cell_type = self.cell_type, split = train_fold, transform = train_transform)
        self.val_ds = VikramDataset(cell_type = self.cell_type, split = val_fold, transform = test_transform) 
        self.test_ds = VikramDataset(cell_type = self.cell_type, split = test_fold, transform = test_transform) 
        self.test_transform = test_transform
        self.test_fold = test_fold
        
    def train_dataloader(self):
        
        return DataLoader(self.train_ds, 
                          batch_size=self.cfg.train_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=True) 
    
    def val_dataloader(self):

        return DataLoader(self.val_ds, 
                          batch_size=self.cfg.valid_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False)
        
    def dls_for_predictions(self):
        
        test_dl =  DataLoader(self.test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False)
        yield "forw_pred", test_dl
        if self.cfg.reverse_augment:
            
            self.test_transform.transforms.append(t.ReverseTest())
            self.test_ds = VikramDataset(cell_type = self.cell_type, split = self.test_fold, transform = self.test_transform) 
            
            rev_test_dl =  DataLoader(self.test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False)
            yield "rev_pred", rev_test_dl