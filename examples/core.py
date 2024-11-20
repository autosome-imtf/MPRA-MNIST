from tqdm import tqdm

import mpramnist
from mpramnist.malinoisdataset import MalinoisDataset

import mpramnist.transforms as t
import mpramnist.target_transforms as t_t

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

left_flank = MalinoisDataset.LEFT_FLANK
right_flank = MalinoisDataset.RIGHT_FLANK

NUM_EPOCHS = 5
BATCH_SIZE = 1076
lr = 0.00326
NUM_WORKERS = 103
weight_d = 0.00034

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Basset(nn.Module):
    def __init__(self, output_dim = 280):
        super().__init__()

        self.linear1_channels=1000
        #self.linear2_channels=1000, 

        self.activation1 = nn.ReLU()
        self.activation = nn.ReLU()
        
        self.dropout1 = nn.Dropout(0.11625)
        #self.dropout2 = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.output_activation = nn.Sigmoid()

        # Layer 2 (convolutional), constituent parts
        self.conv1_filters = torch.nn.Parameter(torch.zeros(300, 4, 19))
        torch.nn.init.kaiming_uniform_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(300)
        self.maxpool1 = nn.MaxPool1d(3)

        # Layer 3 (convolutional), constituent parts
        self.conv2_filters = torch.nn.Parameter(torch.zeros(200, 300, 11))
        torch.nn.init.kaiming_uniform_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(200)
        self.maxpool2 = nn.MaxPool1d(4)

        # Layer 4 (convolutional), constituent parts
        self.conv3_filters = torch.nn.Parameter(torch.zeros(200, 200, 7))
        torch.nn.init.kaiming_uniform_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(200)
        self.maxpool3 = nn.MaxPool1d(2)

        # Layer 5 (fully connected), constituent parts
        self.fc4 = nn.LazyLinear(1000, bias=True)
        self.batchnorm4 = nn.BatchNorm1d(1000)
        
        # Layer 6 (fully connected), constituent parts
        #self.fc5 = nn.LazyLinear(1000, bias=True)
        #self.batchnorm5 = nn.BatchNorm1d(1000)

        # Output layer (fully connected), constituent parts
        #self.fc6 = nn.LazyLinear(output_dim, bias=True)

    def encode(self, x):
        # Layer 1
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)

        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)

        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)

        x = self.flatten(cnn)
        return x

    def decode(self, x):
        # Layer 4
        
        cnn = self.fc4(x)
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout1(cnn)
        '''
        # Layer 5
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        x = self.dropout2(cnn)
        '''
        return x
        
    def classify(self, x):
        
        #output = self.fc6(x)
        output = x
        return output

    def forward(self, x):
        '''
        # Output layer
        logits = self.fc6(cnn)
        y_pred = self.output_activation(logits)

        return y_pred
        '''
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output  = self.classify(decoded)
        return output

def shannon_entropy(x):
    p_c = nn.Softmax(dim=1)(x)    
    return torch.sum(- p_c * torch.log(p_c), axis=1)
def pearson_correlation(x, y):
    vx = x - torch.mean(x, dim=0)
    vy = y - torch.mean(y, dim=0)
    pearsons = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)) + 1e-10)
    return pearsons, torch.mean(pearsons)

from torchmetrics import PearsonCorrCoef
class MPRA_Basset(pl.LightningModule):
    
    def __init__(self,
                 output_dim = 3,
                 learning_rate=1e-4,
                 optimizer='Adam',
                 scheduler=False,
                 weight_decay=1e-6,
                 epochs=1,
                 extra_hidden_size = 100,
                 criterion = 'MSELoss',
                 last_activation='Tanh',
                 sneaky_factor=1,
                 **kwargs):
        
        super().__init__()

        self.output_dim = output_dim
        
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.extra_hidden_size = extra_hidden_size
        self.sneaky_factor = sneaky_factor
        
        #self.criterion = getattr(nn, criterion)()
        self.criterion = criterion
        self.last_activation = getattr(nn, last_activation)()
        
        self.basset_net = Basset()
        
        
        #self.basset_last_hidden_width = self.basset_net.linear2_channels

        self.output_1 = nn.Sequential(
            # 1 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            # 2 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            # 3 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            nn.Dropout(0.5757),
            # last layer
            nn.Linear(140, 1)
            )
        
        self.output_2 = nn.Sequential(
            # 1 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            # 2 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            # 3 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            nn.Dropout(0.5757),
            # last layer
            nn.Linear(140, 1)
            )
        
        self.output_3 = nn.Sequential(
            # 1 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            # 2 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            # 3 layer
            nn.LazyLinear(140, bias=False),
            nn.BatchNorm1d(140),
            self.last_activation,
            nn.Dropout(0.5757),
            # last layer
            nn.Linear(140, 1, bias = False)
            )       
        self.val_pearson = PearsonCorrCoef()
        self.example_input_array = torch.rand(1, 4, 600)
        
    def forward(self, x):
        basset_last_hidden = self.basset_net.decode(self.basset_net.encode(x))
        output_1 = self.output_1(basset_last_hidden)
        output_2 = self.output_2(basset_last_hidden)
        output_3 = self.output_3(basset_last_hidden)
        if self.output_dim == 2:
            output_1 = torch.cat((output_1, output_2), dim=1)
        elif self.output_dim == 3:
            output_1 = torch.cat((output_1, output_2, output_3), dim=1)
        return output_1
        
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.to(device).float())

        targets = targets.squeeze(1).to(device)
        
        shannon_pred, shannon_target = shannon_entropy(outputs).to(device), shannon_entropy(targets).to(device)
        loss = self.criterion(outputs, 
                              targets) + self.sneaky_factor*self.criterion(shannon_pred, shannon_target)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.to(device).float())

        targets = targets.squeeze(1).to(device)
        
        loss = self.criterion(outputs, 
                              targets)
        self.log('val_loss', loss, prog_bar=True)
        corr = self.val_pearson(outputs[:, 0], 
                              targets[:, 0])
        self.log("val_pearson", corr, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'pred': outputs, 'target': targets}
        
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.to(device).float())

        targets = targets.squeeze(1).to(device)
        
        loss = self.criterion(outputs, 
                              targets)
        self.log('test_loss', loss)
        corr = self.val_pearson(outputs[:, 0], 
                              targets[:, 0])
        self.log("test_pearson", 
                 corr ,
                 on_epoch=True,
                 prog_bar=True,
                 on_step=False,)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, tuple) or isinstance(batch, list):
            x, _ = batch
        else:
            x = batch
        return self(x)
        
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate,
                                                         weight_decay=self.weight_decay, betas=(0.8661, 0.8792), amsgrad=True)  
        if self.scheduler:
            lr_scheduler = {
                'scheduler' : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                                   T_0=4096, T_mult=1, eta_min=0.0),
                'name': 'learning_rate'
                           }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

class L1KLmixed(nn.Module):
    """
    A custom loss module that combines L1 loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the L1 loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.L1Loss): The L1 loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and KL divergence losses.

    Example:
        loss_fn = L1KLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)

# preprocessing
train_transform = t.Compose([
    t.AddFlanks(left_flank, right_flank),
    t.CenterCrop(600),
    #t.Reverse(0.5),
    t.Seq2Tensor()
])
val_test_transform = t.Compose([
    t.AddFlanks(left_flank, right_flank),
    t.CenterCrop(600),
    t.Seq2Tensor()
])

target_transform = t_t.Compose([
    t_t.Normalize(mean = 0.500, std = 1.059) # original for Malinois 
])
activity_columns = ['HepG2','SKNSH', "K562"]
# load the data
train_dataset = MalinoisDataset(activity_columns = activity_columns, 
                              split = "train", 
                              filtration = "original",
                                duplication_cutoff = 0.5,
                                use_reverse_complement = True,
                              transform = train_transform,
                               target_transform = target_transform) 

val_dataset = MalinoisDataset(activity_columns = activity_columns, 
                              split = "val", 
                              filtration = "original",
                              transform = val_test_transform,
                             target_transform = target_transform) 

test_dataset = MalinoisDataset(activity_columns = activity_columns, 
                              split = "test",
                              filtration = "original",
                              transform = val_test_transform,
                              target_transform = target_transform)

print(train_dataset)
print(val_dataset)
# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

seq_model = MPRA_Basset(output_dim = len(train_dataset[0][1]),
                       criterion = L1KLmixed(beta = 5.0),
                       last_activation='ReLU', learning_rate = lr,
                       weight_decay = weight_d)
'''
callback_topmodel = pl.callbacks.ModelCheckpoint(monitor='val_pearson',
                                                 save_top_k=1,
                                                 dirpath="./Malinois_reverse_Quality",
                                                 filename="max_pearson_exact_params_200_epochs")
callback_es = pl.callbacks.early_stopping.EarlyStopping(monitor='val_pearson', patience=10)

logger = pl_loggers.TensorBoardLogger("./logs", name = "Malinois_60_200_epochs")
'''
# Initialize a triner
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    min_epochs=60, max_epochs=200,
    gradient_clip_val=1,
    precision='16-mixed', 
    #callbacks=[
        #TQDMProgressBar(refresh_rate=1), 
        #callback_es, 
        #callback_topmodel],
    #logger = logger, 
    
)

# Train the model
trainer.fit(seq_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
trainer.test(seq_model, dataloaders=test_loader)