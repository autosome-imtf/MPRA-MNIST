import math

import torch
import torch.nn as nn
import torch.nn.functional as F
    
"""def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)"""

class LinearNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.lin1 = nn.Linear(4*seq_len,1)
        self.S = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0],self.seq_len*4) # reshape to flatten sequence dimension
        x = self.lin1(x) # Linear wraps up the weights/bias dot product operations
        x = self.S(x)
        return x


class small_cnn(nn.Module):
    
    def __init__(self, seq_len = 230, block_sizes=[16, 24, 32, 40, 48], kernel_size=7):
        
        super().__init__()
        self.seq_len = seq_len
        self.loss = torch.nn.CrossEntropyLoss()
        nn_blocks = []
      
        for in_bs, out_bs in zip([4] + block_sizes, block_sizes):
            
            block = nn.Sequential(
                nn.Conv1d(in_bs, out_bs, kernel_size=kernel_size, padding=1),
                nn.Sigmoid(),
                nn.BatchNorm1d(out_bs)
            )
            nn_blocks.append(block)
            
        self.conv_net = nn.Sequential(
            *nn_blocks,
            nn.Flatten(),
            nn.Linear(block_sizes[-1] * (seq_len + len(block_sizes)*(3-kernel_size)), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # reshape view to batch_size x 4channel x seq_len
        # permute to put channel in correct order
        #x = x.permute(0,2,1) 
        out = self.conv_net(x)
        #print(out.shape)
        return out