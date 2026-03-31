import torch.nn as nn
import torch.nn.functional as F

class MPRAnn(nn.Module):
    def __init__(self, output_dim, in_channels: int = 4, end_sigmoid: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=250, kernel_size=7, padding="valid")
        self.bn1 = nn.BatchNorm1d(250)
        self.conv2 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=8, padding="valid")
        self.bn2 = nn.BatchNorm1d(250)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=3, padding="valid")
        self.bn3 = nn.BatchNorm1d(250)
        self.conv4 = nn.Conv1d(in_channels=250, out_channels=100, kernel_size=2, padding="valid")
        self.bn4 = nn.BatchNorm1d(100)
        #self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1) 
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(100, 300)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(300, self.output_dim)
        self.end_sigmoid = end_sigmoid



    def forward(self, seq):
        #seq = seq.permute(0, 2, 1)
        seq = self.conv1(seq)
        seq = F.relu(seq)
        seq = self.bn1(seq)
        seq = self.conv2(seq)
        seq = F.softmax(seq, dim=1)
        seq = self.bn2(seq)
        seq = self.pool1(seq)
        seq = self.dropout1(seq)
        seq = self.conv3(seq)
        seq = F.softmax(seq, dim=1)
        seq = self.bn3(seq)
        seq = self.conv4(seq)
        seq = F.softmax(seq, dim=1)
        seq = self.bn4(seq)
        seq = self.global_pool(seq) 
        seq = seq.squeeze(-1)  
        seq = self.dropout2(seq)
        seq = seq.reshape((seq.shape[0], -1))
        seq = self.fc1(seq)
        seq = F.sigmoid(seq)
        seq = self.dropout3(seq)
        seq = self.fc2(seq)
        if self.end_sigmoid:
            seq = F.sigmoid(seq)
        seq = seq.squeeze(-1)

        return seq