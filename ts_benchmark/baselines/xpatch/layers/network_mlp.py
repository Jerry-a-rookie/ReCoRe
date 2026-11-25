import torch
from torch import nn


class NetworkMLP(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(NetworkMLP, self).__init__()

        
        self.pred_len = pred_len

        
        
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc3 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, x):
        

        x = x.permute(0, 2, 1)  

        
        B = x.shape[0]  
        C = x.shape[1]  
        I = x.shape[2]  
        x = torch.reshape(x, (B * C, I))  

        
        
        x = self.fc1(x)
        x = self.avgpool1(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.avgpool2(x)
        x = self.ln2(x)

        x = self.fc3(x)

        
        x = torch.reshape(x, (B, C, self.pred_len))  

        x = x.permute(0, 2, 1)  

        return x
