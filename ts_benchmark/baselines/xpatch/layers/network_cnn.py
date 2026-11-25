import torch
from torch import nn


class NetworkCNN(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, trend):
        super(NetworkCNN, self).__init__()

        
        self.pred_len = pred_len

        
        
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":  
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        
        self.conv1 = nn.Conv1d(
            self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        
        self.fc2 = nn.Linear(self.dim, patch_len)

        
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num // trend, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num // trend)

        
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num // trend * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, x):
        

        x = x.permute(0, 2, 1)  

        
        B = x.shape[0]  
        C = x.shape[1]  
        I = x.shape[2]  
        x = torch.reshape(x, (B * C, I))  

        
        
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        

        
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        res = x

        
        x = self.conv1(x)
        x = self.gelu2(x)
        x = self.bn2(x)

        
        res = self.fc2(res)
        x = x + res

        
        x = self.conv2(x)
        x = self.gelu3(x)
        x = self.bn3(x)

        
        x = self.flatten1(x)
        x = self.fc3(x)
        x = self.gelu4(x)
        x = self.fc4(x)

        
        x = torch.reshape(x, (B, C, self.pred_len))  

        x = x.permute(0, 2, 1)  

        return x
