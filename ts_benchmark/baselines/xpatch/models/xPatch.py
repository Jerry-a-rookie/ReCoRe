import torch
import torch.nn as nn
import math

from ..layers.decomp import DECOMP
from ..layers.network import Network



from ..layers.revin import RevIN


class xPatchModel(nn.Module):
    def __init__(self, configs):
        super(xPatchModel, self).__init__()

        
        seq_len = configs.seq_len  
        pred_len = configs.pred_len  
        c_in = configs.enc_in  

        
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        
        self.ma_type = configs.ma_type
        alpha = configs.alpha  
        beta = (
            configs.beta
        )  

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        
        

    def forward(self, x):
        

        
        if self.revin:
            x = self.revin_layer(x, "norm")

        if (
            self.ma_type == "reg"
        ):  
            x = self.net(x, x)
            
            
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        
        if self.revin:
            x = self.revin_layer(x, "denorm")

        return x
