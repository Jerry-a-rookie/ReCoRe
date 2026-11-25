import torch
from torch import nn


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha):
        super(EMA, self).__init__()
        
        self.alpha = alpha

    
    def forward(self, x):
        
        
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double, device=x.device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)

    
    
    
    
    
    
    
    
    
    
