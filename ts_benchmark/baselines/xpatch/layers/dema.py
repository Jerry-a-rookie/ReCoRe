import torch
from torch import nn


class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    """

    def __init__(self, alpha, beta):
        super(DEMA, self).__init__()
        
        
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        
        
        alpha = self.alpha.to(x.device) if isinstance(self.alpha, torch.Tensor) else self.alpha
        beta = self.beta.to(x.device) if isinstance(self.beta, torch.Tensor) else self.beta
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = alpha * xt + (1 - alpha) * (s_prev + b)
            b = beta * (s - s_prev) + (1 - beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)
