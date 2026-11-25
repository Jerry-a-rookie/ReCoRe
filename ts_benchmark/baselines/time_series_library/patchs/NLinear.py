import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs, individual=False):
        super(NLinear, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if (
            self.task_name == "classification"
            or self.task_name == "anomaly_detection"
            or self.task_name == "imputation"
        ):
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        
        
        self.channels = configs.enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):
        
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  

    def forecast(self, x_enc):
        
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        
        return self.encoder(x_enc)

    def classification(self, x_enc):
        
        enc_out = self.encoder(x_enc)
        
        
        output = enc_out.reshape(enc_out.shape[0], -1)
        
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc)
            return dec_out  
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  
        if self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  
        return None
