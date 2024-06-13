import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])

        self.act = F.gelu
        self.dropout = nn.Dropout(p=configs.dropout)

        self.projection_classification = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)
        self.projection_regression = nn.Linear(configs.enc_in * configs.seq_len, 1)

    def forward(self, x_enc, x_mark_enc):
        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)

        output = self.act(x_enc)
        output = self.dropout(output)

        output = output * x_mark_enc.unsqueeze(-1)

        output = output.reshape(output.shape[0], -1)
        output_classification = self.projection_classification(output)
        output_regression = self.projection_regression(output)
        return output_classification, output_regression
