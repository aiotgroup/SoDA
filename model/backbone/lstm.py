# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from ..config import LSTMConfig


class LSTM1D(nn.Module):
    """
    LSTM不对数据切块，通过模型观察数据global情况
    """

    def __init__(self, seq_len, acc_axis, gyr_axis, hidden_dim, num_layers,
                 bidirectional=True, batch_first=True, dropout=0.1):
        super(LSTM1D, self).__init__()
        self.seq_len = seq_len
        self.acc_axis = acc_axis
        self.gyr_axis = gyr_axis
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.core = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)

        self.embedding_acc = nn.Linear(self.acc_axis, hidden_dim, bias=False)
        self.embedding_gyr = nn.Linear(self.gyr_axis, hidden_dim, bias=False)

        nn.init.xavier_normal_(self.embedding_acc.weight)
        nn.init.xavier_normal_(self.embedding_gyr.weight)

    def forward(self, accData, gyrData):
        # batch_size, n_channels, seq_len -> batch_size, seq_len, n_channels
        accData, gyrData = accData.permute(0, 2, 1), gyrData.permute(0, 2, 1)
        accData, gyrData = self.embedding_acc(accData), self.embedding_gyr(gyrData)
        batch_data = accData + gyrData

        batch_data = self.core(batch_data)[0]
        batch_data = torch.mean(batch_data, dim=1)
        return batch_data

    def get_output_size(self):
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim


def lstm(model_name: str, config: LSTMConfig):
    # lstm_s
    attributes = model_name.split('_')
    scales = attributes[1]
    if scales == 'es':
        # Extra Small
        config.hidden_dim = 128
        config.num_layers = 1
    elif scales == 'ms':
        # Medium Small
        config.hidden_dim = 256
        config.num_layers = 2
    elif scales == 's':
        # Small
        config.hidden_dim = 512
        config.num_layers = 3
    elif scales == 'b':
        # Base
        config.hidden_dim = 768
        config.num_layers = 4
    return LSTM1D(seq_len=config.seq_len, acc_axis=config.acc_axis, gyr_axis=config.gyr_axis,
                  hidden_dim=config.hidden_dim, num_layers=config.num_layers, bidirectional=config.bidirectional,
                  batch_first=config.batch_first, dropout=config.dropout)
