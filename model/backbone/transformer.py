# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from ..config import TransformerConfig


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head=12, d_ff=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, batch_data):
        # batch_size, 1 + num_patches, d_model
        residual = batch_data
        batch_data = self.norm1(batch_data)
        batch_data = self.msa(batch_data, batch_data, batch_data)[0]
        batch_data = self.dropout(batch_data)
        batch_data = residual + batch_data

        residual = batch_data
        batch_data = self.norm2(batch_data)
        batch_data = self.activation(self.linear1(batch_data))
        batch_data = self.dropout1(batch_data)
        batch_data = self.linear2(batch_data)
        batch_data = self.dropout2(batch_data)
        batch_data = residual + batch_data
        return batch_data


class ViT(nn.Module):
    def __init__(self, seq_len, patch_size, acc_axis, gyr_axis, d_model, num_layers=12, n_head=12, d_ff=2048,
                 dropout=0.1, max_num_patches=224, pooling=False):
        super(ViT, self).__init__()
        assert seq_len % patch_size == 0
        self.acc_axis = acc_axis
        self.gyr_axis = gyr_axis
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.pooling = pooling

        self.cls_embedding = nn.Parameter(torch.empty((1, 1, d_model)), requires_grad=True)
        self.embedding_acc = nn.Linear(self.acc_axis * self.patch_size, d_model, bias=False)
        self.embedding_gyr = nn.Linear(self.gyr_axis * self.patch_size, d_model, bias=False)

        self.position_embedding = nn.Parameter(torch.empty((1, max_num_patches, d_model)), requires_grad=True)

        self.encoders = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_head, self.d_ff, self.dropout) for _ in
              range(num_layers)]
        )

        nn.init.xavier_normal_(self.embedding_acc.weight)
        nn.init.xavier_normal_(self.embedding_gyr.weight)

        self.norm = nn.LayerNorm(self.d_model)

    def _pickup_patching(self, batch_data):
        # batch_size, n_channels, seq_len
        batch_size, n_channels, seq_len = batch_data.size()
        batch_data = batch_data.view(batch_size, n_channels, seq_len // self.patch_size, self.patch_size)
        batch_data = batch_data.permute(0, 2, 1, 3)
        batch_data = batch_data.contiguous().view(batch_size, seq_len // self.patch_size, n_channels * self.patch_size)
        return batch_data

    def forward(self, accData, gyrData):
        accData, gyrData = self._pickup_patching(accData), self._pickup_patching(gyrData)
        accData, gyrData = self.embedding_acc(accData), self.embedding_gyr(gyrData)
        batch_data = accData + gyrData
        batch_size, num_patches, _ = batch_data.size()

        # 拼接CLS向量
        batch_data = torch.cat((self.cls_embedding.repeat(batch_size, 1, 1), batch_data), dim=1)
        # 加上Position Embedding
        batch_data += self.position_embedding.repeat(batch_size, 1, 1)[:, :1 + num_patches, :]

        batch_data = self.encoders(batch_data)
        batch_data = self.norm(batch_data)
        if self.pooling:
            batch_data = torch.mean(batch_data, dim=1)
        else:
            batch_data = batch_data[:, 0, :]
        return batch_data

    def get_output_size(self):
        return self.d_model


def vit(model_name, config: TransformerConfig):
    # vit_s_16
    attributes = model_name.split('_')
    scales = attributes[1]
    config.patch_size = int(attributes[2])
    if scales == 'es':
        # Extra Small
        config.d_model = 128
        config.num_layers = 2
        config.n_head = 4
    elif scales == 'ms':
        # Medium Small
        config.d_model = 256
        config.num_layers = 4
        config.n_head = 4
    elif scales == 's':
        # Small
        config.d_model = 512
        config.num_layers = 8
        config.n_head = 8
    elif scales == 'b':
        # Base
        config.d_model = 768
        config.num_layers = 12
        config.n_head = 12
    elif scales == "l":
        # Large
        config.d_model = 1024
        config.num_layers = 24
        config.n_head = 16
    return ViT(seq_len=config.seq_len, patch_size=config.patch_size, acc_axis=config.acc_axis, gyr_axis=config.gyr_axis,
               d_model=config.d_model, num_layers=config.num_layers, n_head=config.n_head,
               d_ff=config.d_model * config.expansion_factor, dropout=config.dropout, pooling=config.pooling)
