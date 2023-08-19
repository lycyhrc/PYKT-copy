import os
import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)   # (BS,sl) when batch_first=True
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def forward(self, q, r):

        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r   # [BS, sl-1] 问题ID和回答结果的信息结合在一起,在一个统一的嵌入空间中表示问题和回答的交互
            xemb = self.interaction_emb(x)    # [BS, sl-1, es]
        
        h, _ = self.lstm_layer(xemb)  # [BS, sl-1, hs]
        h = self.dropout_layer(h)  
        y = self.out_layer(h)   # [BS, sl-1, num_c]
        y = torch.sigmoid(y)  

        return y # [BS, sl-1, num_c]