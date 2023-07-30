import torch
from torch.nn import Module, Embedding, Dropout, Linear, MultiheadAttention, LayerNorm

from pykt.models.utils import get_clones, pos_encode, FFN, ut_mask


class SAKT(Module):
    def __init__(self, num_c, seq_len, emb_size,dropout, num_attn_heads,num_edr=2,emb_type="qid", emb_path=""):
        super().__init__()
        self.emb_type = emb_type
        self.dropout = dropout
        self.num_c = num_c
        self.emb_size = emb_size
        self.seq_len = seq_len

        self.num_ecd = self.num_edr
        self.dropout = dropout
        self.num_attn_heads = num_attn_heads

        if emb_type.startswith("qid"):
            interaction_emb = Embedding(num_c*2, emb_size) # 习题-回答
            exercise_emb = Embedding(num_c, emb_size)   # 练习
        positional_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout),num_edr)
        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    def base_emb(self, q, r, qry):
        x = q * self.num_c +r  #
        exemb = self.exercise_emb(qry)
        xemb = self.interaction_emb(x)
        posemb = self.positional_emb(pos_encode(xemb.shape[1])) # 位置编码单独计算（len(col),emb_size）
        memb = xemb + posemb
        return exemb, memb
        # 位置编码需要单独计算
    def forward(self, q, r , qry, qtest=False):
        emb_type = self.emb_type  # 局部变量，只在定义它的方法或函数中有效,函数外self
        exemb, memb = None, None
        if emb_type.startswith =="qid":
            exemb, memb = self.base_emb(q,r,qry)

        for i in range(self.num_c):
            memb = self.blocks[i](self,exemb, memb, memb) # q, k, v

        # 输出
        memb = self.dropout_layer(memb) # 对memb应用dropout层
        pred = self.pred(memb) # 使用一个线性层对dropout后的结果进行预测
        p = torch.sigmoid(pred)
        # 最后，我们使用squeeze方法以消除维度大小为1的维度。在这里，squeeze(-1)表示我们要消除最后一个维度（如果它的大小是1）。
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            return p, memb

class Blocks:
    def __init__(self, emb_size, num_attn_head,dropout)-> None:
        super.__init__()

        self.attn_multi = MultiheadAttention(emb_size, num_attn_head, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2) # 维度置换（序列长度，批次大小，嵌入维度）
        # attn -> drop -> skip -> norm
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len=k.shape[0])  # 使用因果mask，即只允许在计算attention时当前位置的词看到它之前的词
        attn_emb, _ = self.attn_multi(q, k, v, attn_mask=causal_mask)  # 调用多头注意力模块进行计算，得到注意力后的向量表示

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)  # 注意力向量与原始输入向量进行相加

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb