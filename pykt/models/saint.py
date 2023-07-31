import torch
import torch.nn as nn
from torch.nn import Module, Dropout

from pykt.models.utils import get_clones, FFN, ut_mask, pos_encode

device = "cpu" if not torch.cuda.is_available() else "cuda"
class SAINT(Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1,emb_type="qid", emb_path=""):
        super().__init__()
        
        self.model_name = "saint"
        self.num_c = num_c
        self.num_q = num_q
        self.num_en = n_blocks
        self.num_de = n_blocks
        self.emb_type = emb_type
        
        self.emb_pos = nn.Embedding(seq_len,embedding_dim=emb_size)
        
        if emb_type.startswith("qid"):
            self.encoder = get_clones(Encoder_block(emb_size, num_attn_heads, num_q, num_c, seq_len, dropout),self.num_en)

        self.decoder = get_clones(Decoder_block(emb_size, 2, num_attn_heads, seq_len, dropout), self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)
    def forward(self, ex_in, cat_in, res_in, qtest=None):
        emb_type = self.emb_type

        # 生成(exercise、category)位置编码, 自然顺序
        if self.num_q > 0:
            in_pos = pos_encode(ex_in.shape[1])
        else:
            in_pos = pos_encode(cat_in.shape[1])
        in_pos = self.emb_pos(in_pos)

        # 通过encoder
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            if emb_type == "qid":
                ex_in = self.encoder[i](ex_in, cat_in, in_pos, first_block=first_block)
            cat_in = ex_in

        # 传递decoder
        start_token = torch.tensor([[2]]).repeat(res_in.shape[0], 1).to(device)
        res_in = torch.cat((start_token, res_in), dim=-1)
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            res_in = self.decoder[i](res_in,in_pos,ex_in,first_block=first_block)

        # 预测
        res = self.out(self.dropout(res_in))
        res = torch.sigmoid(res).squeeze(-1)
        if not qtest:
            return res
        else:
            return res, res_in

class Encoder_block(Module):
    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout, emb_path=""):
        super().__init__()
        self.seq_len = seq_len
        self.dim_model = dim_model
        self.total_ex = total_ex
        self.total_cat = total_cat
        self.emb_path = emb_path
        if total_ex > 0:
            if emb_path == "":
                self.emb_ex = nn.Embedding(total_ex,embedding_dim=dim_model)
            else:
                pass  # 从与训练模型读取
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim=dim_model)

        # multihead
        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        # ffn
        self.ffn_en = FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self,in_ex, in_cat, in_pos, first_block = True):

        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.emb_ex(in_ex)
                else:
                    pass
                    # in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            # print(out.shape)
            # print(in_pos.shape)
            out = out + in_pos
        else:
            out = in_ex

        # multihead
        out = out.permute(1, 0, 2)
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out

        out, atten_wt = self.multi_en(out,out,out,attn_mask=ut_mask(seq_len=n))
        out = self.dropout1(out)
        out = out + skip_out

        # ffn
        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out

        return out

class Decoder_block(Module):
    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len

        self.emb_res = nn.Embedding(total_res+1, embedding_dim=dim_model)
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model,num_heads=heads_de,dropout=dropout)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de, dropout=dropout)
        self.ffn_en = FFN(dim_model, dropout)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, in_res, in_pos, en_out, first_block=True):
        if first_block:
            in_resp = self.emb_res(in_res)
            out = in_resp + in_pos
        else:
            out = in_res

        # Multihead 1
        out = out.permute(1, 0, 2)
        n,_,_ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, atten_wt = self.multi_de1(out, out, out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout1(out)
        out = out + skip_out

        # Multi-head 2
        en_out = en_out.permute(1,0,2)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, atten_wt = self.multi_de2(out,en_out, en_out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout2(out)
        out = out + skip_out

        # ffn
        out = out.permute(1,0,2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout3(out)
        out = out + skip_out

        return out





