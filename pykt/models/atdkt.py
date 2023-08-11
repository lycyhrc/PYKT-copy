import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
    CrossEntropyLoss
from .utils import ut_mask

device = "cpu" if not torch.cuda.is_available() else "cuda"


class ATDKT(Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, dropout=0.1, emb_type='qid',
                 num_layers=1, num_attn_heads=5, l1=0.5, l2=0.5, l3=0.5, start=50, emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "atdkt"
        print(f"qnum: {num_q}, cnum: {num_c}")
        print(f"emb_type: {emb_type}")
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            Linear(self.hidden_size // 2, self.num_c))

        if self.emb_type.endswith("predhis"):
            self.l1 = l1
            self.l2 = l2
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size)  # add concept emb
            if self.emb_type.find("qemb") != -1:
                self.question_emb = Embedding(self.num_q, self.emb_size)

            self.start = start
            self.hisclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, 1))
            self.hisloss = nn.MSELoss()

        if self.emb_type.endswith("predcurc"):  # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.l3 = l3
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size)  # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                d_model = self.hidden_size  # * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            else:
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
                Linear(self.hidden_size // 2, self.num_c))
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size)  # add concept emb

            self.closs = CrossEntropyLoss()
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size // 2, 1))
                self.hisloss = nn.MSELoss()

    def predcurc(self, dcur, q, c, r, xemb, train):
        emb_type = self.emb_type
        y2, y3 = 0, 0
        if emb_type.find("delxemb") != -1: #default
            qemb = self.question_emb(q)  # BS*sl*es
            cemb = self.concept_emb(c)  # BS*sl*es
            catemb = qemb + cemb   # BS*sl*es
        else:
            catemb = xemb
            if self.num_q > 0:
                qemb = self.question_emb(q)
                catemb = qemb + xemb

            if emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c)
                catemb += cemb

        # cemb = self.concept_emb(c)
        # catemb = cemb
        if emb_type.find("trans") != -1:
            mask = ut_mask(seq_len=catemb.shape[1])
            qh = self.trans(catemb.transpose(0, 1), mask).transpose(0, 1)  # BS*sl*hs
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            sm = dcur["smasks"].long()
            start = 0
            cpreds = self.qclasifier(qh[:, start:, :])  # BS*sl*num_c
            flag = sm[:, start:] == 1  # 添加掩码，用于计算loss
            y2 = self.closs(cpreds[flag], c[:, start:][flag])  # CrossEntropyLoss

        # predict response
        xemb = xemb + qh + cemb  # BS*sl*hs
        if emb_type.find("qemb") != -1:  # no
            xemb = xemb + qemb
        h, _ = self.lstm_layer(xemb)   # BS*sl*hs

        # predict history correctness rates
        rpreds = None
        if train and emb_type.find("his") != -1:
            sm = dcur["smasks"].long()
            start = self.start
            rpreds = torch.sigmoid(self.hisclasifier(h)).squeeze(-1)  # BS&sl*1->BS&sl
            rsm = sm[:, start:]
            rflag = rsm == 1
            rtrues = dcur["historycorrs"][:, start:]
            y3 = self.hisloss(rpreds[:, start:][rflag], rtrues[rflag])
            # (掩码矩阵计算出有效位置，[]用于乘法)
        # predict response
        h = self.dropout_layer(h)
        y = self.out_layer(h)  # BS*Sl*num_c
        y = torch.sigmoid(y)
        return y, y2, y3

    def forward(self, dcur, train=False):  ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()

        y2, y3 = 0, 0

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x)   # response encoder BS*sl*es
        rpreds, qh = None, None
        if emb_type == "qid":  # 【model】w/o QT & IR
            h, _ = self.lstm_layer(xemb)  # BS*ls*hs
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))  # BS*ls*num_c
        elif emb_type.endswith("predhis"):  # only predict history correct ratios 训练时不使用
            # predict response
            if self.emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c)
                xemb = xemb + cemb
            if emb_type.find("qemb") != -1:
                qemb = self.question_emb(q)
                xemb = xemb + qemb
            h, _ = self.lstm_layer(xemb)
            # predict history correctness rates
            if train:
                sm = dcur["smasks"].long()
                start = self.start
                rpreds = torch.sigmoid(self.hisclasifier(h)[:, start:, :]).squeeze(-1)
                rsm = sm[:, start:]
                rflag = rsm == 1
                rtrues = dcur["historycorrs"][:, start:]
                y2 = self.hisloss(rpreds[rflag], rtrues[rflag])

            h = self.dropout_layer(h)
            y = self.out_layer(h)
            y = torch.sigmoid(y)
        elif emb_type.endswith("predcurc"):  # predict current question' current concept 【train use】
            y, y2, y3 = self.predcurc(dcur, q, c, r, xemb, train)

        if train:
            return y, y2, y3
        else:
            return y
