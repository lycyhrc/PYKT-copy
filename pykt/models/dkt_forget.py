import torch
from torch.nn import Module, Embedding, LSTM, Dropout, Linear

device = "cpu" if not torch.cuda.is_available() else "cuda"

class DKTForget(Module):
    def __init__(self, num_c, num_rgap, num_sgap, num_pcount, emb_size, dropout=0.1, emb_type='qid', emb_path=""):
        super().__init__()
        self.model_name = "dkt_forget"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, emb_size)
        n_total = num_rgap + num_sgap + num_pcount

        self.lstm_layer = LSTM(self.emb_size+n_total, self.hidden_size)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size+n_total, self.num_c)

    def forward(self, q, r, dgaps):
        # "rgaps"：表示问题之间的相对间隔;"sgaps"：表示学习会话之间的间隔;"pcounts"：表示问题的出现次数 (_shft是移位版本)
        q, r = q.to(device), r.to(device)
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            theta_in = self.c_integration(xemb, dgaps["rgaps"].to(device).long(), dgaps["sgaps"].to(device).long(), dgaps["pcounts"].to(device).long())

        h, _ = self.lstm_layer(theta_in)
        theta_out = self.c_integration(h, dgaps["shft_rgaps"].to(device).long(), dgaps["shft_sgaps"].to(device).long(), dgaps["shft_pcounts"].to(device).long())
        theta_out = self.dropout_layer(theta_out)
        y = self.out_layer(theta_out)
        y = torch.sigmoid(y)

        return y

class CIntegration(Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_dim) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        ntotal = num_rgap + num_sgap + num_pcount
        self.cemb = Linear(ntotal, emb_dim, bias=False)
        print(f"num_sgap: {num_sgap}, num_rgap: {num_rgap}, num_pcount: {num_pcount}, ntotal: {ntotal}")

    def forward(self, vt, rgap, sgap, pcount):
        rgap, sgap, pcount = self.rgap_eye[rgap].to(device), self.sgap_eye[sgap].to(device), self.pcount_eye[pcount].to(device)
        ct = torch.cat((rgap, sgap, pcount), -1)
        Cct = self.cemb(ct)
        theta = torch.mul(vt, Cct)
        theta = torch.cat((theta, ct), -1)
        return theta