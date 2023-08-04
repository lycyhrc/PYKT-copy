import torch
from torch.nn import Module, Embedding, LSTM, GRU, Linear, ReLU, Sequential, Dropout, Sigmoid
import torch.nn.functional as F

class KQN(Module):
    def __init__(self,num_c,rnn_type,n_hidden,n_rnn_hidden,n_mlp_hidden,n_rnn_layers,dropout,emb_type="qid",emb_path=""):

        super(KQN,self).__init__()
        self.model_name = "kqn"
        self.num_c = num_c
        self.n_hidden = n_hidden
        self.n_rnn_hidden = n_rnn_hidden
        self.n_mlp_hidden = n_mlp_hidden
        self.n_rnn_layers = n_rnn_layers
        self.emb_type = emb_type
        self.rnn_type = rnn_type.lower()

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.n_hidden)

        if rnn_type == 'lstm':
            self.rnn_layers = LSTM(n_hidden,n_rnn_hidden,num_layers=n_rnn_layers,batch_first=True)
        elif rnn_type == 'gru':
            self.rnn_layers = GRU(n_hidden, n_rnn_hidden,num_layers=n_rnn_layers, batch_first=True)

        self.linear_layer = Linear(n_rnn_hidden,n_hidden)

        self.skill_encoder = Sequential(
            Linear(num_c, n_mlp_hidden),
            ReLU(),
            Linear(n_mlp_hidden, n_hidden),
            ReLU()
        )
        self.drop_layer = Dropout(dropout)
        self.sigmoid = Sigmoid()

    def forward(self, q, r, qshft, qtest=False):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            h, _ = self.rnn_layers(xemb)
            encoded_knowledge = self.linear_layer(h)
        encoded_knowledge = self.drop_layer(encoded_knowledge)  # 在训练过程中使用dropout，而在评估和测试阶段我们不使用，PyTorch的nn.Dropout通过,train/.eval方法会自动处理这个问题

        qshft_onehot = F.one_hot(qshft, num_classes=self.num_c)
        encoded_skills = self.skill_encoder(qshft_onehot.float())
        encoded_skills = F.normalize(encoded_skills, p=2, dim=2)

        logits = torch.sum(encoded_knowledge * encoded_skills, dim=2)
        logits = self.sigmoid(logits)
        if not qtest:
            return logits
        else:
            return logits, encoded_knowledge, encoded_skills
