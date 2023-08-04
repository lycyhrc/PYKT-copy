import torch
import torch.nn as nn
from torch import cat,squeeze,unsqueeze,sum
from torch.nn import Module
from torch.nn import Embedding, Linear, Sigmoid, Tanh, Dropout


class DIMKT(Module):
    def __init__(self, num_q, num_c, emb_size, dropout, emb_type, batch_size, num_steps, difficult_levels, emb_path=""):
        super(DIMKT, self).__init__()
        self.model_name = 'dimkt'
        self.emb_size = emb_size
        self.num_q = num_q
        self.num_c = num_c
        self.emb_type = emb_type

        # add train para
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.difficult_levels = difficult_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.dropout = Dropout(dropout)

        # if emb_type.startswith("qid"):
        #     self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.q_emb = Embedding(num_q + 1, emb_size, padding_idx=0)
        self.c_emb = Embedding(num_c + 1, emb_size, padding_idx=0)
        self.qd_emb = Embedding(difficult_levels + 2, emb_size, padding_idx=0)
        self.sd_emb = Embedding(difficult_levels + 2, emb_size, padding_idx=0)
        self.a_emb = Embedding(2, self.emb_size)

        # add special
        self.knowledge_state = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.emb_size)), requires_grad=True)

        self.input_layer = Linear(4 * emb_size, emb_size)
        self.linear2_layer = Linear(emb_size, emb_size)
        self.linear3_layer = Linear(emb_size, emb_size)
        self.linear4_layer = Linear(2 * emb_size, emb_size)  # W4/W5（d_sdf+d_a）*dk
        self.linear5_layer = Linear(2 * emb_size, emb_size)
        self.output_layer = Linear(4 * emb_size, emb_size)

    def forward(self,q,c,sd,qd,a, qshft,cshft,sdshft,qdshft):
        if self.batch_size != len(q):
            self.batch_size = len(q)

        q_emb = self.q_emb(q)
        c_emb = self.c_emb(c)
        sd_emb = self.sd_emb(sd)
        qd_emb = self.qd_emb(qd)
        a_emb = self.a_emb(a)

        target_q = self.q_emb(qshft)
        target_c = self.c_emb(cshft)
        target_sd = self.sd_emb(sdshft)
        target_qd = self.qd_emb(qdshft)

        # 输入
        input_data = cat([q_emb,c_emb,sd_emb,qd_emb],-1)  # batch_size, sequence_length, feature_dim
        input_data = self.input_layer(input_data)
        target_data = cat([target_q, target_c, target_sd, target_qd], -1)  # batch_size, sequence_length, feature_dim
        target_data = self.input_layer(target_data)

        # 在时序数据的每个序列的开始位置增加一个零向量作为填充（padding），
        # 然后将每个序列拆分为单独的时间步
        shape = list(input_data.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=self.device)
        input_data = cat((padd,input_data),1)
        slice_input_data = input_data.split(1, dim=1)

        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=self.device)
        sd_emb = cat((padd,sd_emb),1)
        slice_sd_embedding = sd_emb.split(1,dim=1)
        qd_emb = cat((padd, qd_emb),1)
        slice_qd_embedding = qd_emb.split(1,dim=1)

        shape = list(a_emb.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=self.device)
        a_emb = cat((padd,a_emb),1)
        slice_a_embedding = a_emb.split(1,dim=1)

        k = self.knowledge_state.repeat(self.batch_size,1).to(self.device)   # 行复制
        h = list()
        seqlen = q.size(1)
        for i in range(1, seqlen+1):
            input_data_i = squeeze(slice_input_data[i], 1)
            qd_i = squeeze(slice_qd_embedding[i],1)
            sd_i = squeeze(slice_sd_embedding[i],1)
            a_i = squeeze(slice_a_embedding[i],1)

            qq = input_data_i - k
            gate_SDF = self.linear2_layer(qq)
            gate_SDF = self.sigmoid(gate_SDF)

            SDFt_tilde = self.linear3_layer(qq)
            SDFt_tilde = self.tanh(SDFt_tilde)
            SDFt_tilde = self.dropout(SDFt_tilde)
            SDFt = gate_SDF * SDFt_tilde

            x = cat((SDFt, a_i),-1)
            gates_PKA = self.linear4_layer(x)
            gates_PKA = self.sigmoid(gates_PKA)

            PKAt_tilde = self.linear5_layer(x)
            PKAt_tilde = self.tanh(PKAt_tilde)
            PKAt_tilde = self.dropout(PKAt_tilde)
            PKAt = gates_PKA * PKAt_tilde

            k_s = cat((k,a_i,qd_i,sd_i),1)
            gates_KSU = self.output_layer(k_s)
            gates_KSU = self.sigmoid(gates_KSU)

            k = gates_KSU * k + (1 - gates_KSU) * PKAt

            h_i = unsqueeze(k,dim=1)
            h.append(h_i)

        output = cat(h,axis=1)
        logits = sum(target_data*output, dim=-1)
        y = self.sigmoid(logits)

        return y
