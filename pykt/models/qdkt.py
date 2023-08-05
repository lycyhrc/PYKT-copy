from torch.nn import Module
from torch.nn import Embedding, LSTM, Dropout, Linear
import torch
import torch.nn.functional as F

from pykt.models.que_base_model import QueBaseModel
from pykt.utils.utils import debug_print


# QDKNet是网络模型的定义，包含Embedding,STM,dropout和linear层
class QDKTNet(Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path=""
                 , pretrain_dim=768, device="cpu"):
        super().__init__()
        # 初始化一些属性和层
        self.model_name = "qdkt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.device = device
        self.emb_type = emb_type

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_q)

    def forward(self, q, c,r,data=None):
        # 前向传播方法，定义了网络计算流程
        x = (q + self.num_q * r)[:,:-1]  # 前N-1个question-responses序列
        xemb = self.interaction_emb(x)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        y = (y * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)

        return y


# QDKT是一个扩展自QueBaseModel的模型类，包含了一个QDKTNet对象以及一些训练和预测的方法
class QDKT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,
                 device="cpu"):
        model_name = "qdkt"

        debug_print(f"emb_type is {emb_type}", fuc_name="QDKT")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device)
        self.model = QDKTNet(num_q=num_q, num_c=num_c, emb_size=emb_size, dropout=dropout, emb_type=emb_type,
                             emb_path=emb_path, pretrain_dim=pretrain_dim)

        self.model = self.model.to(device)
        self.emb_type = emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")

    def train_one_step(self, data, process=True, return_all=False):
        outputs, data_new = self.predict_one_step(data, return_details=True, process=process)
        loss = self.get_loss(outputs, data_new['rshft'], data_new['sm'])
        return outputs,loss

    def predict_one_step(self, data, return_details=False,process=True):
        data_new = self.batch_to_device(data, process=process)
        outputs = self.model(data_new['cq'].long(), data_new['cc'], data_new['cr'].long(), data=data_new) # 前N条
        if return_details:
            return outputs, data_new
        else:
            return outputs