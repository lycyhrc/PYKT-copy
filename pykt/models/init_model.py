import torch
import numpy as np
import os

from .atdkt import ATDKT
from .dimkt import DIMKT
from .dkt import DKT
from .dkt_forget import DKTForget
from .dkvmn import DKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .kqn import KQN
from .qdkt import QDKT
from .sakt import SAKT
from .saint import SAINT
from .simplekt import simpleKT
from .sparsekt import sparseKT

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(
            device)
    elif model_name == "dimkt":
        model = DIMKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],**model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                         emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type,
                      emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                         emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(
            device)
    elif model_name == "iekt":  # max_concepts
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                     max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"], device=device).to(device)
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        model.apply(model.init_weights)
        model = model.to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    # 加载已经训练好的模型，路径saved_model
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))  # 加载保存在硬盘上的模型权重
    model.load_state_dict(net)  # 将加载的模型权重加载到现有的模型中
    return model