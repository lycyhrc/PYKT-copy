import torch
import numpy as np
import os

from pykt.models.dkt import DKT
from pykt.models.dkt_forget import DKTForget

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        # **model_config会将model_config字典中的每一项转换为一个参数传入DKT模型的构造函数中
        model = DKT(data_config['num_c'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"],data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model