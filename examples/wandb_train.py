import copy
import json
import os

import torch
from torch.optim import SGD, Adam

from pykt.datasets.init_dataset import init_dataset4train
from pykt.models.init_model import init_model
from pykt.preprocess.utils import set_seed, debug_print

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'


def save_config(train_config, model_config, data_config, params, save_dir):
    d ={
        "train_config":train_config,
        "model_config":model_config,
        "data_config": data_config,
        "params":params
    }
    save_path = os.path.join(save_dir,"config.json")
    with open(save_path,"w") as fout:
        json.dump(d, fout)

def main(params):
    #设置 wandb环境（params["use_wandb"] == 0时不使用wandb）
    if "use_wandb" not in params:
        params["use_wandb"] = 1
    if params["use_wandb"] == 1:
        import wandb
        wandb.init()

    set_seed(params["seed"]) # 设置随机种子，保证结果可复现
    # 变量名 <- 参数名
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"],params["dataset_name"],\
    params["fold"],params["emb_type"],params["save_dir"]

    debug_print(text="load config files.", fuc_name="main")

    # 读取和处理模型训练配置信息
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]

        # 根据模型名称和数据集名称来调整batch_size，避免内存溢出
        # 这部分代码应该根据实际硬件和数据情况进行调整
        pass

        model_config = copy.deepcopy(params) # 深拷贝，原始变量不受影响
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]

        if 'batch_size' in params:
            train_config['batch_size'] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']

        batch_size, num_epochs, optimizer = train_config['batch_size'], train_config['num_epochs'], train_config['optimizer']

        with open("../configs/data_config.json") as fin:
            data_config = json.load(fin)
        if 'maxlen' in data_config[dataset_name]:  # prefer to use the maxlen in data config
            train_config["seq_len"] = data_config[dataset_name]['maxlen']
        seq_len = train_config["seq_len"]

        print("Start init data")
        print(dataset_name,model_name,data_config,fold,batch_size)

        debug_print(text="init_dataset", fuc_name="main")
        if model_name not in ["dimkt"]:
            train_loader,valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
        else:
            pass

        params_str = "_".join([str(v) for k, v in params.items() if not k in ['other_config']])
        print(f"params: {params},params_str:{params_str}")

        if params['add_uuid'] == 1 and params["use_wandb"]==1:
            import uuid
            params_str = params_str +f"_{ str(uuid.uuid4())}" # 生成唯一哈希值
        ckpt_path = os.path.join(save_dir,params_str)  # saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_bacf1d0a-56de-4742-b905-d225522e05c0
        if not os.path.isdir((ckpt_path)):
            os.makedirs(ckpt_path)
        print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
        print(f"model_config: {model_config}")
        print(f"train_config: {train_config}")

        save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)

        learning_rate = params["learning_rate"]
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        if model_name in ["saint", "saint++", "sakt", "atdkt", "simplekt", "bakt_time"]:
            model_config["seq_len"] = seq_len

        debug_print(text="init_model", fuc_name="main")
        print(f"model_name:{model_name}")
        model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        print(f"model is {model}")

        if model_name =="dimkt":
            pass
        else:
            if optimizer =='sgd':
                opt = SGD(model.parameters(),learning_rate,momentum=0.9)
            elif optimizer =='adam':
                opt = Adam(model.parameters(), learning_rate)

        testauc, testacc = -1, -1
        window_testauc, window_testacc = -1, -1
        validauc, validacc = -1, -1
        best_epoch = -1
        save_model = True

