import os
import argparse
import json

import torch

torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy

from pykt.models.train_model import train_model
from pykt.models.init_model import init_model
from pykt.utils.utils import debug_print, set_seed
from pykt.datasets.init_dataset import init_dataset4train
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'


def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)


def main(params):
    # 1. 参数初始化与环境设置（wandb、随机种子、KT参数）
    if "use_wandb" not in params:
        params['use_wandb'] = 1

    if params['use_wandb'] == 1:
        import wandb
        wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
    # 2. 加载配置文件（训练、数据）
    debug_print(text="load config files.", fuc_name="main")
    # 从 'kt_config.json' 加载训练配置
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        # 根据不同模型调整批次大小，以解决内存不足问题（OOM）
        if model_name in ["dkvmn", "deep_irt", "sakt", "saint", "saint++", "akt", "atkt", "lpkt", "skvmn", "dimkt"]:
            train_config["batch_size"] = 64  ## because of OOM
        if model_name in ["simplekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64  ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16
        if model_name in ["qdkt", "qikt"] and dataset_name in ['algebra2005', 'bridge2algebra2006']:
            train_config["batch_size"] = 32
        model_config = copy.deepcopy(params)
        # 创建模型配置的深层拷贝，排除不需要的参数
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        # 如果在参数中存在批次大小和训练周期数，则覆盖模型配置中的值
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']

    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config[
        "optimizer"]
    # 从 'data_config.json' 提取其他相关配置
    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:  # 优先使用数据配置中的最大序列长度
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]
    # 3. 数据集初始化（init_dataset4train）
    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)

    debug_print(text="init_dataset", fuc_name="main")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    else:
        # 不同的模型需要不同的参数，所以要单独处理 'dimkt'
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size,
                                                            diff_level=diff_level)
    # 生成保存模型时使用的参数字符串
    params_str = "_".join([str(v) for k, v in params.items() if not k in ['other_config']])

    # 如果需要，并且使用了 wandb，为了保证唯一性，为 params_str 添加一个 UUID
    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str + f"_{str(uuid.uuid4())}"
    # 创建用于保存模型的检查点目录
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    # 输出训练和模型配置的详细信息
    print(
        f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    # 对特定模型（如 'dimkt'）进行额外处理
    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    # 保存配置信息到 JSON 文件
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    # 提取学习率和其他优化器相关的信息
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
        if remove_item in model_config:
            del model_config[remove_item]
    # 如果模型需要序列长度信息，则在模型配置中更新它
    if model_name in ["saint", "saint++", "sakt", "atdkt", "simplekt", "bakt_time"]:
        model_config["seq_len"] = seq_len
    # 4.模型和优化器初始化（init_model）
    debug_print(text="init_model", fuc_name="main")
    print(f"model_name:{model_name}")
    # 根据提供的模型名称、模型配置、数据配置和嵌入类型，初始化模型
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    # 针对特定模型（例如 'hawkes'、'iekt'、'dimkt'）进行额外处理
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

    # 初始化各项指标，用于保存最佳的模型和结果
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    # 5. 模型训练（train_model）
    debug_print(text="train model", fuc_name="main")

    # 开始模型训练，并获取各项指标
    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader,
                                                                                                   valid_loader,
                                                                                                   num_epochs, opt,
                                                                                                   ckpt_path, None,
                                                                                                   None, save_model)
    # 如果需要保存模型，则加载最佳模型并保存
    if save_model:
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
        best_model.load_state_dict(net)

    # 打印最终的测试结果
    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(
        round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(
        validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))

    # 模型保存路径
    model_save_path = os.path.join(ckpt_path, emb_type + "_model.ckpt")
    print(f"end:{datetime.datetime.now()}")

    # 如果启用了 wandb，则记录验证结果、最佳轮次和保存的模型路径
    if params['use_wandb'] == 1:
        wandb.log({
            "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch, "model_save_path": model_save_path})
