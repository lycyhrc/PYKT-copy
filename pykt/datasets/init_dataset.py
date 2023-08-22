import os,sys

from torch.utils.data import DataLoader

from pykt.datasets.lpkt_dataloader import LPKTDataset
from pykt.datasets.lpkt_utils import generate_time2idx

from .atdkt_dataloader import ATDKTDataset
from .data_loader import KTDataset
from .dimkt_dataloader import DIMKTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .que_data_loader import KTQueDataset
from ..config.config import que_type_models


def init_test_datasets(data_config, model_name, batch_size, diff_level=None):
    dataset_name = data_config["dataset_name"]
    print(f"model_name is {model_name}, dataset_name is {dataset_name}")
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget", "bakt_time"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
        elif model_name in que_type_models:
            test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1],
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1],
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            test_question_dataset = None
            test_question_window_dataset= None
    # elif model_name in ["lpkt"]:
    #     print(f"model_name in lpkt")
    #     at2idx, it2idx = generate_time2idx(data_config)
    #     test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
    #     test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
    #     test_question_dataset = None
    #     test_question_window_dataset= None
    # elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
    #     test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
    #     test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
    #     if "test_question_file" in data_config:
    #         test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
    #         test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    # elif model_name in que_type_models:
    #     test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
    #                     input_type=data_config["input_type"], folds=[-1],
    #                     concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    #     test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
    #                     input_type=data_config["input_type"], folds=[-1],
    #                     concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    #     test_question_dataset = None
    #     test_question_window_dataset= None
    elif model_name in ["atdkt"]:
        test_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    # elif model_name in ["dimkt"]:
    #     test_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1}, diff_level=diff_level)
    #     test_window_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1}, diff_level=diff_level)
    #     if "test_question_file" in data_config:
    #         test_question_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)
    #         test_question_window_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)
    else:
        # concept-level
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        # question-level
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)

    # 初始化为DataLoader形式
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader,test_question_window_loader = None,None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(test_loader):
        print(batch)
        if i == 2:  # 停止打印数据 after 5 batches
            break
    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    # 如果 cur 中的对应值大于 max_rgap、max_sgap 和 max_pcount 的当前值，则更新这些值。
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount

def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, diff_level=None):
    """
       函数功能：
       初始化训练集。基于指定的数据集名称、模型名称、数据配置、交叉验证折数，生成对应的训练数据集和验证数据集，并进行相关配置。

       data_config["train_valid_file"]：大多数数据集初始化
       data_config["train_valid_file_quelevel"]：lpkt 和 que_type_models

       输入参数：
       dataset_name: 字符串，指定数据集的名称。
       model_name: 字符串，指定模型的名称。
       data_config: 字典，指定数据集的详细配置信息。
       i: 整数，当前交叉验证的折数。
       batch_size: 整数，指定每个batch的大小。
       diff_level: 整数，指定难度等级（如果有的话）。

       输出：
       train_loader: DataLoader对象，生成训练数据加载器，以便进行批量训练。
       valid_loader: DataLoader对象，生成验证数据加载器，以便进行批量验证。
    """
    print(f"dataset_name:{dataset_name}")
    data_config = data_config[dataset_name]  # 指定数据集的详细配置文件
    all_folds = set(data_config["folds"])   # 所有 folds
    if model_name in ["dkt_forget", "bakt_time"]:
        max_rgap, max_sgap, max_pcount = 0, 0, 0   # 初始化最大的回顾间隔，学习间隔和问题出现次数为0
        curvalid = DktForgetDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"], {i})
        curtrain = DktForgetDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"], all_folds - {i})
        # 更新最大的回顾间隔，学习间隔和问题出现次数，确保得到的最大值是在整个数据集（包括训练集和验证集）中的最大值，从而能够正确地处理这些特征
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name in que_type_models:
        curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds={i},
                                concept_num=data_config["num_c"], max_concepts=data_config['max_concepts'])
        curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds-{i},
                                concept_num=data_config["num_c"], max_concepts=data_config['max_concepts'])

    elif model_name == "lpkt":
        at2idx, it2idx = generate_time2idx(data_config)
        # json_str = json.dumps(at2idx)
        # with open('at2idx.json', 'w') as json_file:
        #     json_file.write(json_str)
        # json_str_2 = json.dumps(it2idx)
        # with open('it2idx.json', 'w') as json_file2:
        #     json_file2.write(json_str_2)
        curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx,
                               it2idx, data_config["input_type"], {i})
        curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx,
                               it2idx, data_config["input_type"], all_folds - {i})
    # elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
    #     curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
    #                          data_config["input_type"], {i})
    #     curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
    #                          data_config["input_type"], all_folds - {i})
    elif model_name in ["atdkt"]:
        curvalid = ATDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], {i})
        curtrain = ATDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], all_folds - {i})
    elif model_name == "dimkt":
        curvalid = DIMKTDataset(data_config["dpath"],
                                os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], {i}, diff_level=diff_level)
        curtrain = DIMKTDataset(data_config["dpath"],
                                os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], all_folds - {i}, diff_level=diff_level)
    else:
        # 对于其他模型，创建基本的KTDataset训练集和验证集
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], all_folds - {i})
    # 创建训练和验证数据加载器
    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)

    # 对于特定模型（dkt_forget 和 bakt_time），更新数据集配置信息中的 num_rgap、num_sgap 和 num_pcount
    if model_name in ["dkt_forget", "bakt_time"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1

    return train_loader, valid_loader