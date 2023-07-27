import os.path

from torch.utils.data import DataLoader

from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset

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
    # elif model_name in ["atdkt"]:
    #     test_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
    #     test_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
    #     if "test_question_file" in data_config:
    #         test_question_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
    #         test_question_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    # elif model_name in ["dimkt"]:
    #     test_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1}, diff_level=diff_level)
    #     test_window_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1}, diff_level=diff_level)
    #     if "test_question_file" in data_config:
    #         test_question_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)
    #         test_question_window_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)
    else:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader,test_question_window_loader = None,None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    # 如果 cur 中的对应值大于 max_rgap、max_sgap 和 max_pcount 的当前值，则更新这些值。
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount


def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, diff_level=None):
    print(f"dataset_name: {dataset_name}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"]) # 5 folds
    if model_name in ["dkt_forget", "bakt_time"]:
        max_rgap, max_sgap, max_pcount = 0,0,0
        curvalid = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds-{i})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name == "lpkt":
        pass
    else:
        print(KTDataset)
        curvalid = KTDataset(os.path.join(data_config["dpath"],data_config["train_valid_file"]), data_config["input_type"], {i}) # i= 0
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], all_folds - {i})
    train_loader = DataLoader(curtrain,batch_size)
    valid_loader = DataLoader(curvalid, batch_size)

    if model_name in ["dkt_forget", "bakt_time"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1

    return train_loader, valid_loader