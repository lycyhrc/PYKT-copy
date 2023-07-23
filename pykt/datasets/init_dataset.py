import os.path
from .data_loader import KTDataset
from torch.utils.data import DataLoader
def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, diff_level=None):
    print(f"dataset_name: {dataset_name}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"]) # 5 folds
    if model_name in ["dkt_forget", "bakt_time"]:
        pass
    elif model_name == "lpkt":
        pass
    else:
        curvalid = KTDataset(os.path.join(data_config["dpath"],data_config["train_valid_file"]), data_config["input_type"], {i}) # i= 0
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], all_folds - {i})
    trian_loader = DataLoader(curtrain,batch_size)
    valid_loader = DataLoader(curvalid, batch_size)

    return trian_loader,valid_loader