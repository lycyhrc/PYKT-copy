import os.path

import numpy as np
import pandas as pd

from pykt.preprocess.split_datasets import read_data, get_max_concepts, calStatistics, save_id2idx, ALL_KEYS, \
    train_test_split, KFold_split, save_dcur, write_config, ONE_KEYS


def id_mapping_que(df):
    id_keys = ["questions","concepts","uid"]
    dres = dict() # 存储处理后的结果
    dkeyid2idx = dict() # 存储每个标签对唯一数值ID
    print(f"df.columns: {df.columns}")

    # 除了"id_keys"中的列以外的其他列直接复制到dres中
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]

    # 对 DataFrame df的每一行进行迭代。对于每一行数据，遍历"id_keys"中的每个键
    for i,row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            # 对于存在的键，首先确保这个键在`dkeyid2idx`和`dres`中都有对应的条目
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key,[])

            # 对于当前行的当前键对应的值，将其按照","分割，对于每个得到的标签，再将其按照"_"分割，得到子标签
            curids = []
            for id in row[key].split(","):
                sub_ids = id.split("_") # 子标签
                sub_curids = []

                # 如果某个子标签还没有被分配过数值 ID，那么就给它分配一个新的数值 ID
                for sub_id in sub_ids:
                    if sub_id not in dkeyid2idx[key]:
                        dkeyid2idx[key][sub_id] = len(dkeyid2idx[key]) # 如果某个子标签还没有被分配过数值 ID，那么就给它分配一个新的数值 ID，这个 ID 等于当前键对应的字典中已有的条目数量
                    sub_curids.append(str(dkeyid2idx[key][sub_id])) # 转换为字符串
                curids.append("_".join(sub_curids)) # 用"_"连接起来
            # 将处理后的标签添加到`dres`对应的键中
            dres[key].append(",".join(curids))
    # 在遍历完所有的行和键之后，将`dres`转换为 DataFrame
    finaldf = pd.DataFrame(dres)
    # 返回新的 DataFrame 和字典`dkeyid2idx`
    return finaldf,dkeyid2idx


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    print(df)
    print(effective_keys)
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))  # [str(k) for k in dcur[key][j: j + maxlen]]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate([dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    # after preprocess data, report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf


def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        if lenrs > maxlen:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][0: maxlen]))  # [str(k) for k in dcur[key][0: maxlen]]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen + 1, lenrs + 1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k) for k in dcur[key][j - maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate([dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf


def main(dname, fname, dataset_name, configf, min_seq_len=3, maxlen = 200, kfold=5):
    """

    :param dname: (str) data folder path
    :param fname: (str) the data file used to split, need 6 columns format is: (NA indicates the dataset has no corresponding info)
            uid,seqlen: 50121,4
            quetion ids: NA
            concept ids: 7014,7014,7014,7014
            responses: 0,1,1,1
            timestamps: NA
            cost times: NA
    :param dataset_name:(str) dataset name
    :param configf: (str) the dataconfig file path
    :param min_seq_len: (int, optional): the min seqlen,sequences less than this value will be filtered out. Defaults to 3.
    :param maxlen:(int, optional): the max seqlen. Defaults to 200.
    :param kfold:(int, optional): the folds num needs to split. Defaults to 5.
    :return:
    """
    stares = []
    total_df, effective_keys = read_data(fname)
    print(total_df)
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1
    print(max_concepts)

    oris,_,qs,cs,seqnum = calStatistics(total_df, stares, "original") # allselect 不考虑
    print("="*20)
    print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # just for id map
    total_df, dkeyid2idx = id_mapping_que(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json")) # 新的DataFrame和一个字典dkeyid2idx（用于保存每个标签对应的唯一数值ID
    effective_keys.add("fold")

    # ALL_KEYS的列表中的每个键是否在effective_keys中
    df_save_keys = []
    for key in ALL_KEYS:
        if key in effective_keys:
            df_save_keys.append(key)

    # train test split
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    splitdf[df_save_keys].to_csv(os.path.join(dname, "train_valid_question.csv"),index=None)
    ins, ss, qs, cs, seqnum = calStatistics(splitdf, stares, "original train+valid question level")
    print(f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print(splitdf)
    # generate sequences
    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    ins,ss,qs,cs,seqnum = calStatistics(split_seqs, stares,"train+valid sequences question level")
    print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    # split_seqs.to_csv(os.path.join(dname, "train_valid_sequences_quelevel.csv"), index=None)

    # for test dataset
    # add default fold -1 to test
    test_df["fold"] = [-1] *test_df.shape[0]
    test_seqs = generate_sequences(test_df, list(effective_keys),min_seq_len,maxlen)
    # dispaly
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original question level")
    print(f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    ins, ss, qs, cs, seqnum = calStatistics(test_seqs, stares, "test sequences question level")
    print(f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print("=" * 20)

    test_window_seqs = generate_window_sequences(test_df,list(effective_keys),maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "test window question level")
    print(f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # sava csv
    test_df.to_csv(os.path.join(dname, "test_quelevel.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences_quelevel.csv"), index=None)
    test_window_seqs.to_csv(os.path.join(dname, "test_window_sequences_quelevel.csv"), index=None)

    other_config = {
        "train_valid_original_file_quelevel":"train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv"
    }

    write_config(dataset_name=dataset_name, dkeyid2idx = dkeyid2idx,effective_keys=effective_keys,
                 configf=configf,dpath=dname, k= kfold, min_seq_len=min_seq_len,maxlen=maxlen,flag=True,
                 other_config=other_config)

    print("="*20)
    print("\n".join(stares))