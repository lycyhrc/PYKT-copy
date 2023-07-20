import copy
import os

import numpy as np
import pandas as pd

ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
            "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    with open(fname, "r", encoding='utf8') as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()  # key-value形式
        while i < len(lines):
            line = lines[i].strip()
            if i % 6 == 0:  # stuid
                effective_keys.add("uid")
                tmps = line.split(",")
                stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:
                    i += 6
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 6 == 1:  # question ids
                qs = []
                if line.find("NA") == -1:  # 有question记录
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 6 == 2:
                cs = []
                if line.find("NA") == -1:  # 有concept记录
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 6 == 3:  # response
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:
                                print(f"error response in line: {i}")
                                flag = False
                                break
                            rs.append(r)
                        except:
                            print(f"error response in line: {i}")
                            flag = False
                            break
                    if not flag:
                        i += 3
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 6 == 4:  # timesteps
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timesteps")
                    timesets = line.split(",")
                dcur["timesteps"] = timesets
            elif i % 6 == 5:  # usets
                utsets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    utsets = line.split(",")
                dcur["usetimes"] = utsets

                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
    df = pd.DataFrame(dres)
    print(f"delete bad stu num of len: {delstu},delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
    return df, effective_keys


def get_max_concepts(total_df):
    """
    找出 DataFrame 中 "concepts" 列的值被逗号和下划线分割后得到的最大子字符串数量。(statics2011=1)
    :param total_df:
    :return:max_concepts
    """
    max_concepts = 1
    for i, row in total_df.iterrows():  # 遍历一个df
        cs = row["concepts"].split(",")
        max_length = 0
        for c in cs:
            length = len(c.split("_"))
            if length > max_length:
                max_length = length
        num_concepts = max_length

        if num_concepts > max_concepts:
            max_concepts = num_concepts
        return max_concepts


def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s)
                            for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]


def extend_multi_concepts(df, effective_keys):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("has no questions or concepts! return original.")
        return df, effective_keys


def id_mapping(df):
    """
    为 DataFrame 中的 "questions"、"concepts" 和 "uid" 列中的每个唯一值分配一个新的唯一标识符，并创建一个新的 DataFrame，其中这些列的值被替换为新的标识符。    :param df:
    :return:
    """
    id_keys = ["qusetions", "concepts", "uid"]
    dres = dict()  # 存储新的df数据
    dkeyid2idx = dict()  # 存储每个列的唯一值到新标识符的映射
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:  # 当前列不在id_keys
            dres[key] = df[key]  # 数据复制到 dres 中
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())  # 如果这个键已经存在，则不改变它的值
            dres.setdefault(key, [])
            curids = []  # 用于存储当前行的新标识符
            for id in row[key].split(","):  # 遍历当前行的当前键的值
                if id not in dkeyid2idx[key]:  # 遍历当前行的当前键的值
                    dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))  # ：将 curids 中的所有元素连接成一个由逗号分隔的字符串，然后添加到 dres[key] 中
    finaldf = pd.DataFrame(dres)  # 将 dres 转换为一个新的 DataFrame finaldf
    return finaldf, dkeyid2idx


def save_id2idx(dkeyid2idx, param):
    pass


def train_test_split(df, test_ratio=0.2):
    """
    在给定的 DataFrame df 上进行操作，将其随机分割为训练集和测试集。
    :param df:
    :param test_ratio: 测试集的比例（默认为 0.2）
    :return: 训练集和测试集
    """
    df = df.sample(frac=1.0, random_state=42)
    datanum = df.shape[0]
    test_num = int(datanum * test_ratio)
    train_num = datanum - test_num
    train_df = df[0:train_num]
    test_df = df[train_num:]

    print(f"total num: {datanum},train+valid num: {train_num},test num: {test_num}")
    return train_df, test_df


def KFold_split(train_df, k=5):
    """
    为 k 折交叉验证（k-fold cross-validation）提供数据
    :param train_df:
    :param k:
    :return:
    """
    df = train_df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_ratio = 1 / k
    test_num = int(datanum * test_ratio)
    rest = datanum % k  # k折划分不完全的数据

    start = 0
    folds = []
    for i in range(0, k):
        end = start + test_num + (i < rest)  # 简单写法：i < rest 判断是否需要给当前的子集添加一个额外的元素
        folds.extend([i] * (end - start))
        print(f"fold: {i + 1}, start: {start}, end: {end}, total num: {datanum}")
        start = end
    # report
    finaldf = copy.deepcopy(df)
    finaldf["fold"] = folds  # 在 finaldf 中添加一个新的列 "fold"，其值为 folds
    return finaldf


ONE_KEYS = ["fold", "uid"]


def save_dcur(row, effective_keys):
    """
    从行中提取出有效键对应的值，并根据键是否在 ONE_KEYS 中以不同的方式处理这些值。
    如果键在 ONE_KEYS 中，那么对应的值将直接被存储；否则，对应的值将被分割为一个列表。
    :param row:
    :param effective_keys:
    :return:
    """
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            dcur[key] = row[key].split(",")
        else:
            dcur[key] = row[key]
    return dcur


def generate_sequences(df, effective_keys, min_seq_len, maxlen, pad_val=-1):
    """
    函数在给定的 DataFrame df 上进行操作，将每一行的数据转换为一系列的序列。这个函数的主要目的是为序列模型（如循环神经网络）提供数据。
    :param df:  train+valid df
    :param effective_keys:
    :param min_seq_len:
    :param maxlen:
    :return:
    """
    save_keys = list(effective_keys) + ['selectmasks']
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs > j + maxlen:
            rest = rest - maxlen
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j:j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum = dropnum + rest
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

    # after preprocess data,report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf


def main(dname, fname, dataset_name, config_data, min_seq_len=3, maxlen=200, kfold=5):
    stares = []
    total_df, effective_keys = read_data(fname)
    # print(total_df)
    total_df.to_csv('../data/statics2011/total_df.csv', index=False)
    # print(effective_keys)
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1
    # print(max_concepts)

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
    print("=" * 20)
    print(
        f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    extends, _, qs, cs, seqnum = calStatistics(
        total_df, stares, "extend multi")
    print("=" * 20)
    print(
        f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))

    # train test split & generate sequences
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    print(splitdf)

    # to csv
    config = []
    for key in ALL_KEYS:
        if key in effective_keys:
            config.append(key)
    print(effective_keys)
    splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)  # 选取effective_keys列存储

    # original train+valid
    ins, ss, qs, cs, seqnum = calStatistics(
        splitdf, stares, "original train+valid")
    print(
        f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # 
    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    print(split_seqs)
    ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "train+valid sequences")
    print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs.to_csv(os.path.join(
        dname, "train_valid_sequences.csv"), index=None)
