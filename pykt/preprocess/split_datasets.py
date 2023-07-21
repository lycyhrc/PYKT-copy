import copy
import json
import os

import numpy as np
import pandas as pd

ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
                "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
        """
            从指定的文本文件中读取和处理学生交互数据，返回一个pandas DataFrame和有效的键。
            这个函数处理的数据包含六个部分：学生ID、问题ID、概念、回答、时间步长和使用时间。数据以6行为一个单位进行处理。
            针对学生交互数量不足或者回答不在指定响应集的数据，该函数会进行清洗，清洗完的数据以字典形式存储，并转化为pandas DataFrame。

            参数:
            fname: 文件名称字符串。
            min_seq_len: 最小的学生交互数量。默认为3，如果学生的交互数量少于这个数，该学生的数据将被删除。
            response_set: 指定的有效响应集。默认为[0, 1]，如果学生的响应不在这个集合内，该学生的数据将被删除。

            返回值:
            df: 一个pandas DataFrame，包含清洗完毕的学生数据。
            effective_keys: 一个集合，包含所有有效的键，这些键是在处理过程中实际存在的数据列名。

            该函数在处理过程中会打印删除的学生数量、删除的交互数量、错误的响应数量以及有效的交互数量。
        """
        # 初始化一些变量
        effective_keys = set() # 有效键的集合
        dres = dict() # 存储处理后对数据，形式{"key" : [v1,v2,...vn]}
        delstu, delnum, badr = 0, 0, 0 # 删除对 学生、交互、错误响应数量
        goodnum = 0 # 记录有效的交互数量
        # 打开txt文件，读取所有行
        with open(fname, "r", encoding='utf8') as fin:
            i = 0
            lines = fin.readlines()
            dcur = dict()  # 存储当前处理的数据
            while i < len(lines):
                line = lines[i].strip()
                if i % 6 == 0:  # stuid：处理学生ID
                    effective_keys.add("uid")
                    tmps = line.split(",")
                    stuid, seq_len = tmps[0], int(tmps[1])
                    if seq_len < min_seq_len: # 如果学生的交互数量少于最小数量，删除这个学生的数据
                        i += 6
                        dcur = dict()
                        delstu += 1
                        delnum += seq_len
                        continue
                    dcur["uid"] = stuid
                    goodnum += seq_len
                elif i % 6 == 1:  # question ids：处理问题ID
                    qs = []
                    if line.find("NA") == -1:  # 如果存在问题ID
                        effective_keys.add("questions")
                        qs = line.split(",")
                    dcur["questions"] = qs
                elif i % 6 == 2: # concept：处理概念
                    cs = []
                    if line.find("NA") == -1:  # 如果存在概念
                        effective_keys.add("concepts")
                        cs = line.split(",")
                    dcur["concepts"] = cs
                elif i % 6 == 3:  # response：处理回答
                    effective_keys.add("responses")
                    rs = []
                    if line.find("NA") == -1:
                        flag = True
                        for r in line.split(","):
                            try:
                                r = int(r)
                                if r not in response_set: # 如果回答不在设定的响应集内，删除这个学生的数据
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
                elif i % 6 == 4:  # timesteps：处理时间步长
                    ts = []
                    if line.find("NA") == -1:
                        effective_keys.add("timesteps")
                        timesets = line.split(",")
                    dcur["timesteps"] = timesets
                elif i % 6 == 5:  # usets：处理使用时间
                    utsets = []
                    if line.find("NA") == -1:
                        effective_keys.add("usetimes")
                        utsets = line.split(",")
                    dcur["usetimes"] = utsets

                    # 处理完每一位学生的所有数据后，将这些数据添加到最终的数据字典（dres）中
                    for key in effective_keys:
                        dres.setdefault(key, [])
                        if key != "uid":
                            dres[key].append(",".join([str(k) for k in dcur[key]]))
                        else:
                            dres[key].append(dcur[key]) # is uid
                    dcur = dict()
                i += 1
        # 将数据字典转换为pandas DataFrame
        df = pd.DataFrame(dres)
        print(f"delete bad stu num of len: {delstu},delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
        # 返回数据和有效的键
        return df, effective_keys


def get_max_concepts(total_df):
        """
        找出 DataFrame 中 "concepts" 列的值被逗号和下划线分割后得到的最大子字符串数量。(statics2011=1)
        :param total_df: 输入的 DataFrame
        :return:max_concepts： 最大的子字符串数量
        """
        max_concepts = 1
        for concepts in total_df["concepts"]:  # 从DataFrame提取出"concepts"列的每个元素
            for c in concepts.split(","):  # 将每个元素按逗号分割
                max_concepts = max(max_concepts, len(c.split("_")))  # 计算每个概念被下划线分割后得到的子字符串数量，并更新最大值
        return max_concepts


def calStatistics(df, stares, key):
        """
       从输入的 DataFrame (df) 中，统计并返回以下信息：

       - 有效响应的数量 (allin)：计算 'responses' 列中不为 '-1' 的响应总数。
       - 有效选择的数量 (allselect)：计算 'selectmasks' 列中为 '1' 的选择总数。
       - 唯一问题的数量：计算 'questions' 列中不重复（唯一）且不为 '-1' 的问题总数。
       - 唯一概念的数量：计算 'concepts' 列中不重复（唯一）且不为 '-1' 的概念总数。
       - DataFrame 的行数：即数据集中的总记录数量。

       上述所有统计信息都以逗号分隔的字符串形式，添加到输入的 stares 列表中。

       函数返回这些统计信息的数值形式，按顺序为：有效响应的数量，有效选择的数量，唯一问题的数量，唯一概念的数量，以及 DataFrame 的行数。

       :param df: 输入的 DataFrame，要从中进行统计的数据集。
       :param stares: 输入的列表，用于保存统计信息的字符串形式。
       :param key: 用于标识当前处理的数据集的关键字。
       :return: 返回一个包含五个元素的元组，分别为有效响应的数量，有效选择的数量，唯一问题的数量，唯一概念的数量，以及 DataFrame 的行数。
       """
        # 初始化所有统计量
        allin, allselect = 0, 0  # 有效响应次数，有效选择次数
        allqs, allcs = set(), set() # 不同问题数量，不同概念数量
        # 遍历 DataFrame的每一行
        for i, row in df.iterrows():
            # 如果存在'responses'列，统计有效的响应次数（即不为'-1'的响应）
            rs = row["responses"].split(",")
            curlen = len(rs) - rs.count("-1")
            allin += curlen

            # 如果存在'selectmasks'列，统计有效的选择次数（即为'1'的选择）
            if "selectmasks" in row:
                ss = row["selectmasks"].split(",")
                slen = ss.count("1")
                allselect += slen

            # 如果存在'concepts'列，统计唯一的概念数（即不为'-1'的概念）
            if "concepts" in row:
                cs = row["concepts"].split(",")
                fc = list()
                for c in cs:
                    cc = c.split("_")
                    fc.extend(cc)
                curcs = set(fc) - {"-1"}
                allcs |= curcs  # 将 curcs 中的所有元素添加到 allcs 中，这就完成了两个集合的并集操作

            # 如果存在'questions'列，统计唯一的问题数（即不为'-1'的问题）
            if "questions" in row:
                qs = row["questions"].split(",")
                curqs = set(qs) - {"-1"}
                allqs |= curqs

        # 将统计结果添加到 stares 列表中
        stares.append(",".join([str(s) for s in [key, allin, df.shape[0], allselect]])) # 四个值转换为字符串并用 , 连接，然后添加到stares列表的末尾
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


def get_inter_qidx(df):
        """
        在给定的 DataFrame df 上进行操作，为每个交互（interaction）生成一个全局ID。
        这个函数的主要目的是为每个交互生成一个唯一的标识符，以便在后续的处理中跟踪和引用这些交互。
        :param test_df:
        :return:
        """
        qidx_ids = [] # 存储每个interaction ID
        bias = 0 # 用于计算全局ID
        inter_num = 0 #用于计算交互对总数量
        for _, row in df.iterrows():
            ids_list = [str(x + bias)
                        for x in range(len(row['responses'].split(',')))] #当前行的交互数量，然后生成全局ID，全局ID等于交互的索引加上偏移量bias
            inter_num += len(ids_list)
            ids = ",".join(ids_list)
            qidx_ids.append(ids)
            bias += len(ids_list)
        assert inter_num - 1 == int(ids_list[-1])

        return qidx_ids


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
                    # [str(k) for k in dcur[key][0: maxlen]]))
                    dres[key].append(",".join(dcur[key][0: maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen+1, lenrs+1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k)
                                         for k in dcur[key][j-maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(
                    ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate(
                        [dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(
                ",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf




def add_qidx(dcur, global_qidx):
    idxs,rests = [],[]
    # idx =-1
    for r in dcur["is_repeat"]:
        if str(r) =='0':
            global_qidx = global_qidx + 1
        idxs.append(global_qidx)


def expand_question(dcur, global_qidx, pad_val = -1):
    dextend, dlast = dict(),dict()
    repeats = dcur["is_repeat"]
    last = -1
    dcur["qidxs"], dcur["rest"],global_qidx = add_qidx(dcur, global_qidx)
    for i in range(len(repeats)):
        if str(repeats[i])=='0':
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dlast[key] = dcur[key][0: i]
        if i==0:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key,[])
                dextend[key].append(dcur[key][0])
            dextend.setdefault("selectmasks",[])
            dextend["selectmasks"].append([pad_val])
        else:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend["selectmasks"][-1] +=[i]
                if last == "0" and str(repeats[i])=="0":
                    dextend[key][-1] += [dcur[key][i]]
                else:
                    dextend[key].append(dlast[key]+[dcur[key][i]])
            dextend.setdefault("selectmasks",[])
            if last == "0" and str(repeats[i])=="0":
                dextend["selectmasks"][-1] += [1]
            elif len(dlast["responses"]) == 0:  # the first question
                dextend["selectmasks"].append([pad_val])
            else:
                dextend["selectmasks"].append(
                    len(dlast["responses"]) * [pad_val] + [1])
        last = str(repeats[i])
    return dextend, global_qidx

def generate_question_sequences(df, effective_keys, window=True, min_seq_len=3, maxlen=200, pad_val = -1):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print(f"has no questions or concepts, has no question sequences")
        return False,None
        save_keys = list(effective_keys) + ["selectmasks","qidxs","rest","orirow"]
        dres = {}
        global_qidx = -1
        df["index"] = list(range(0, df.shape[0]))
        for i,row in df.iterrows():
            dcur = save_dcur(row,effective_keys)
            dcur["orirow"] = [row["index"]] * len(dcur["responses"])

            dexpand,global_qidx = expand_question(dcur,global_qidx)
            seq_num = len(dexpand["responses"])
            for j in range(seq_num):
                curlen = len(dexpand["responses"][j])
                if curlen < 2: # 不预测第1个题
                    continue
                if curlen < maxlen:
                    for key in dexpand:
                        pad_dim = maxlen - curlen
                        paded_info = np.concatenate([dexpand[key][j][0:],np.array([pad_val] * pad_dim)])
                        dres.setdefault(key,[])
                        dres[key].append(",".join([str(k) for k in paded_info]))
                    for key in ONE_KEYS:
                        dres.setdefault(key,[])
                        dres[key].append(dcur[key])
                else:
                    # 超出范围设置window
                    if window:
                        if dexpand["selectmasks"][j][maxlen-1] == 1:
                            for key in dexpand:
                                dres.setdefault(key,[])
                                dres[key].append(",".join([str(k) for k in dexpand[key][j][0:maxlen]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key,[])
                                dres[key].append(dcur[key])

                        for n in range(maxlen+1,curlen+1):
                            if dexpand["selectmasks"][j][n-1] == 1:
                                for key in dexpand:
                                    dres.setdefault(key,[])
                                    if key =="selectmasks":
                                        dres[key].append(",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
                                    else:
                                        dres[key].append(",".join([str(k) for k in dexpand[key][j][n-maxlen: n]]))

                                for key in ONE_KEYS:
                                    dres.setdefault(key,[])
                                    dres[key].append(dcur[key])
                    else:
                        # no window
                        k = 0
                        rest = curlen
                        while curlen >= k + maxlen:
                            rest = rest -maxlen
                            if dexpand["selectmasks"][j][k + maxlen - 1] == 1:
                                for key in dexpand:
                                    dres.setdefault(key,[])
                                    dres[key].append(",".join([str(s) for s in dexpand[key][j][k: k + maxlen]]))
                                for key in ONE_KEYS:
                                    dres.setdefault(key, [])
                                    dres[key].append(dcur[key])
                            k += maxlen
                        if rest < min_seq_len:  # 剩下长度<min_seq_len不预测
                            continue
                        pad_dim = maxlen -rest
                        for key in dexpand:
                            dres.setdefault(key,[])
                            paded_info = np.concatenate([dexpand[key][j][k:], np.array([pad_val] * pad_dim)])
                            dres[key].append(",".join([str(s) for s in paded_info]))
                        for key in ONE_KEYS:
                            dres.setdefault(key,[])
                            dres[key].append(dcur[key])
        dfinal = dict()
        for key in ALL_KEYS:
            if key in save_keys:
                dfinal[key] = dres[key]
        finaldf = pd.DataFrame(dfinal)
        return True,finaldf


def write_config(dataset_name, dkeyid2idx, effective_keys, configf, dpath, k, min_seq_len, maxlen, flag, other_config={}):
    input_type, num_q, num_c = [], 0, 0
    if "questions" in effective_keys:
        input_type.append("questions")
        num_q = len(dkeyid2idx["questions"])
    if "concepts" in effective_keys:
        input_type.append("concepts")
        num_c = len(dkeyid2idx["concepts"])
    folds = list(range(0, k))
    dconfig = {
        "dpath": dpath,
        "num_q": num_q,
        "num_c": num_c,
        "input_type": input_type,
        "max_concepts": dkeyid2idx["max_concepts"],
        "min_seq_len": min_seq_len,
        "maxlen": maxlen,
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": folds,
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv"
    }
    dconfig.update(other_config)
    if flag:
        dconfig["test_question_file"] = "test_question_sequences.csv"
        dconfig["test_question_window_file"] = "test_question_window_sequences.csv"

    # load old config
    with open(configf) as fin:
        read_text = fin.read()
        if read_text.strip() == "":
            data_config = {dataset_name: dconfig}
        else:
            data_config = json.loads(read_text)
            if dataset_name in data_config:
                data_config[dataset_name].update(dconfig)
            else:
                data_config[dataset_name] = dconfig

    with open(configf, "w") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)


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
        print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

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
            f"original train+valid interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

        #
        split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
        # print(split_seqs)
        ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "train+valid sequences")
        print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        split_seqs.to_csv(os.path.join(dname, "train_valid_sequences.csv"), index=None)

        # add default fold -1 to test!
        test_df["fold"] = [-1]*test_df.shape[0] # 添加fold=-1列
        test_df["cidxs"] = get_inter_qidx(test_df)
        test_seqs = generate_sequences(test_df, list(effective_keys)+['cidxs'], min_seq_len,maxlen)
        ins,ss,qs,cs,seqnum = calStatistics(test_df, stares, "test original")
        print(f"original test interaction num: {ins},select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        ins,ss,qs,cs,seqnum = calStatistics(test_seqs, stares,"test sequences")
        print(f"test sequence interactions num: {ins},select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        print("="*20)

        test_window_seqs = generate_window_sequences(test_df, list(effective_keys) + ['cidxs'], maxlen)
        flag, test_question_seqs = generate_question_sequences(test_df,effective_keys,False,min_seq_len,maxlen) # False
        flag, test_question_window_seqs = generate_question_sequences(test_df, effective_keys, True, min_seq_len, maxlen)

        test_df = test_df[config + ["cidxs"]]
        test_df.to_csv(os.path.join(dname,"test.csv"), index=None)
        test_seqs.to_csv(os.path.join(dname,"test_sequences.csv"),index=None)
        test_window_seqs.to_csv(os.path.join(dname, "test_window_sequences.csv"),index=None)
        ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "test window")
        print(f"test window interactions num: {ins},select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

        # print(test_df)
        # print(test_seqs)
        # print(test_window_seqs)

        if flag:
            test_question_seqs.to_csv(os.path.join(dname, "test_question_sequences.csv"),index=None)
            test_question_window_seqs.to_csv(os.path.join(dname,"test_question_window_sequences.csv"),index=None)

            ins,ss,qs,cs,seqnum = calStatistics(test_question_seqs,stares,"test question")
            print(f"test question interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
            ins, ss, qs, cs, seqnum = calStatistics(test_question_window_seqs, stares, "test question")
            print(f"test question window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")


        write_config(dataset_name=dataset_name, dkeyid2idx = dkeyid2idx, effective_keys=effective_keys,
                     config=config,dpath=dname,k=kfold,min_seq_len=min_seq_len,maxlen=maxlen,flag=flag)

        print("="*20)
        print("\n".join(stares))