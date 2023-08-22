import os.path

import pandas as pd
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset
ModelConf = {
    "dkt_forget": ["timestamps"]
}
# 定义数据集类，继承自torch.utils.data.Dataset
class DktForgetDataset(Dataset):
    """dkt_forget 的数据集
             可用于初始化数据集：dkt_forget
                 训练数据、有效数据
                 常用测试数据（概念级别评估）、真实教育场景测试数据（问题级别评估）。
         参数：
             file_path (str): train_valid/测试文件路径
             input_type (list[str]): 数据集的输入类型，值在["questions", "concepts"]中
             Folds (set(int))：用于生成数据集的折叠，-1表示测试数据
             qtest (bool, 可选): 是否是问题评估。 默认为 False。
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(DktForgetDataset, self).__init__()
        self.sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = list(folds)
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_dkt_forget_qtest.pkl"
        else:
            processed_data = file_path + folds_str + "_dkt_forget.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.dqtest = \
                    self.__load_data__(self.sequence_path, folds)
                save_data = [self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.dqtest]
            else:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount = \
                    self.__load_data__(self.sequence_path, folds)
                save_data = [self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.dqtest = pd.read_pickle(
                    processed_data)
            else:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount = pd.read_pickle(processed_data)
        print(
            f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}, \
                       max_rgap: {self.max_rgap}, max_sgap: {self.max_sgap}, max_pcount: {self.max_pcount}")

    def __len__(self):
        """
            返回数据集长度
            返回:int: 数据集的长度
        """
        return len(self.dori['rseqs'])

    def __getitem__(self, index):
        """
            获取索引的数据
            参数:
                index (int): 需要获取的数据的索引
            返回:
                (tuple): 包含以下元素的元组:
                   - **q_seqs (torch.tensor)**: 0~seqlen-2交互的问题id序列
                    - **c_seqs (torch.tensor)**: 0~seqlen-2交互的知识概念id序列
                    - **r_seqs (torch.tensor)**: 0~seqlen-2交互的响应id序列
                    - **qshft_seqs (torch.tensor)**: 1~seqlen-1交互的问题id序列
                    - **cshft_seqs (torch.tensor)**: 1~seqlen-1交互的知识概念id序列
                    - **rshft_seqs (torch.tensor)**: 1~seqlen-1交互的响应id序列
                    - **mask_seqs (torch.tensor)**: 掩蔽值序列，形状为seqlen-1
                    - **select_masks (torch.tensor)**: 是否选择计算性能，0未选择，1已选择，只对1~seqlen-1有效，形状为seqlen-1
                    - **dcur (dict)**: 仅当self.qtest为True时使用，用于问题级评估
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_" + key] = self.dori[key]
                continue
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        dcurgaps = dict()
        for key in self.dgaps:
            seqs = self.dgaps[key][index][:-1] * mseqs
            shft_seqs = self.dgaps[key][index][1:] * mseqs
            dcurgaps[key] = seqs
            dcurgaps["shft_" + key] = shft_seqs

        if not self.qtest:
            return dcur, dcurgaps
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dcurgaps, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
                加载数据

                参数:
                    sequence_path (str): 序列文件路径
                    folds (list[int]): 折叠列表
                    pad_val (int, optional): 填充值，默认为-1。

                返回:
                    (tuple): 包含以下元素的元组:

                    - **q_seqs (torch.tensor)**: 0~seqlen-1交互的问题id序列
                    - **c_seqs (torch.tensor)**: 0~seqlen-1交互的知识概念id序列
                    - **r_seqs (torch.tensor)**: 0~seqlen-1交互的响应id序列
                    - **mask_seqs (torch.tensor)**: 掩蔽值序列，形状为seqlen-1
                    - **select_masks (torch.tensor)**: 是否选择计算性能，0未选择，1已选择，只对1~seqlen-1有效，形状为seqlen-1
                    - **max_rgap (int)**: 重复时间间隔的最大数量
                    - **max_sgap (int)**: 序列时间间隔的最大数量
                    - **max_pcount (int)**: 过去练习次数的最大数量
                    - **dqtest (dict)**: 只有当self.qtest为True时才不为空，用于问题级评估
        """
        dori = {"qseqs":[], "cseqs":[],"rseqs":[],"tseqs":[],"utseqs":[],"smasks": []}  # 空字典：存储原始的问题序列、答案序列等信息
        dgaps = {"rgaps":[],"sgaps":[],"pcounts":[]}  # 空字典：存储计算出的问题间隔、学习会话间隔和问题出现次数
        max_rgap, max_sgap, max_count = 0,0,0 

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        dqtest = {"qidxs":[],"rests":[],"orirow":[]}

        # 判断是否有timestamps字段
        flag = True
        for key in ModelConf["dkt_forget"]:
            if key not in df.columns:
                print(f"key: {key} not in data: {self.sequence_path}! can not run dkt_forget model!")
                flag = False
        assert flag == True

        for i, row in df.itterrow():
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            # 用calC方法计算出问题间隔、学习会话间隔和问题出现次数，存入dgaps字典
            rgap, sgap, pcount = self.calC(row)  
            dgaps["rgaps"].append(rgap)
            dgaps["sgaps"].append(sgap)
            dgaps["pcounts"].append(pcount)
            max_rgap = max(rgap) if max(rgap) > max_rgap else max_rgap
            max_sgap = max(sgap) if max(sgap) > max_sgap else max_sgap
            max_pcount = max(pcount) if max(pcount) > max_pcount else max_pcount

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            if key not in ["rseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])
        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        for key in dgaps:
            dgaps[key] = LongTensor(dgaps[key])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dgaps, max_rgap, max_sgap, max_pcount, dqtest

        return dori, dgaps, max_rgap, max_sgap, max_pcount

    def log2(self, t):
        import math
        return round(math.log(t + 1, 2))

    def calC(self, row):
        repeated_gap, sequence_gap, past_counts = [], [], []
        uid = row["uid"]
        # default: concepts 行数据中提取出问题序列（skills）和时间戳序列（timestamps）
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = row["timestamps"].split(",")
        dlastskill, dcount = dict(), dict()
        pret = None
        for s, t in zip(skills, timestamps):   # 许多情况下，这些值的范围可能会非常大，直接使用这些值可能会导致模型训练的不稳定。通过取对数，我们可以将这些大的值映射到一个较小的范围，从而使模型训练更加稳定。
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1: # skill没有出现，且！=-1
                curRepeatedGap = 0
            else:
                curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1  # minutes 计算当前问题时间t与上一次出现该问题之间的时间（dlastskill[s]）间隔，转换为分钟数（/1000/60），并取其对数（以2为底），然后加1得到当前的问题间隔
            dlastskill[s] = t
            repeated_gap.append(curRepeatedGap)

            if pret == None or t == -1:  # 序列中的第一个问题，或者问题的id为-1
                curLastGap = 0
            else:
                curLastGap = self.log2((t - pret) / 1000 / 60) + 1    # 计算当前问题时间t与上一个问题之间pret的时间间隔，转换为分钟数，并取其对数（以2为底），然后加1得到当前的学习会话间隔
            pret = t
            sequence_gap.append(curLastGap)

            dcount.setdefault(s, 0) 
            past_counts.append(self.log2(dcount[s]))  # 记录每个问题的出现次数，并取其对数（以2为底）得到当前的问题出现次数
            dcount[s] += 1 # 每次 + 1
        return repeated_gap, sequence_gap, past_counts
