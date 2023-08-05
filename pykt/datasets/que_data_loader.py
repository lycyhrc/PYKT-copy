import os.path

import pandas as pd
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset


class KTQueDataset(Dataset):
    def __init__(self, file_path, input_type, folds, concept_num, max_concepts, qtest=False):
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise "The input types must contain both questions and concepts"

        fols = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        processed_data = file_path + folds_str + "_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            self.dori = self.__load_data__(sequence_path, fols)
            save_data = self.dori
            pd.to_pickle(save_data,processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, "
              f"qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
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
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key == 'cseqs':
                seqs = self.dori[key][index][:-1, :]
                shft_seqs = self.dori[key][index][1:, :]
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        return dcur

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        从给定的CSV文件加载数据。

        参数：
            sequence_path (str): 包含学习交互序列的CSV文件的路径。
            folds (list): 需要包含在加载数据中的折叠列表，用于交叉验证。
            pad_val (int, optional): 用于填充序列的值，默认为-1。

        返回：
            dict: 包含已加载和预处理数据的字典。
        """
        # 初始化一个用于存储数据的空字典
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        # 从CSV文件读取数据，并根据 'fold' 筛选行
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()
        interaction_num = 0
        # 遍历行并根据输入类型提取序列
        for i,row in df.iterrows():
            if "concepts" in self.input_type:
                # 在行中预处理概念序列
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts  # 如果 concept 为 "-1"，则生成一个全为 -1，长度为 self.max_concepts 的列表
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills + [-1] * (self.max_concepts - len(skills))   # 如果 skills 的长度小于
                        # self.max_concepts，则在其后面添加 -1，使其长度达到 self.max_concepts
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

        for key in dori:
            if key not in ["rseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:, :-1] != pad_val) * (dori["rseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")

        return dori