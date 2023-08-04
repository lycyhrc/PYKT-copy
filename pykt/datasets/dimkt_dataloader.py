import csv
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset


def question_diffcult(df, questions_difficult_path, questions, responses, diff_level):
    qd = {}
    df = df.reset_index(drop=True)
    set_questions = set(np.array(df[questions]))  # 不重复问题
    for i in tqdm(set_questions):  # 遍历每一个问题
        count = 0
        idx = df[(df.questions == i)].index.tolist()
        tmp_data = df.iloc[idx]  # 哪些行是同一个问题
        correct_1 = tmp_data[responses]
        if len(idx) < 30:
            qd[i] = 1
            continue
        else:
            for response_value in np.array(correct_1):   # tuple to list in -1 index
                count += response_value  # 0,1 序列
            if count == 0:
                qd[i] = 1
            else:
                avg = int((count / len(correct_1)) * diff_level) + 1
                qd[i] = avg

    with open(questions_difficult_path, 'w', newline='',encoding='UTF-8') as f: # dict to csv
        writer = csv.writer(f)
        writer.writerow(qd.keys())
        writer.writerow(qd.values())

    return


def skill_difficult(df, skills_difficult_path, concepts, responses, diff_level):
    sd = {}
    df = df.reset_index(drop=True)
    set_skills = set(np.array(df[concepts]))  # 不重复问题
    for i in tqdm(set_skills):
        count = 0
        idx = df[(df.concepts == i)].index.tolist()
        tmp_data = df.iloc[idx]
        correct_1 = tmp_data[responses]
        if len(idx) < 30:
            sd[i] = 1
            continue
        else:
            for response_value in np.array(correct_1):
                count += response_value  # 0,1 序列
            if count == 0:
                sd[i] = 1
            else:
                avg = int((count / len(correct_1)) * diff_level) + 1
                sd[i] = avg

    with open(skills_difficult_path, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(sd.keys())
        writer.writerow(sd.values())

    return


def difficult_compute(df, skills_difficult_path, questions_difficult_path, diff_level):

    concepts, questions, responses = [], [], []
    for i,row in tqdm(df.iterrows()):
        concept = [int(_) for _ in row["concepts"].split(",")]
        question = [int(_) for _ in row["questions"].split(",")]
        response = [int(_) for _ in row["responses"].split(",")]
        length = len(response)
        index = -1

        for j in range(length):
            if response[length - j - 1] != -1:  # 列表中最后一个非-1值的索引
                index = length -j
                break

        # 记录每个user有效的concepts,questions,responses序列（非-1）
        concepts = concepts + concept[:index]
        questions = questions + question[:index]
        responses = responses + response[:index]

    # 一维 question
    df2 = pd.DataFrame({'concepts': concepts, 'questions': questions, 'responses': responses})

    question_diffcult(df2,questions_difficult_path,'questions','responses',diff_level=diff_level)
    skill_difficult(df2, skills_difficult_path, 'concepts', 'responses', diff_level=diff_level)

    return


class DIMKTDataset(Dataset):
    def __init__(self, dpath, file_path, input_type, folds, qtest=False, diff_level=None):
        super(DIMKTDataset, self).__init__()
        self.sequence_path = file_path  # csv file
        self.input_type = input_type
        self.qtest = qtest

        # dimkt: difficulty level（exists & read）
        self.diff_level = diff_level
        skills_difficult_path = dpath + f'/skills_difficult_{diff_level}.csv'
        questions_difficult_path = dpath + f'/questions_difficult_{diff_level}.csv'

        # first storage
        if not os.path.exists(skills_difficult_path) or not os.path.exists(questions_difficult_path):
            print("start computing difficulties")
            train_file_path = dpath + "/train_valid_sequences.csv"  # 在padding序列计算difficulty
            df = pd.read_csv(train_file_path)
            difficult_compute(df, skills_difficult_path, questions_difficult_path, diff_level=self.diff_level)

        folds = sorted(list(folds))
        folds_str = '_' +'_'.join([str(_) for _ in folds])
        # pkl storage
        if self.qtest:
            processed_data = file_path + folds_str + f"_dimkt_qtest_{diff_level}.pkl"
        else:
            processed_data = file_path + folds_str + f"_dimkt_{diff_level}.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(self.sequence_path, skills_difficult_path, questions_difficult_path, folds)
                save_data = [self.dori,self.dqtest]
            else:
                self.dori = self.__load_data__(self.sequence_path, skills_difficult_path, questions_difficult_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Exists!! Read data from processed file: {processed_data}")
            if qtest:
                self.dori,self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}, "
             f"sdlen: {len(self.dori['sdseqs'])}, qdlen:{len(self.dori['qdseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
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
            - **qd_seqs (torch.tensor)**: question difficult sequence of the 0~seqlen-2 interactions
            - **sd_seqs (torch.tensor)**: knowledge concept difficult sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **qdshft_seqs (torch.tensor)**: question difficult sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **sdshft_seqs (torch.tensor)**: knowledge concept difficult sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        # 处理逻辑 masks/smaks:直接复制，len==0，origin/shft 复制，其他：前N-1，后N-1进行mask
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key], dcur["shft_"+key] = seqs, shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur,dqtest

    def __load_data__(self, sequence_path, skills_difficult_path, questions_difficult_path, folds, pad_val=-1):
        dori ={"qseqs":[],"cseqs":[],"rseqs":[],"tseqs":[],"utseqs":[],"smasks":[],
               "sdseqs":[],"qdseqs":[]}

        # 从csv文件中读取数据，并且只保留fold字段在folds列表中的行
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]  # 检查df["fold"]列中的每个值是否在folds列表中(0作为验证集)

        # 读取两个csv文件
        sds = {}
        qds = {}
        with open(skills_difficult_path, 'r', encoding="UTF8") as f:
            reader = csv.reader(f)
            sds_keys = next(reader)
            sds_vals = next(reader)
            for i in range(len(sds_keys)):
                sds[int(sds_keys[i])] = int(sds_vals[i])
        with open(questions_difficult_path, 'r', encoding="UTF8") as f:
            reader = csv.reader(f)
            qds_keys = next(reader)
            qds_vals = next(reader)
            for i in range(len(qds_keys)):
                qds[int(qds_keys[i])] = int(qds_vals[i])

        interaction_num = 0
        dqtest = {"qidxs": [], "rests": [], "orirow": []}
        sds_keys = [int(_) for _ in sds_keys]
        qds_keys = [int(_) for _ in qds_keys]
        # 遍历数据集中的每一行
        for i, row in df.iterrows():
            if "concepts" in self.input_type:
                tmp = [int(c) for c in row["concepts"].split(",")]
                tmp1 = []
                dori["cseqs"].append(tmp)  # （同dataloader）生成cseqs
                # (Add)生成sdseqs
                for j in tmp:
                    if j == -1:
                        tmp1.append(-1)
                    elif j not in sds_keys:
                        tmp1.append(1)
                    else:
                        tmp1.append(int(sds[j]))
                dori["sdseqs"].append(tmp1)
            if "questions" in self.input_type:
                tmp = [int(_) for _ in row["questions"].split(",")]
                tmp1 = []
                dori["qseqs"].append(tmp)
                # (Add)生成qdseqs
                for j in tmp:
                    if j == -1:
                        tmp1.append(-1)
                    elif j not in qds_keys:
                        tmp1.append(1)
                    else:
                        tmp1.append(int(qds[j]))
                dori["qdseqs"].append(tmp1)

            if "timestamps" in row:
                dori["tseqs"].append([int(t) for t in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(u) for u in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            # 计算选择遮罩序列中值为1的个数
            interaction_num = interaction_num + dori["smasks"][-1].count(1)
            # 如果启用了问题级别的评估，那么还需要添加评估用的序列
            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            if key not in ["rseqs"]:  # response序列转化为1.
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        # 计算遮罩序列
        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        # 计算选择遮罩序列
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")

        # 如果启用了问题级别的评估，那么还需要转换评估用的序列
        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]

            return dori, dqtest
        return dori
