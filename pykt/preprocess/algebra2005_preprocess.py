#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

KEYS = ["Anon Student Id", "KC(Default)", "Questions"]

def read_data_from_csv(read_file, write_file):
    stares= []

    df = pd.read_table(read_file, encoding = "utf-8", low_memory=False)
    df["Problem Name"] = df["Problem Name"].apply(replace_text)
    df["Step Name"] = df["Step Name"].apply(replace_text)
    df["Questions"] = df.apply(lambda x:f"{x['Problem Name']}----{x['Step Name']}",axis=1) # 将 "Problem Name" 和 "Step Name" 列合并为一个新的 "Questions" 列
 
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, '~~')
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    # 数据清洗：删除包含缺失值的行，过滤掉 "Correct First Attempt" 列中非 0 或 1 的响应，只保留需要的列，并对 "KC(Default)" 列进行预处理，将其中的 "" 替换为 "####"，"," 替换为 "@@@@"
    df["index"] = range(df.shape[0])
    df = df.dropna(subset=["Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"])
    df = df[df["Correct First Attempt"].isin([0,1])]
    df = df[["index", "Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"]]
    df["KC(Default)"] = df["KC(Default)"].apply(replace_text)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, "~~")
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    # 数据转换：按照 "Anon Student Id" 分组，对每个用户的数据进行排序，并将每个用户的问题序列、答案序列、开始时间序列转换为列表，然后将这些列表写入文本文件。
    data = []
    ui_df = df.groupby(['Anon Student Id'], sort=False)

    for ui in ui_df:
        u, curdf = ui[0], ui[1]
        curdf.loc[:, "First Transaction Time"] = curdf.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(t))  # 将 "First Transaction Time" 列的值从日期时间字符串转换为 Unix 时间戳。
        curdf = curdf.sort_values(by=["First Transaction Time", "index"])  # 按照 "First Transaction Time" 和 "index" 对数据进行排序。
        curdf["First Transaction Time"] = curdf["First Transaction Time"].astype(str)

        seq_skills = [x.replace("~~", "_") for x in curdf["KC(Default)"].values]  # 将 "KC(Default)" 列的值中的 "~~" 替换为 "_"，并将结果存储在 seq_skills 列表中。
        seq_ans = curdf["Correct First Attempt"].values
        seq_start_time = curdf["First Transaction Time"].values
        seq_problems = curdf["Questions"].values
        seq_len = len(seq_ans) #  列表的长度，即用户的答案数量
        seq_use_time = ["NA"]
        
        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        
        data.append(
            [[u, str(seq_len)], seq_problems, seq_skills, format_list2str(seq_ans), seq_start_time, seq_use_time])

    write_txt(write_file, data)

    print("\n".join(stares))

    return
