# _*_ coding:utf-8 _*_

import pandas as pd
from .utils import sta_infos, write_txt,format_list2str,change2timestamp
#ref https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect

KEYS = ["user_id", "skill_id", "problem_id"]

def read_data_from_csv(read_file, write_file):
    stares = []

    # load data  只选择了一部分列进行处理
    df = pd.read_csv(read_file, low_memory=False, usecols=[
                 "user_id", "skill_id", "start_time", "problem_id", "correct","ms_first_response"])
    df['correct'] = df['correct'].apply(int)
    # 使用 sta_infos 函数统计原始数据的一些信息，如交互数量、用户数量、问题数量、概念数量等，并打印出来
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    # 它对数据进行预处理，包括删除缺失值、过滤掉非 0 或 1 的响应，并添加时间戳
    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=["user_id", "skill_id", "start_time","problem_id", "correct","ms_first_response"])
    df = df[df['correct'].isin([0,1])]#filter responses

    # add timestamp and duration
    df['start_timestamp'] = df['start_time'].apply(lambda x:change2timestamp(x,hasf='.' in x))

    # 预处理后，它再次使用 sta_infos 函数统计数据的信息
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    # 按用户分组，对每个用户的数据进行排序，并将每个用户的问题序列、答案序列、响应时间序列、开始时间序列和问题序列转换为列表，然后将这些列表写入文本文件
    user_inters = []
    for user, group in df.groupby(['user_id'], sort=False):
        group = group.sort_values(['start_timestamp','tmp_index'])
        seq_skills = group['skill_id'].tolist()
        seq_ans = group['correct'].tolist()
        seq_response_cost = group['ms_first_response'].tolist()
        seq_start_time = group['start_timestamp'].tolist()
        seq_problems = group['problem_id'].tolist()
        seq_len = len(group)
        user_inters.append(
            [[str(user), str(seq_len)],
            format_list2str(seq_problems),
            format_list2str(seq_skills),
            format_list2str(seq_ans),
            format_list2str(seq_start_time),
            format_list2str(seq_response_cost)])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

