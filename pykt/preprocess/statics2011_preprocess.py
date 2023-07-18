from pandas import read_csv

from pykt.preprocess.utils import replace_text, sta_infos, change2timestamp, format_list2str, write_text

KEYS = ["Anon Student Id", "KC"]


def read_data_from_csv(read_path, write_file):
    stares = []
    print(f"{read_path} {write_file}")
    df = read_csv(read_path)

    # 数据格式处理
    df['Problem Name'] = df['Problem Name'].apply(replace_text)
    df['Step Name'] = df['Step Name'].apply(replace_text)
    df["KC"] = df.apply(lambda x: "{}----{}".format(x["Problem Name"], x["Step Name"]), axis=1)

    # 数据统计信息
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"Original interaction num: {ins},User num: {us}, Concept num: {cs}, Question num: {qs}, Avg(ins) per s: {avgins}, Avg(c) per q: {avgcq}, na: {na}")

    # 数据清洗(删除缺失值)
    df['tmp_index'] = range(len(df))  #添加一行，序列从0开始
    df = df.dropna(subset=['Problem Name','Step Name','First Transaction Time','First Attempt'])
    df = df[df['First Attempt']!='hint']
    # 数据清洗(results数值化)
    _df = df.copy()
    _df.loc[df["First Attempt"]=="correct","First Attempt"] = str(1)
    _df.loc[df["First Attempt"]=="incorrect","First Attempt"] = str(0)
    _df.loc[:,"First Transaction Time"] = _df.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(t))

    # 数据统计信息
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(
        f"After interaction num: {ins},User num: {us}, Concept num: {cs}, Question num: {qs}, Avg(ins) per s: {avgins}, Avg(c) per q: {avgcq}, na: {na}")

    #写入文件格式
    user_inters = []
    for group_name, group_data in _df.groupby("Anon Student Id"): # student num

        group_data = group_data.sort_values(by=['First Transaction Time',"tmp_index"])
        group_data['First Transaction Time'] = group_data['First Transaction Time'].astype(str)

        seq_skills = group_data["KC"].values
        seq_ans = group_data["First Attempt"].values
        seq_start_time = group_data["First Transaction Time"].values
        seq_len = len(seq_ans)
        seq_problems = ["NA"]
        seq_use_time = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans) ==  len(seq_start_time)
        # ID 几条数据
        user_inters.append([[group_name, str(seq_len)], seq_problems, seq_skills, format_list2str(seq_ans), seq_start_time, seq_use_time])

    write_text(write_file, user_inters)

    print("\n".join(stares))
    return