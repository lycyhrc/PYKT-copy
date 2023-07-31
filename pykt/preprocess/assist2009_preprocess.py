import pandas as pd

from pykt.preprocess.utils import sta_infos, format_list2str, write_text

KEYS =['user_id','skill_id','problem_id']
def read_data_from_csv(readfile, write_file):
    stares = []
    df = pd.read_csv(readfile, encoding='utf-8', dtype=str)  # 这里要加上dtype

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])
    df_drop_null = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct", "order_id"])
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df_drop_null, KEYS, stares)
    print(
        f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    ui_df = df_drop_null.groupby(['user_id'], sort=False)

    user_datas = []
    for ui in ui_df:
        uid, curdf = int(ui[0][0]), ui[1]
        tmp_inter = curdf.sort_values(by=['order_id', 'index'])
        seq_problems = tmp_inter["problem_id"].tolist()
        seq_skills = tmp_inter["skill_id"].tolist()
        seq_ans = tmp_inter["correct"].tolist()
        seq_start_time = ['NA']
        seq_response_cost = ['NA']

        user_datas.append(
            [[str(uid), str(len(tmp_inter))], format_list2str(seq_problems), format_list2str(seq_skills),
             format_list2str(seq_ans), seq_start_time, seq_response_cost])

    write_text(write_file, user_datas)

    print("\n".join(stares))

    return