import os

import pandas as pd

from pykt.preprocess.utils import sta_infos, write_text

KEYS =['user_id','sequence_id']

def read_data_from_csv(readfile, write_file):
    stares = []
    df = pd.read_csv(readfile)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])
    df_zero_one = df[df['correct'].isin([0, 1])]   # 有效response值

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df_zero_one, KEYS, stares)
    print(
        f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    data = []
    ui_df = df_zero_one.groupby(['user_id'], sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1]

        # problem -> concept
        concepts = curdf["sequence_id"].astype(str).tolist()
        responses = curdf["correct"].astype(int).astype(str).tolist()
        timestamps = ["NA"]
        questions = ["NA"]
        usetimes = ["NA"]
        uids = [str(uid), str(len(responses))]
        data.append([uids, questions, concepts, responses, timestamps, usetimes])
        # print(data)
    write_text(write_file, data)

    print("\n".join(stares))

    return