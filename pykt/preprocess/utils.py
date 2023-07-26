import time
import datetime

def sta_infos(df, keys, stares, split_str="_"):
    """
       输入:
           df: DataFrame，需要进行统计的数据集。
           keys: list，用于提取唯一值的列名列表。
           stares: list，用于存储统计结果的列表。
           split_str: string，用于连接统计结果的分隔符，默认为"_"。

       功能:
           该函数统计df中特定列的唯一值数量，然后计算平均信息数量，最后把统计结果添加到stares列表中。
           默认只处理keys列表中的前两个键，如果keys的长度大于2，则不做处理。

       输出:
           ins: int，数据集中的总记录数。
           us: int，数据集中keys[0]列的唯一值数量。
           qs: string，固定输出为"NA"。
           cs: int，数据集中keys[1]列的唯一值数量（如果存在的话）。
           avgcqf: string，固定输出为"NA"。
           naf: string，固定输出为"NA"。
           avgins: float，平均信息数量，等于总记录数除以keys[0]列的唯一值数量。
    """
    # 分为concept和question
    uids = df[keys[0]].unique() # 返回该列中的唯一值数组
    if len(keys) == 2: #
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        pass
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs, avgcqf, naf = df.shape[0], len(uids),"NA",len(cids),"NA","NA" # 只有概念
    if len(keys)>2:
        pass
    curr = [ins, us, qs, cs, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))

    return ins, us, qs, cs, avgins, avgcqf, naf

def replace_text(text):
    text = text.replace("_","####").replace(",","@@@@")
    return text

from datetime import datetime
def change2timestamp(t,hasf = True):
    """
       这个函数将日期时间的字符串表示形式转换为Unix时间戳（毫秒）。

       参数:
       t (str): 日期时间的字符串表示形式。
       hasf (bool): 一个标志，表示日期时间字符串是否包含小数秒。
                    默认值为True，表示函数预期会有小数秒。

       返回值:
       float: 代表输入日期时间的Unix时间戳（毫秒）。
       """
    if hasf:
        try:
            timeStamp = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
        except ValueError:
            timeStamp = datetime.datetime.strptime(t, "%Y/%m/%d %H:%M").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y/%m/%d %H:%M").timestamp() * 1000
    return timeStamp

def format_list2str(input_list):
    return [str(x) for x in input_list]

def write_text(file, stu_data):
    with open(file, "w") as f:
        for data in stu_data:
            for d in data:
                f.write(",".join(d) +"\n")

def set_seed(seed):
    """Set the global random seed.

        Args:
            seed (int): random seed
    """
    try:
        import  torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        print("Set seed failed, detail are ",e)
        pass

    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import datetime
def get_now_time():
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string

def debug_print(text, fuc_name=""):
    print(f"{get_now_time()} - {fuc_name} - said: {text}")
