from datetime import datetime
import time


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

def change2timestamp(t):
    datetime_obj = datetime.strptime(t, "%Y/%m/%d %H:%M")
    timeStamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return timeStamp

def format_list2str(input_list):
    return [str(x) for x in input_list]

def write_text(file, stu_data):
    with open(file, "w") as f:
        for data in stu_data:
            for d in data:
                f.write(",".join(d) +"\n")