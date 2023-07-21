import pandas as pd

from pykt.preprocess.split_datasets import read_data, get_max_concepts, calStatistics


def id_mapping_que(df):
    id_keys = ["questions","concepts","uid"]
    dres = dict() # 存储处理后的结果
    dkeyid2idx = dict() # 存储每个标签对唯一数值ID
    print(f"df.columns: {df.columns}")

    # 除了"id_keys"中的列以外的其他列直接复制到dres中
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]

    # 对 DataFrame df的每一行进行迭代。对于每一行数据，遍历"id_keys"中的每个键
    for i,row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            # 对于存在的键，首先确保这个键在`dkeyid2idx`和`dres`中都有对应的条目
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key,[])

            # 对于当前行的当前键对应的值，将其按照","分割，对于每个得到的标签，再将其按照"_"分割，得到子标签
            curids = []
            for id in row[key].split(","):
                sub_ids = id.split("_") # 子标签
                sub_curids = []

                # 如果某个子标签还没有被分配过数值 ID，那么就给它分配一个新的数值 ID
                for sub_id in sub_ids:
                    if sub_id not in dkeyid2idx[key]:
                        dkeyid2idx[key][sub_id] = len(dkeyid2idx[key]) # 如果某个子标签还没有被分配过数值 ID，那么就给它分配一个新的数值 ID，这个 ID 等于当前键对应的字典中已有的条目数量
                    sub_curids.append(str(dkeyid2idx[key][sub_id])) # 转换为字符串
                curids.append("_".join(sub_curids)) # 用"_"连接起来
            # 将处理后的标签添加到`dres`对应的键中
            dres[key].append(",".join(curids))
    # 在遍历完所有的行和键之后，将`dres`转换为 DataFrame
    finaldf = pd.DataFrame(dres)
    # 返回新的 DataFrame 和字典`dkeyid2idx`
    return finaldf,dkeyid2idx

def main(dname, fname, dataset_name, configf, min_seq_len=3, maxlen = 200, kfold=5):
    """

    :param dname: (str) data folder path
    :param fname: (str) the data file used to split, need 6 columns format is: (NA indicates the dataset has no corresponding info)
            uid,seqlen: 50121,4
            quetion ids: NA
            concept ids: 7014,7014,7014,7014
            responses: 0,1,1,1
            timestamps: NA
            cost times: NA
    :param dataset_name:(str) dataset name
    :param configf: (str) the dataconfig file path
    :param min_seq_len: (int, optional): the min seqlen,sequences less than this value will be filtered out. Defaults to 3.
    :param maxlen:(int, optional): the max seqlen. Defaults to 200.
    :param kfold:(int, optional): the folds num needs to split. Defaults to 5.
    :return:
    """
    stares = []
    total_df, effective_keys = read_data(fname)
    print(total_df)
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1
    print(max_concepts)

    oris,_,qs,cs,seqnum = calStatistics(total_df, stares, "original") # allselect 不考虑
    print("="*20)
    print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # just for id map
    total_df, dkeyid2idx = id_mapping_que(total_df)

