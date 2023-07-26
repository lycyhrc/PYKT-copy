import os,sys


def process_raw_data(dataset_name, dname2paths):
    """
        输入:
            dataset_name: string，需要处理的数据集的名称。
            dname2paths: dict，一个字典，键为数据集的名称，值为数据集的路径。

        功能:
            这个函数首先会获取输入数据集的路径，然后在该路径下生成一个新的文件"data.txt"。
            如果dataset_name是"xxx"，则会从相应的模块中导入处理函数。
            该函数会读取原始数据集并进行处理，然后将处理后的数据保存到"data.txt"中。

        输出:
            dname: string，处理后的数据集的目录路径。
            writef: string，处理后的数据写入的文件路径。
        """

    read_path = dname2paths[dataset_name]
    dname = "/".join(read_path.split("/")[0:-1]) #获取路径
    writef = os.path.join(dname, "data.txt") #在数据集对目录下生成txt文件
    print(writef)
    print(f" 开始处理数据：{dataset_name}")
    if dataset_name=="statics2011":
        from .statics2011_preprocess import read_data_from_csv
    elif dataset_name=="poj":
        from .poj_preprocess import read_data_from_csv
    else:
        pass
    #  数据集处理逻辑
    if dataset_name=="":
        pass
    read_data_from_csv(read_path,writef) # csv / txt

    return dname, writef