import argparse
import sys
sys.path.append('/Users/youh/PycharmProjects/PYKT_copy')
from pykt.preprocess.data_proprocess import process_raw_data


# 定义数据存储路径
dname2paths = {
    "statics2011":"../data/statics2011/AllData_student_step_2011F.csv"
}

# 项目的数据配置json文件
config_data = '../configs/data_config.json'

# 接受输入参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="statics2011")
    parser.add_argument("-f", "--file_path", type=str, default="../data/statics2011/AllData_student_step_2011F.csv")
    parser.add_argument("-m", "--min_seq_len", type=int, default=3)
    parser.add_argument("-l", "--maxlen", type=int, default=200)
    parser.add_argument("-k", "--kfold", type=int, default=5) # -k命令行用 -kfold 命令行/参数调用
    args = parser.parse_args()
    print(args)

    # otherdata_process_raw_data
    # if args.dataset_name=="peiyou":
    #     dname2paths["peiyou"] = args.file_path
    #     print(f"path: {args.file_path}")

    # data_process logic
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-" * 50)
    print(f"dname: {dname}, writef: {writef}")