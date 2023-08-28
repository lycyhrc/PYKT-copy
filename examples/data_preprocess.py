import argparse
import sys
sys.path.append('/Users/youh/PycharmProjects/PYKT_copy')

from pykt.preprocess.data_proprocess import process_raw_data
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question

# 定义数据存储路径
dname2paths = {
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "poj": "../data/poj/poj_log.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    # "junyi2015": "../data/junyi2015/junyi_ProblemLog_original.csv"
    "assist2012":"../data/assist2012/2012-2013-data-with-predictions-4-final.csv"
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
    parser.add_argument("-k", "--kfold", type=int, default=5)  # -k命令行用 -kfold 命令行/参数调用
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

    # for concept level model
    split_concept(dname, writef, args.dataset_name, config_data, args.min_seq_len, args.maxlen, args.kfold)
    print("="*100)

    # for question level model
    split_question(dname, writef, args.dataset_name, config_data, args.min_seq_len, args.maxlen, args.kfold)
