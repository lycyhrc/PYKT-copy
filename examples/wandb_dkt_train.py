import argparse

from wandb_train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default="statics2011")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str,default="qid")  # 默认qid类型
    parser.add_argument("--save_dir",type=str,default="save_model")  # 允许用户指定一个目录路径，模型和相关文件将在该目录路径中保存。
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold",type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--emb_size",type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    params = vars(args)
    main(params)
