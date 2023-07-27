import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import one_hot

from pykt.config.config import que_type_models

device = "cpu" if not torch.cuda.is_available() else "cuda"


def save_cur_predict_result(dres, q, r, d, t, m, sm, p):
    # 初始化一个空列表用于保存结果
    results = []

    # 遍历每个数据样本
    for i in range(0, t.shape[0]):
        # 使用掩码选择预测结果和真实标签，并将它们转换为 CPU 张量
        cps = torch.masked_select(p[i], sm[i]).detach().cpu()
        cts = torch.masked_select(t[i], sm[i]).detach().cpu()

        # 使用掩码选择其他可能需要的信息（例如，问题和响应），并将它们转换为 CPU 张量
        cqs = torch.masked_select(q[i], m[i]).detach().cpu()
        crs = torch.masked_select(r[i], m[i]).detach().cpu()

        # 使用掩码选择其他可能需要的信息（例如，其他数据），并将它们转换为 CPU 张量
        cds = torch.masked_select(d[i], sm[i]).detach().cpu()

        # 初始化空列表用于保存转换为 Python 列表的结果
        qs, rs, ts, ps, ds = [], [], [], [], []

        # 转换问题和响应为 Python 列表
        for cq, cr in zip(cqs.int(), crs.int()):
            qs.append(cq.item())
            rs.append(cr.item())

        # 转换真实标签、预测得分和其他数据为 Python 列表
        for ct, cp, cd in zip(cts.int(), cps, cds.int()):
            ts.append(ct.item())
            ps.append(cp.item())
            ds.append(cd.item())

        # 计算 AUC，如果出现异常（例如，只有一个类别的标签），则设置 AUC 为 -1
        try:
            auc = metrics.roc_auc_score(y_true=np.array(ts), y_score=np.array(ps))
        except Exception as e:
            auc = -1

        # 计算准确率
        pre_labels = [1 if p>=0.5 else 0 for p in ps]
        acc =metrics.accuracy_score(ts, pre_labels)

        # 保存结果到字典和列表中
        dres[len(dres)] = [qs, rs, ds, ts, ps, pre_labels, auc, acc]
        results.append(str([qs, rs, ds, ts, ps, pre_labels, auc, acc]))

    return "\n".join(results)  # 返回结果列表的字符串形式，每个结果占一行


def evaluate(model, test_loader, model_name, rel=None, save_path=""):
    # 如果提供了保存路径，则打开一个文件以便写入结果
    if save_path !="":
        fout = open(save_path,"w",encoding="utf-8")
    # 禁用梯度计算，因为在评估模式下我们不需要计算梯度
    with torch.no_grad():
        y_trues = []  # 所有epoch的true label
        y_scores = []  # 所有epoch对predict score
        dres = dict()
        test_mini_index = 0 # # 记录当前处理的批次索引
        for test_data in test_loader: # 遍历测试数据加载器中的所有数据批次
            # 根据模型名称处理数据
            # 注意，这里有许多不同的模型，每种模型可能需要不同的数据处理方式
            # 这里的代码需要与模型的实现紧密配合
            if model_name in ["dkt_forget","bakt_time"]:
                dcur, dgaps = test_data
            else:
                dcur = test_data
            if model_name in ["dimkt"]:
                q, c, r, sd, qd = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["sdseqs"], dcur["qdseqs"]
                qshft, cshft, rshft, sdshft, qdshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur[
                    "shft_sdseqs"], dcur["shft_qdseqs"]
                sd, qd, sdshft, qdshft = sd.to(device), qd.to(device), sdshft.to(device), qdshft.to(device)
            else:
                q, c, r, = dcur['qseqs'], dcur['cseqs'], dcur['rseqs']
                qshft, cshft, rshft = dcur['shft_qseqs'], dcur['shft_cseqs'], dcur['shft_rseqs']
            m, sm = dcur["masks"], dcur["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), \
                qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)

            if model.model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
                model.model.eval()
            else:
                model.eval()

            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            if model_name in ['atdkt']:
                y = model(dcur)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["rkt"]:
                pass
            elif model_name in ["bakt_time"]:
                y = model(dcur, dgaps)
                y = y[:, 1:]
            elif model_name in ["dkt","dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt_forget"]:
                y = model(c.long(), r.long(), dgaps)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            else:
                pass
            # 如果给出了保存路径，调用 save_cur_predict_result 函数来保存预测结果
            if save_path!="":
                result = save_cur_predict_result(dres, c, r, cshft, rshft, m, sm, y)
                fout.write(result+"\n")
            # 使用掩码选择预测结果和真实标签，然后将它们转换为 numpy 数组并添加到列表中
            y = torch.masked_select(y, sm).detach().cpu()
            t = torch.masked_select(rshft, sm).detach.cpu()

            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            test_mini_index += 1 # 更新批次索引
        # 将所有批次的预测结果和真实标签连接在一起
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        # 计算性能指标，包括AUC和准确率
        auc = metrics.accuracy_score(y_trues=ts,y_scores=ps)
        pre_labels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, pre_labels)
    # 返回计算得到的性能指标
    return auc, acc