import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
# from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config.config import que_type_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name
    # 根据不同的模型类型计算损失函数
    if model_name in ["dkt", "sakt", "saint","dkt_forget", "dkvmn", "deep_irt", "kqn", "atkt", "atktfix", "gkt",
                      "skvmn", "hawkes"]:
        # 对于这些模型，使用二元交叉熵损失函数  mask_select保留sm中为true的元素
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

    elif model_name in ["atdkt", "simplekt", "bakt_time", "sparsekt"]:
        # 对于这些模型，使用二元交叉熵损失函数，并根据不同的嵌入类型计算整体损失
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1 * loss1 + model.l2 * ys[1] + model.l3 * ys[2]
            else:
                loss = model.l1 * loss1 + model.l2 * ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1 * loss1 + model.l2 * ys[1]
        else:
            loss = loss1
    elif model_name == "dkt+":
        # 对于dkt+模型，使用二元交叉熵损失函数，并添加一些附加的损失项
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(),
                                      r_curr.double())  # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    # 对于这些模型，使用二元交叉熵损失函数，并添加附加的正则化损失项
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos",
                        "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    # 对于lpkt模型，使用BCELoss损失函数，并对每个序列进行求和
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')
        loss = criterion(y, t).sum()

    return loss


def model_forward(model, data):
    model_name = model.model_name

    # 从data中提取输入数据
    if model_name in ["dkt_forget"]:
        dcur, dgaps = data
    else:
        dcur = data
    # 来自于 KTDataset中__getitem__方法返回的dcur数据
    q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
    m, sm = dcur["masks"], dcur["smasks"]

    ys, preloss = [], []
    # 对不同类型的模型进行前向传播
    cq = torch.cat((q[:, 0:1], qshft), dim=1)  # 生成新的序列，其中包含原始序列的第一元素，以及移位序列的所有元素
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:, 0:1], tshft), dim=1)
    if model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:, 0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())  # y：模型对学生在各个知识点的掌握程度的预测
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1) # 预测输出 y 和 one-hot 编码的向量进行元素级别的乘法，最后沿着最后一个维度进行求和
        ys.append(y)  # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["simplekt", "sparsekt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:, 1:], y2, y3]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn", "deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:, 1:])
    elif model_name in ["sakt", "kqn"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long()) # 原始序列第一元素，以及移位序列的所有元素
        ys.append(y[:, 1:]) # 除了第一个元素
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos",
                        "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:, 1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        # y, features = model(c.long(), r.long())
        # y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # loss = cal_loss(model, [y], r, rshft, sm)
        # # at
        # features_grad = grad(loss, features, retain_graph=True)
        # # p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        # # p_adv = Variable(p_adv).to(device)
        # pred_res, _ = model(c.long(), r.long(), p_adv)
        # # second loss
        # pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        # adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        # loss = loss + model.beta * adv_loss
        pass
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)
        # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())  # , csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models:
        y, loss = model.train_one_step(data)

    # 对于大多数模型，计算损失值并返回
    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss


def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None,
                test_window_loader=None, save_model=False):
    max_auc, best_epoch = 0, -1 # 初始化最大的AUC和最佳epoch数
    train_step = 0  # 记录训练步数
    # 对于LPKT模型，使用学习率调整策略
    if model.model_name == 'lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    # 进行num_epochs轮训练
    for i in range(1, num_epochs + 1):
        loss_mean = []  # 记录每个epoch的平均损失
        # 遍历训练集中的每个batch进行训练
        for data in train_loader:
            train_step += 1  # 更新训练步数
            if model.model_name in que_type_models:
                model.model.train()
            else:
                model.train()
            # 前向传播计算损失函数
            loss = model_forward(model, data)
            opt.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            opt.step()  # 更新模型参数

            loss_mean.append(loss.detach().cpu().numpy()) # 记录当前batch的损失值
            if model.model_name == "gkt" and train_step % 10 == 0: # 每10个步数打印一次训练信息（仅适用于GKT模型）
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text=text, fuc_name="train_model")
        if model.model_name == 'lpkt':
            scheduler.step()   # 更新学习率（仅适用于LPKT模型）
        loss_mean = np.mean(loss_mean)  # 计算当前epoch的平均损失
        # 在验证集上评估模型性能（计算AUC和准确率）
        auc, acc = evaluate(model, valid_loader, model.model_name)

        # 更新最大AUC和最佳epoch数
        if auc > max_auc + 1e-3:
            if save_model:
                # 保存性能最佳的模型参数
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + "_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                # 在没有保存模型的情况下，评估测试集和测试窗口集性能
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name,
                                                              save_test_path)
            validauc, validacc = auc, acc # 记录当前验证集上的AUC和准确率
        # 打印当前epoch的性能情况
        print(
            f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(
            f"            testauc: {round(testauc, 4)}, testacc: {round(testacc, 4)}, window_testauc: {round(window_testauc, 4)}, window_testacc: {round(window_testacc, 4)}")
        # 如果连续10个epoch没有性能提升，训练提前终止，避免过拟合
        if i - best_epoch >= 10:
            break
    # 返回各项性能指标和最佳epoch数
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch