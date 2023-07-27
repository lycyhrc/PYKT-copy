import os, sys
import torch
import torch.nn as nn
from torch.autograd import grad, Variable
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import pandas as pd
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from torch.autograd import Variable, grad
import pandas as pd
from pykt.config.config import que_type_models
from pykt.preprocess.utils import debug_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    # 获取模型名字
    model_name = model.model_name

    if model_name in ["atdkt", "simplekt", "bakt_time", "sparsekt"]:
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
    # cal_loss(dkt)
    elif model_name in ["rkt", "dimkt", "dkt", "dkt_forget", "dkvmn", "deep_irt", "kqn", "sakt", "saint", "atkt",
                        "atktfix", "gkt", "skvmn", "hawkes"]:
        # 通过遮盖器选择模型输出和移位后的响应
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # 计算二元交叉熵损失
        loss = binary_cross_entropy(y.double(), t.double())

    elif model_name == "dkt+":
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
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos",
                        "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')
        loss = criterion(y, t).sum()

    return loss

def model_forward(model, data, rel=None):
    model_name = model.model_name
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t, sd, qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur[
            "tseqs"].to(device), dcur["sdseqs"].to(device), dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft, sdshft, qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur[
            "shft_rseqs"].to(device), dcur["shft_tseqs"].to(device), dcur["shft_sdseqs"].to(device), dcur[
            "shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(
            device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur[
            "shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1],qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:, 0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:, 1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3]  # first: yshft
    elif model_name in ["simplekt", "sparsekt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:, 1:], y2, y3]
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:, 1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:, 0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn", "deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:, 1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos",
                        "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:, 1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss

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
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y, loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(), c.long(), sd.long(), qd.long(), r.long(), qshft.long(), cshft.long(), sdshft.long(),
                  qdshft.long())
        ys.append(y)

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss
def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0
    # 根据模型的名字（model.model_name）做一些特定的准备工作
    rel = None
    if model.model_name =="rkt":
        pass
    if model.model_name =="lpkt":
        pass

    for epoch in range(1, num_epochs + 1): # 循环执行num_epochs个训练周期，每个周期内遍历数据
        loss_mean = []
        for data in train_loader:
            train_step += 1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            #【模型的前向计算，输出通常是损失值】
            if model.model_name == 'rkt':
                # loss =
                pass
            else:
                loss = model_forward(model,data)  # 每次拿出一批数据送入模型进行前向计算（model_forward()），得到当前批次的损失值
            opt.zero_grad() # 将模型的梯度清零
            loss.backward() # 【计算损失的梯度】
            #【模型参数的更新】
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip) # 使用clip_grad_norm_()对梯度进行裁剪以防止梯度爆炸
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy()) # 把每一步训练的损失值从张量转换为NumPy数组，添加到loss_mean
            # if model.model_name == "gkt" and train_step % 10 == 0:
            #     text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
            #     debug_print(text=text, fuc_name="train_model")
        if model.model_name == 'lpkt':
            # scheduler.step()  # update each epoch
            pass
        loss_mean = np.mean(loss_mean) # 一个训练周期中所有epoch的平均损失
        #【模型在验证集上的评估】
        if model.model_name == 'rkt':
            pass
        else:
            auc,acc = evaluate(model, valid_loader, model.model_name) # 训练每个周期结束后，都会在验证数据集上评估模型的性能
        # 并根据模型在验证集上的表现来决定是否要保存模型的参数
        if auc > max_auc + 1e-3:
            # 【模型参数的保存】
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = epoch

            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                    testauc, testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
        validauc, validacc = auc, acc
        print(f"Epoch: {epoch}, validauc:{validauc:.4}, validacc:{validacc:.4},"
              f" best epoch: {best_epoch}, best auc:{max_auc:.4}, "
              f"train loss:{loss_mean},emb_type:{model.emb_type}, model:{model.model_name}, save_dir:{ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, "
              f"window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")

        if epoch - best_epoch >= 10: # 如果当前的周期数与最佳周期的差值大于等于10，那么训练将提前终止，否则继续下一个周期的训练
            break




