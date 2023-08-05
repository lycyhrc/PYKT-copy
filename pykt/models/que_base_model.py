import os

import numpy as np
import torch
from sklearn import metrics
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader


class QueBaseModel(Module):
    def __init__(self, model_name, emb_type, emb_path, pretrain_dim, device):
        super().__init__()
        self.model_name = model_name
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim
        self.device = device

    def compile(self, optimizer, lr=0.001, loss='binary_crossentropy', metrics=None):
        self.lr = lr
        self.opt = optimizer
        self.loss_func = self._get_loss_func(loss)

    def _get_loss_func(self, loss):
        """
        根据输入的字符串或者已定义的损失函数，返回对应的损失函数。

        参数:
        loss: str 或者 torch.nn.functional 中的函数。如果是字符串，应为以下之一："binary_crossentropy"，"mse"，"mae"。
              分别对应二分类交叉熵损失，均方误差损失，和平均绝对误差损失。
              如果已经是一个定义好的损失函数，直接返回这个函数。

        返回:
        loss_func: 对应的损失函数。

        异常:
        如果输入的字符串不在上述列表中，会抛出"NotImplementedError"错误。

        """
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss  # 这里可以传入一个函数名
        return loss_func

    def _get_optimizer(self, optimizer):
        """
        获取优化器。

        如果传入的优化器（optimizer）是一个字符串，那么此函数将会根据字符串的内容选择一个PyTorch的内置优化器，并将其实例化，学习率为self.lr。
        如果传入的优化器（optimizer）不是一个字符串，那么直接返回这个优化器。

        参数:
        optimizer (str或者torch.optim.Optimizer): 指定的优化器，可以是字符串('gd', 'adagrad', 'adadelta', 'adam')或者一个PyTorch优化器对象。

        返回:
        torch.optim.Optimizer: 模型使用的优化器

        异常:
        ValueError: 如果传入的字符串不在['gd', 'adagrad', 'adadelta', 'adam']这些选项中
        """
        if isinstance(optimizer, str):
            if optimizer == 'gd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            elif optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
            elif optimizer == 'adadelta':
                optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            else:
                raise ValueError("Unknown Optimizer: " + self.optimizer)
        return optimizer

    def train_one_step(self, data, process=True):
        """
        该方法定义了一个训练步骤，但并未在当前的类实现（子类中需要对其进行实现）。它是训练过程的基本组成部分，即处理输入的数据并完成一次训练迭代。

        参数:
        data (Any): 输入的数据，数据类型和形式取决于实际模型和任务需求。
        process (bool): 是否需要对数据进行预处理。默认为True。

        异常:
        NotImplementedError: 该方法在当前类中未被实现。
        """
        raise NotImplemented()

    def predict_one_step(self, data, process=True):
        raise NotImplemented()

    def batch_to_device(self, data, process=True):
        """
        此方法负责将数据批次(batch)处理后传送到设备上，例如GPU或CPU。如果需要对数据进行预处理，
        它将对数据进行特定的转换或修改，并将处理后的数据以字典的形式返回。如果不需要预处理，
        则直接返回原始的数据。

        参数:
        data (Dict): 包含了一批次训练数据的字典。字典的键是数据的名称，值是对应的数据。
        process (bool): 是否需要对数据进行预处理。默认为True。

        返回:
        data_new (Dict): 包含了预处理后的数据的字典。字典的键是数据的名称，值是对应的数据。
        """
        if not process:
            return data
        dcur = data
        # 数据预处理：创建新的数据字典，将原始数据的特定部分进行拼接操作，并赋值给新的字典
        # 通常用于序列类型的数据，如文本、时间序列等，通过拼接操作可以生成更丰富的序列特征
        dori = {}
        dori['cq'] = torch.cat((dcur["qseqs"][:,0:1], dcur["shft_qseqs"]),dim=1)
        dori['cc'] = torch.cat((dcur["cseqs"][:, 0:1], dcur["shft_cseqs"]), dim=1)
        dori['cr'] = torch.cat((dcur["rseqs"][:, 0:1], dcur["shft_rseqs"]), dim=1)
        dori['ct'] = torch.cat((dcur["tseqs"][:, 0:1], dcur["shft_tseqs"]), dim=1)
        dori['q'] = dcur["qseqs"]  # 前N-1
        dori['c'] = dcur["cseqs"]
        dori['r'] = dcur["rseqs"]
        dori['t'] = dcur["tseqs"]
        dori['qshft'] = dcur["shft_qseqs"] # 后N-1
        dori['cshft'] = dcur["shft_cseqs"]
        dori['rshft'] = dcur["shft_rseqs"]
        dori['tshft'] = dcur["shft_tseqs"]
        dori['m'] = dcur["masks"]
        dori['sm'] = dcur["smasks"]

        return dori

    def train(self,train_dataset, valid_dataset,batch_size=16,valid_batch_size=None,num_epochs=32, test_loader=None, test_window_loader=None,save_dir="tmp",save_model=False,patient=10,shuffle=True,process=True):
        """
        训练模型的函数，执行多轮次（epochs）的训练，并在每轮结束时评估模型在验证集上的性能。
        若模型在验证集上的表现优于之前所有轮次，且save_model为True，则保存模型。

        参数：
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        batch_size: 训练时的批次大小，默认为16
        valid_batch_size: 验证时的批次大小，默认为训练时的批次大小
        num_epochs: 训练的轮次数量，默认为32
        test_loader: 用于测试的数据加载器，默认为None
        test_window_loader: 用于时间窗口测试的数据加载器，默认为None
        save_dir: 保存模型的文件夹，默认为"tmp"
        save_model: 是否保存模型，默认为False
        patient: 在模型表现不再提升后，继续训练的轮次数量，默认为10
        shuffle: 是否在每轮训练开始时打乱训练数据的顺序，默认为True
        process: 是否在训练时对数据进行预处理，默认为True

        返回：
        在最优轮次下，模型在测试集、时间窗口测试集以及验证集上的AUC和准确度，以及最优轮次。
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir,exist_ok=True)

        if valid_batch_size is None:
            valid_batch_size = batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)

        max_auc,best_epoch = 0,-1
        train_step = 0

        for i in range(1, num_epochs+1):
            loss_mean =[]
            for data in train_loader:
                train_step += 1
                self.model.train()
                y, loss = self.train_one_step(data,process)
                self.opt.zero_grad()
                loss.backward()  # 计算梯度
                self.opt.step()  # 更新模型参数
                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)
            auc, acc = self.evaluate(valid_dataset,batch_size=valid_batch_size)
            print(f"eval_result is auc: {auc}, acc: {acc}")

            if auc > max_auc +1e-3:
                if save_model:
                    self._save_model()
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                validauc, validacc = auc, acc

            print(
                f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {self.model.emb_type}, model: {self.model.model_name}, save_dir: {self.save_dir}")
            print(
                f"            testauc: {round(testauc, 4)}, testacc: {round(testacc, 4)}, window_testauc: {round(window_testauc, 4)}, window_testacc: {round(window_testacc, 4)}")

            if i - best_epoch >= patient:
                break
        return testauc,testacc,window_testauc,window_testacc

    def evaluate(self, dataset, batch_size, acc_threshold=0.5):
        ps,ts = self.predict(dataset, batch_size=batch_size)
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        pre_lebles = [1 if p >= acc_threshold else 0 for p in ps]
        acc = metrics.accuracy_score(ts, pre_lebles)
        return auc,acc

    def predict(self, dataset, batch_size,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_scores = []
            for data in test_loader:
                dori = self.batch_to_device(data,process=process)
                y = self.predict_one_step(data)  # 在QDKT子类实现
                y = torch.masked_select(y, dori['sm']).detach().cpu()
                t = torch.masked_select(dori['rshft'],dori['sm']).detach().cpu()
                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        return ps, ts

    def get_loss(self,ys,rshft,sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft,sm)
        loss = self.loss_func(y_pred.double(),y_true.double())
        return loss