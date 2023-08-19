import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HawkesKT(nn.Module):
    def __init__(self,n_skills,n__problems,emb_size,time_log,emb_type="qid"):
        super(HawkesKT, self).__init__()
        # 初始化模型参数
        self.model_name = "hawkes"
        self.n_skills = n_skills
        self.n_problems = n__problems
        self.emb_size = emb_size
        self.time_log = time_log
        self.emb_type = emb_type
    
        # base intensity:问题和对应概念的初始参数化表示 每个问题生成一个base intensity
        self.problem_base = nn.Embedding(self.n_problems, 1) # 
        self.skill_base = nn.Embedding(self.n_skills, 1)

        # mutual excitation 和 kernel function
        self.alpha_inter_embeddings = nn.Embedding(2*self.n_skills,self.emb_size)
        self.alpha_skill_embeddings = nn.Embedding(self.n_skills, self.emb_size)
        self.beta_inter_embeddings = nn.Embedding(2*self.n_skills,self.emb_size)
        self.beta_skill_embeddings = nn.Embedding(self.n_skills, self.emb_size)
    
    @staticmethod
    def init_weights(m):
        """
        这个静态方法用于初始化模型的权重。

        如果传入的模块 `m` 是一个嵌入层（`torch.nn.Embedding`），那么这个方法会使用均值为0.0，标准差为0.01的正态分布来初始化嵌入层的权重。

        参数:
        m (torch.nn.Module): 需要初始化权重的模块。

        """
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        
        
    def forward(self, skills,problems,times,labels,qtest=False):
        # 定义前向传播过程
        mask_labels = labels
        inters = skills + mask_labels * self.n_skills  # 计算交互 xemb = c+ num_c*r

        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [Bs,sl,es]
        alpha_target_emb = self.alpha_skill_embeddings(skills) # [Bs,sl,es]
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2,-1))  # [Bs,sl,sl]

        beta_src_emb = self.beta_inter_embeddings(inters)  # [Bs,sl,es]
        beta_target_emb = self.beta_skill_embeddings(skills) # [Bs,sl,es]
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2,-1))  # [Bs,sl,sl]
        
        # 核函数计算：times变量被用于计算时间差delta_t，这个时间差随后被用于计算交叉效应 cross_effects
        if times.shape[1] > 0:
            times = times.double() / 1000
            delta_t = (times[:, :, None] - times[:, None, :]).abs().double() # 计算 times 张量中的每两个元素之间的差的绝对值，并将结果转换为双精度浮点数
        else:
            # 1 if no timestamps
            delta_t = torch.ones(skills.shape[0], skills.shape[1], skills.shape[1]).double().to(device) # [bs,sl,sl] 元素全为1的张量，将其转换为双精度浮点数
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)  # 将 delta_t 张量中的每个元素加上一个很小的数（防止取对数时出现无穷大），然后取对数，最后除以 self.time_log 的对数

        betas = torch.clamp(betas + 1, min=0, max=10)  # 将 betas 张量中的每个元素加1，然后使用 torch.clamp 函数将结果限制在0到10之间
        cross_effects = alphas * torch.exp(-betas * delta_t) # [bs,sl,sl]

        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0) # [1,sl,sl]上三角矩阵掩码
        mask = mask.cuda() if torch.cuda.is_available() else mask
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2) # [bs,sl]

        problem_bias = self.problem_base(problems).squeeze(dim=-1) # [bs,sl]
        skill_bias = self.skill_base(skills).squeeze(dim=-1) # [bs,sl]
        h = problem_bias + skill_bias + sum_t # [bs,sl]
        prediction = h.sigmoid() # [bs,sl]
        if not qtest:
            return prediction
        else:
            return prediction,h


