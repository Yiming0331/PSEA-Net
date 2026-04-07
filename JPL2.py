import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class JPLModule(nn.Module):
    """
    Proxy-Anchor (multi-proxy) + SoftTriple-style增强:
      (1) 类内多中心软分配：正项在 logsumexp 中叠加 log(w)
      (2) 中心去冗余正则：同类代理去相关、类间中心分离
    统一接口：
      - forward(...) -> {'proxy_loss': scalar tensor, 'class_similarities': [B, C]}
      - infer_logits(...)：代理相似度聚合成分类 logits(max/mean/logsumexp/soft)
    """

    def __init__(self,
                 feature_dim: int,
                 num_classes: int,
                 proxies_per_class: int = 10,
                 alpha: float = 32.0,
                 delta: float = 0.1,
                 lambda_soft: float = 0.1,
                 reg_intra_weight: float = 1e-3,
                 reg_inter_weight: float = 1e-3,
                 inter_margin: float = 0.1,
                 aggregate_negative_by_class: bool = False,
                 use_local_recon=False,
                 mask_ratio: float = 0.20,  # 默认 mask 比例
                 recon_temp: float = 1,   # 默认温度
                 recon_weight: float = 1.0, # 重建损失的权重
                 device=None):
        super().__init__()
        self.C = num_classes
        self.D = feature_dim
        self.U = proxies_per_class
        self.alpha = alpha
        self.delta = delta
        self.lambda_soft = lambda_soft
        self.reg_intra_weight = reg_intra_weight
        self.reg_inter_weight = reg_inter_weight
        self.inter_margin = inter_margin
        self.aggregate_negative_by_class = aggregate_negative_by_class

        self.use_local_recon = use_local_recon
                
        self.mask_ratio =mask_ratio
        self.recon_temp = recon_temp
        self.recon_weight =recon_weight

        # 代理参数 [C*U, D]，初始化到单位球
        self.proxies = nn.Parameter(torch.randn(self.C * self.U, self.D))
        with torch.no_grad():
            self.proxies.copy_(F.normalize(self.proxies, dim=1))

        # 代理所属类别 id: [0,0,1,1,2,2,...]
        class_ids = torch.arange(self.C).repeat_interleave(self.U)
        self.register_buffer("proxy_class_ids", class_ids.long())

        if use_local_recon:
            # 定义 encoder 模块：
            # 用于编码原始特征的变换器，由全连接层 + 层归一化 + ReLU 构成。
            # Reconstruction projectors
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
               
            )


        if device is not None:
            self.to(device)

    # -------------------------
    # 分类 logits（推理/评估用）
    # -------------------------
    

    # 利用部分东西进行计算时候打开
    
    def random_masking(self, x):
        """
        快速随机遮挡
        Returns:
            masked_features: 遮挡后的特征 (被遮挡位置为0)
            mask_indices: 遮挡掩码 (1表示被遮挡/需要计算Loss，0表示保留/无需计算Loss)
        """
        B, D = x.shape
        # 生成随机矩阵
        noise = torch.rand(B, D, device=x.device)
        
        # 1. 确定保留的 Mask (keep_mask): 大于 mask_ratio 的位置为 1 (保留)，否则为 0 (丢弃)
        # 例如 mask_ratio = 0.15，则约 85% 为 1 (保留作为上下文)
        keep_mask = (noise > self.mask_ratio).float()
        
        # 2. 确定需要计算 Loss 的 Mask (loss_mask): 刚好与 keep_mask 相反
        # 1 表示该位置被 Mask 了，需要计算重建 Loss
        loss_mask = 1.0 - keep_mask
        
        # 3. 生成遮挡后的特征 (输入给 Encoder 的)
        # 只有 keep_mask 为 1 的位置保留原值，其余位置置 0
        masked_features = x * keep_mask
        
        return masked_features, loss_mask

    # 利用全部东西进行计算时候打开

    # def random_masking(self, x):
    #     """
    #     快速随机遮挡，不关注具体遮了哪，只管生成遮挡后的特征
    #     """
    #     B, D = x.shape
    #     # 生成一个与 x 同形状的随机矩阵，值在 [0, 1)
    #     noise = torch.rand(B, D, device=x.device)
        
    #     # 大于 mask_ratio 的位置保留 (keep=1), 小于的置为 0
    #     # 例如 mask_ratio = 0.15，则约 85% 的位置是 1
    #     keep_mask = (noise > self.mask_ratio).float()
        
    #     # 遮挡特征
    #     masked_features = x * keep_mask
        
    #     return masked_features

    def local_reconstruction_loss(self, features, labels):
        """
        优化后的重建损失：
        1. 快速 Mask
        2. 代理加权 (Attention)
        3. 全局 MSE Loss (直接计算重建特征与原特征的距离)
        """
        B, D = features.shape

        # ---------------------------
        # 1. 快速 Mask (Input Corruption)
        # ---------------------------
        masked_features, loss_mask = self.random_masking(features)
        
        # ---------------------------
        # 2. Encoder 编码
        # ---------------------------
        encoded = self.encoder(masked_features)

        # ---------------------------
        # 3. Proxy Attention (加权代理)
        # ---------------------------
        # 取出当前 batch 对应的所有代理 [B, U, D]
        # self.proxies: [C*U, D] -> view [C, U, D] -> index [B, U, D]
        batch_proxies = self.proxies.view(self.C, self.U, self.D)[labels]

        # 计算 encoded 与 proxies 的相似度
        # encoded: [B, D] -> [B, 1, D]
        encoded_norm = F.normalize(encoded, dim=1).unsqueeze(1)
        # proxies: [B, U, D]
        proxies_norm = F.normalize(batch_proxies, dim=2)
        
        # bmm: [B, 1, D] @ [B, D, U] -> [B, 1, U] -> squeeze -> [B, U]
        sim = torch.bmm(encoded_norm, proxies_norm.transpose(1, 2)).squeeze(1)
        
        # Softmax 计算权重
        attn = F.softmax(sim / self.recon_temp, dim=1)  # [B, U]
        
        # 加权融合: sum( [B, U, 1] * [B, U, D] ) -> [B, D]
        combined = torch.sum(attn.unsqueeze(-1) * batch_proxies, dim=1)

        # ---------------------------
        # 4. Decoder 解码
        # ---------------------------
        # 尝试从 加权后的代理特征 恢复出 原始特征
        reconstructed = self.decoder(combined)

        # ---------------------------
        # 5. 全局 MSE Loss
        # ---------------------------
        # 直接计算重建特征和原始特征的均方误差
        # 这意味着模型必须学会用“代理组合”来完美逼近“原始特征”
        # loss = F.mse_loss(reconstructed, features)
        
        loss_all = (reconstructed - features) ** 2
        loss_masked = loss_all * loss_mask  # [B, D]
        
        # 方案 A: 保持原样 (Per Element Mean) - 数值较小
        # num_masked = loss_mask.sum() + 1e-6
        # loss = loss_masked.sum() / num_masked
        
        # 方案 B: 每个样本求和，再对 Batch 求平均 (Per Sample Mean) - 数值较大，推荐尝试
        # 1. 先把每个样本(B)内部的所有被遮挡特征误差加起来
        sample_loss = loss_masked.sum(dim=1) # [B]
        
        # 2. 统计每个样本分别被遮挡了多少个点 (为了归一化到样本尺度，可选)
        # 如果不除以这个，Loss 会随 D 的维度线性增长
        sample_mask_count = loss_mask.sum(dim=1) + 1e-6 # [B]
        
        # 3. 计算每个样本的平均误差
        per_sample_mse = sample_loss / sample_mask_count # [B]
        
        # 4. 最后对 Batch 取平均
        loss = per_sample_mse.mean()


        return loss


    def local_reconstruction_loss(self, features, labels):
        """
        MIM 风格的重建损失：
        1. 利用未被 Mask 的部分计算代理相似度 (Context -> Proxy)
        2. 只计算被 Mask 部分的 MSE 损失 (Prediction vs Ground Truth)
        """
        B, D = features.shape

        # ---------------------------
        # 1. 快速 Mask (获取特征 和 Loss掩码)
        # ---------------------------
        # masked_features: 大部分也是 0，只有保留部分有值
        # loss_mask: 1 表示该位置是“被遮挡的”，需要计算 Loss
        masked_features, loss_mask = self.random_masking(features)
        
        # ---------------------------
        # 2. Encoder 编码
        # ---------------------------
        # 这里的 encoded 是基于“可见部分”生成的特征表示
        encoded = self.encoder(masked_features)

        # ---------------------------
        # 3. Proxy Attention (利用可见部分寻找代理)
        # ---------------------------
        batch_proxies = self.proxies.view(self.C, self.U, self.D)[labels]

        # 归一化，准备计算余弦相似度
        # encoded 虽然是基于部分特征，但经过 Encoder 后已经是一个完整的 Latent 向量
        encoded_norm = F.normalize(encoded, dim=1).unsqueeze(1)
        proxies_norm = F.normalize(batch_proxies, dim=2)
        
        # 计算相似度
        # 逻辑：用“可见部分”的信息去匹配最接近的代理
        sim = torch.bmm(encoded_norm, proxies_norm.transpose(1, 2)).squeeze(1)
        
        # Softmax 计算权重
        attn = F.softmax(sim / self.recon_temp, dim=1)  # [B, U]
        
        # 加权融合得到目标特征 (Target Representation)
        combined = torch.sum(attn.unsqueeze(-1) * batch_proxies, dim=1)

        # ---------------------------
        # 4. Decoder 解码
        # ---------------------------
        # 尝试从代理特征恢复出原始特征
        reconstructed = self.decoder(combined)

        # ---------------------------
        # 5. Masked MSE Loss (只计算被遮挡部分)
        # ---------------------------
        # 计算所有位置的 MSE (不求平均，保持维度 [B, D])
        loss_all = (reconstructed - features) ** 2
        
        # 只保留 loss_mask 为 1 的位置 (即被遮挡的部分)
        loss_masked = loss_all * loss_mask
        
        # 计算平均 Loss
        # 分母：总共有多少个元素被 Mask 了 (加 epsilon 防止除零)
        num_masked = loss_mask.sum() + 1e-6
        loss = loss_masked.sum() / num_masked
        
        return loss


    def infer_logits(self,
                     features: torch.Tensor,
                     agg: str = "soft",
                     tau: float = 1.0):
        assert agg in {"max", "mean", "logsumexp", "soft"}

        # 归一化（保持与 features 同 dtype，避免不必要的 cast）
        x = F.normalize(features, dim=1)                     # [B, D]
        p = F.normalize(self.proxies.to(features.dtype), dim=1)  # [C*U, D]
        sim = x @ p.t()                                      # [B, C*U]


        #print("tau",tau)

        B = sim.size(0)
        device = sim.device
        logits = torch.empty(B, self.C, device=device, dtype=sim.dtype)

        for c in range(self.C):
            idx = (self.proxy_class_ids == c).nonzero(as_tuple=False).squeeze(1)
            s = sim[:, idx]  # [B, U_c]
            if s.numel() == 0:
                logits[:, c] = torch.finfo(sim.dtype).min
                continue

            if agg == "max":
                cls_score = s.max(dim=1).values
            elif agg == "mean":
                cls_score = s.mean(dim=1)
            elif agg == "logsumexp":
                cls_score = torch.logsumexp(tau * s, dim=1) / tau
            elif agg == "soft":
                # 原来对每个代理还加权重
                #w = torch.softmax(tau * s, dim=1)
                w = torch.softmax(s, dim=1)
                cls_score = (w * s).sum(dim=1)
            logits[:, c] = cls_score

        #print("logits除以tau之前",logits)

        logits = logits / tau

        #print("logits除以tau之后",logits)

        preds = logits.argmax(dim=1)
        return logits, preds
        
    # def infer_logits(self,
    #                 features: torch.Tensor,
    #                 agg: str = "soft",
    #                 tau: float = 1):  # agg 参数不再需要，因为逻辑固定了
        
    #     # 1. 归一化并计算 Cosine Similarity
    #     # features: [B, D]
    #     x = F.normalize(features, dim=1)
    #     # proxies: [Total_Proxies, D]
    #     p = F.normalize(self.proxies.to(features.dtype), dim=1)
        
    #     # sim: [B, Total_Proxies]
    #     # 这里的 sim 对应公式中的 -d (也就是相似度)
    #     sim = x @ p.t()

    #     # ------------------------------------------------------------------
    #     # 2. 实现 Eq (1): D(x, Pi)
    #     # 公式要求对所有代理(Pj ∈ Pt)进行 Softmax
    #     # 这计算的是 feature x 属于某个具体代理 Pi 的概率/权重
    #     # ------------------------------------------------------------------
    #     # [B, Total_Proxies]
    #     # 注意：公式中是 exp(-d/tau)，因为 d是距离，sim是相似度，sim ~ -d
    #     # 所以直接对 sim/tau 做 softmax 即可
    #     all_proxy_weights = torch.softmax(sim / tau, dim=1) 

    #     # ------------------------------------------------------------------
    #     # 3. 实现 Eq (2): p(y=c|x)
    #     # 将属于类别 c 的所有代理的权重(D(x, Pi))相加
    #     # ------------------------------------------------------------------
    #     B = sim.size(0)
    #     device = sim.device
        
    #     # 初始化类别概率矩阵 [B, C]
    #     probs = torch.zeros(B, self.C, device=device, dtype=sim.dtype)

    #     for c in range(self.C):
    #         # 找到属于类别 c 的所有代理的索引
    #         idx = (self.proxy_class_ids == c).nonzero(as_tuple=False).squeeze(1)
            
    #         if idx.numel() > 0:
    #             # 取出这些代理的权重并求和
    #             # sum(dim=1) 对应公式中的 Σ D(x, Pi)
    #             probs[:, c] = all_proxy_weights[:, idx].sum(dim=1)
    #         else:
    #             # 如果某类没有代理（极端情况），概率设为0
    #             probs[:, c] = 0.0

    #     # ------------------------------------------------------------------
    #     # 4. 返回处理
    #     # ------------------------------------------------------------------
    #     # probs 现在就是公式 (2) 的 p(y=c|x)，范围是 [0, 1]，且 sum(dim=1) = 1
        
    #     # 【重要】：
    #     # 如果你在外面使用 nn.CrossEntropyLoss，它内部会再做一次 Softmax。
    #     # 传入已经是概率的 probs 会导致错误。
    #     # 为了兼容标准的 PyTorch Loss，我们需要返回 "logits" (即 log 概率)。
    #     # log(probs) 之后，再经过 CrossEntropy 的 Softmax 依然能保持原本的概率分布趋势，
    #     # 或者你可以直接用 NLLLoss (Negative Log Likelihood Loss)。
        
    #     #print(probs)

    #     logits = torch.log(probs + 1e-12) # 加个极小值防止 log(0)

    #     #print(logits)

    #     preds = probs.argmax(dim=1)
        
    #     # 如果你想完全手动实现 Eq(3)，你需要的是 probs。
    #     # 但为了函数名 infer_logits 的准确性，这里返回 log 后的值。
    #     return logits, preds


    # -------------------------
    # 内部：Proxy-Anchor + 正则 的损失（强制 fp32，避免 AMP 下 dtype 冲突）
    # -------------------------
    def _compute_proxy_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, D = features.shape
        assert D == self.D and labels.shape == (B,)

        # 统一在 fp32 下计算，避免 Half/Float 混写
        with torch.cuda.amp.autocast(enabled=False):
            f32_feats   = features.float()
            f32_proxies = self.proxies.float()

            x = F.normalize(f32_feats,  dim=1)      # [B, D]
            p = F.normalize(f32_proxies, dim=1)     # [C*U, D]

            # 余弦相似度
            sim   = x @ p.t()                       # [B, C*U]
            sim_t = sim.t()                         # [C*U, B]

            # 正负样本 mask（bool）
            pos_mask = (labels.unsqueeze(0) == self.proxy_class_ids.unsqueeze(1))  # [C*U, B]
            neg_mask = ~pos_mask

            # 用全重打开
            # 类内 soft assignment 权重
            # sim_blocks = sim_t.view(self.C, self.U, B)              # [C, U, B]
            # w_blocks   = torch.softmax(self.lambda_soft * sim_blocks, dim=1)
            # w = w_blocks.view(self.C * self.U, B)                   # [C*U, B]

            # 正/负项 logits
            z_pos = -self.alpha * (sim_t - self.delta)
            z_neg =  self.alpha * (sim_t + self.delta)
            
            # 屏蔽无效项
            neg_inf = torch.finfo(z_pos.dtype).min
            z_pos = z_pos.masked_fill(~pos_mask, neg_inf)
            z_neg = z_neg.masked_fill(~neg_mask, neg_inf)

            # 用权重打卡
            #把 logw 加到对数域里（并显式转成同 dtype，防 mask 赋值报错）
            # eps  = 1e-12
            # logw = torch.log(torch.clamp(w, min=eps)).to(z_pos.dtype)
            # z_pos_weighted = z_pos.clone()
            # z_pos_weighted[pos_mask] = z_pos[pos_mask] + logw[pos_mask]

            # 如果用权重了就用这个
            #LSE -> Softplus
            # lse_pos = torch.logsumexp(z_pos_weighted, dim=1)
            # lse_neg = torch.logsumexp(z_neg, dim=1)
            # pos_term = F.softplus(lse_pos)
            # neg_term = F.softplus(lse_neg)

            # 如果没有用权重，就用这个
            # # LSE -> Softplus
            lse_pos = torch.logsumexp(z_pos, dim=1)
            lse_neg = torch.logsumexp(z_neg, dim=1)
            pos_term = F.softplus(lse_pos)
            neg_term = F.softplus(lse_neg)


            has_pos = pos_mask.any(dim=1)
            pos_loss = (pos_term[has_pos].sum() / has_pos.sum().clamp(min=1))
            neg_loss = neg_term.mean()
            
            pa_loss  = pos_loss + neg_loss


            

            # 正则：类内去相关、类间中心分离（同样用 fp32）
            reg_loss = 0.0
            # if self.reg_intra_weight > 0 and self.U > 1:
            #     P = F.normalize(f32_proxies, dim=1).view(self.C, self.U, self.D)
            #     cos_mat = torch.einsum('cud, cvd -> cuv', P, P)               # [C, U, U]
            #     off_diag = cos_mat - torch.eye(self.U, device=f32_feats.device).unsqueeze(0)
            #     reg_intra = (off_diag ** 2).mean()
            #     reg_loss += self.reg_intra_weight * reg_intra

            # if self.reg_inter_weight > 0:
            #     P = F.normalize(f32_proxies, dim=1).view(self.C, self.U, self.D)
            #     mean_centers = F.normalize(P.mean(dim=1), dim=1)              # [C, D]
            #     cos_cc = mean_centers @ mean_centers.t()
            #     mask = ~torch.eye(self.C, dtype=torch.bool, device=f32_feats.device)
            #     reg_inter = F.relu(cos_cc[mask] - self.inter_margin).mean()
            #     reg_loss += self.reg_inter_weight * reg_inter

            loss = pa_loss + reg_loss

        # 若在 AMP 下，返回的 loss 保持与 features 同 dtype（通常是 fp16/bf16 标量）
        return loss.to(features.dtype)

    # -------------------------
    # 公共前向：返回 dict（训练/推理统一）
    # -------------------------
    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor | None = None,
                *,
                agg: str = "soft",
                tau: float = 1):
        """
        Args:
          features : [B, D]
          labels   : [B] 或 None (None 时不计算 proxy_loss)
          agg, tau : 分类聚合方式与温度，传给 infer_logits
        Returns:
          {
            'proxy_loss': scalar tensor(labels=None 时为 0.0,同设备/同 dtype),
            'class_similarities': [B, C]
          }
        """


        #print("self.U", self.U)
        logits, _ = self.infer_logits(features, agg=agg, tau=tau)

        if labels is not None:
            proxy_loss = self._compute_proxy_loss(features, labels)
            if self.use_local_recon:
                # 不需要手动传 self
                reconloss = self.local_reconstruction_loss(features, labels)
                proxy_loss = proxy_loss + reconloss
        else:
            proxy_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)

        return {'class_similarities': logits, 'proxy_loss': proxy_loss}

