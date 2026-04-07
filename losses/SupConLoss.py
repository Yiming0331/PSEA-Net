"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        # 温度系数t 控制模型对负样本的区分程度。值越小，模型越关注那些特别像正样本的负样本（hard negatives）
        self.temperature = temperature
        # contrast_mode: 对比模式。'all' 表示所有视图（views）都作为锚点（Anchor）去和其他视图对比；'one' 表示只用第一个视图做锚点。一般都用 'all'
        self.contrast_mode = contrast_mode
        # base_temperature: 用于最后缩放 Loss 的基准温度，通常等于 temperature。
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """


        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        """
        维度检查: 确保输入是 3 维。如果大于 3 维（比如 [bsz, 2, 128, 1, 1]),
        会展平最后几维，保证形状规范化为 [bsz, n_views, feature_dim]
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # 这是有监督对比学习的核心逻辑。我们需要知道哪些样本之间是“同类”的。
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        # SimCLR 模式: 如果没有 labels 也没有 mask，则退化为无监督模式
        # 这意味着：如果不考虑数据增强，
        # 每张图只和它自己是正样本(这只是初始 mask，后面会根据 views 扩展)。
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # SupCon 模式: 如果提供了 labels：
        elif labels is not None:
            # labels 变成列向量 [bsz, 1]
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # torch.eq(labels, labels.T): 利用广播机制生成一个 [bsz, bsz] 的布尔矩阵
            """
            含义: 如果样本 $i$ 和样本 $j$ 的标签相同，则位置 $(i, j)$ 为 1,
            否则为 0。这就是正样本对的指示矩阵。
            """
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 通常是 2 (因为有两个 crop)
        # unbind(dim=1) 将其拆分为两个 [batch_size, dim] 的张量。
        # cat(..., dim=0) 将它们在 batch 维度拼接。
        # 结果 contrast_feature 形状变成 [batch_size * 2, dim]
        # 现在，前 batch_size 个是第一个视图(crop 1)，后 batch_size 个是第二个视图(crop 2)。
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # 通常用 'all'。意味着所有的 2 * batch_size 个样本都会轮流做锚点（Anchor），去和其他样本对比。
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss