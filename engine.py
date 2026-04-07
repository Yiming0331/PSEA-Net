# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
from losses.SupConLoss import SupConLoss
from typing import Iterable, Optional
import numpy as np
from einops import rearrange
import torch
import numpy
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Iterable, Optional, Union, List
import torch.nn.functional as F


#     return interpolated

def slerp(A: torch.Tensor, B: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """
    Spherical linear interpolation between two batched points A and B on a unit hypersphere.

    Parameters:
    - A: First set of points, shape (batch_size, d). 形状为 (batch_size, d) 的张量，表示一组起始向量（已批量化）。
    - B: Second set of points, shape (batch_size, d). 形状同 A表示一组目标向量
    - t: 
    插值参数，范围通常为 [0, 1]，可以是一个标量或形如 (batch_size, 1) 的张量。t=0 表示完全是 A,t=1 表示完全是 B。
    Interpolation parameter in range [0, 1], shape (batch_size, 1) or single value.

    Returns:
    - torch.Tensor: Interpolated points, shape (batch_size, d).
    """
    # Ensure inputs are unit vectors
    # 确保输入向量 A 和 B 都是单位向量，SLERP 要求输入在单位超球面上。
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)

    # Compute dot product for each pair of points
    # 对 A 和 B 每对向量，计算点积（也就是余弦相似度）。
    # clamp 是为了防止 acos 运算中出现超出定义域的数值误差。
    dot = torch.sum(A * B, dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)  # Avoid numerical issues

    # Compute the angle for each pair
    # 得到每对向量在单位球面上的夹角 θ，单位为弧度。
    theta = torch.acos(dot)

    # Slerp formula
    # coeff_a 和 coeff_b 是插值时 A 和 B 的权重系数。
    sin_theta = torch.sin(theta)
    t_theta = t * theta
    # t=0 时，coeff_a = 1, coeff_b = 0；t=1 时，coeff_a = 0, coeff_b = 1
    coeff_a = torch.sin(theta - t_theta) / sin_theta
    coeff_b = torch.sin(t_theta) / sin_theta

    # Compute the interpolated points
    # 最终返回的是插值向量，仍位于单位球面上。
    interpolated = coeff_a * A + coeff_b * B

    return interpolated




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, num_cilps:int, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    contrastive_nomixup=False, hard_contrastive=False,
                    finetune=False
                    ):
    # TODO fix this for finetuning
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    #criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        batch_size = targets.size(0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #print("targets",targets)

        if mixup_fn is not None:
            # batch size has to be an even number
            if batch_size == 1:
                continue
            if batch_size % 2 != 0:
                    samples, targets = samples[:-1], targets[:-1]
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast(device_type='cuda'):
            
            outputs = model(samples,targets=targets)
            
            #视频级别时用这个
            #output = outputs[0].reshape(batch_size, num_cilps, -1).mean(dim=1) 
            
            #print("barloss",barloss)
            # 帧级别用缩略图时候用这个
            targets = targets.repeat_interleave(num_cilps)
            
            output = outputs[0]

            # print("output.shape",output.shape)
            # print("targets.shape",targets)


            jpl_loss = outputs[2]
            lce = criterion(output, targets)
            #print("loss",loss)
            loss = jpl_loss+ lce

        loss_value = loss.item()

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, world_size, distributed=True, amp=False, num_crops=1, num_clips=1):
    criterion = torch.nn.CrossEntropyLoss()
    to_np = lambda x: x.data.cpu().numpy()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    logits = []
    binary_label = []
    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        #print(target)
        # compute output
        batch_size = images.shape[0]

        with torch.amp.autocast(device_type='cuda'):

            output = model(images)

        output = output[0].reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        
        # print(output)
        
        
        output_np = to_np(output[:,1])

        
        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
            output_ = concat_all_gather(output)
            target_ = concat_all_gather(target)
            output_np_ = to_np(output_[:,1])
            logits.append(output_np_)
            binary_label.append(target_.detach().cpu())
        else:
            outputs.append(output)
            targets.append(target)
            logits.append(output_np)
            binary_label.append(target.detach().cpu())
        batch_size = images.shape[0]

        acc1 = accuracy(output, target, topk=(1,))[0]
        metric_logger.meters['acc1'].update(acc1.item(), images.size(0))

    # import pdb;pdb.set_trace()

    acc_outputs = numpy.stack(logits,0).reshape(-1,1)
    acc_label = numpy.stack(binary_label,0).reshape(-1,1)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    auc_score = roc_auc_score(acc_label, acc_outputs)    
    
    #=== EER 计算 ===
    fpr, tpr, thresholds = roc_curve(acc_label, acc_outputs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())

    # print('* Acc@1 {top1.global_avg:.3f} AUC {auc} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1,auc=auc_score,losses=metric_logger.loss))

    print('* Acc@1 {top1.global_avg:.3f} AUC {auc} EER {eer} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, auc=auc_score, eer=eer, losses=metric_logger.loss))
          
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}






def do_TTT(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, num_cilps:int, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    contrastive_nomixup=False, hard_contrastive=False,
                    finetune=False
                    ):
    # TODO fix this for finetuning

    # 检查模型是否允许更新梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Param {name} requires grad: True")
            break
    else:
        print("ALL PARAMETERS ARE FROZEN (requires_grad=False)!")


    if finetune:
        model.train(not finetune)
    else:
        model.train()
    criterion_supcon = SupConLoss().to(device)

    tau = 0.45

    #criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        

        samples = samples.to(device, non_blocking=True)

        #print("targets",targets)


        with torch.amp.autocast(device_type='cuda'):
            
            outputs = model(samples)
            # 原始jpl损失
            loss1 = outputs[2]

            # 相似度
            confident_logits = outputs[0]
            #print("raw_logits",raw_logits)

            # 原始得特征
            confident_features = outputs[1]
            #barloss = outputs[1]
            #print("barloss",barloss)


            # # 计算熵
            
            confident_probs = torch.softmax(confident_logits, dim=1)
            # #print("probs",probs)


            # 如果不筛选给下面注释调即可 436-449
            entropy = -(confident_probs * torch.log(confident_probs + 1e-6)).sum(dim=1)

            # 计算当前 batch entropy 的 45% 分位数
            # torch.quantile 需要 float 类型
            threshold_lambda = torch.quantile(entropy, tau)

            # === 公式 (21): Confident Set M ===
            # 筛选 entropy 小于 lambda 的样本
            mask_M = entropy < threshold_lambda

            # 提取自信样本
            confident_logits = confident_logits[mask_M]
            confident_features = confident_features[mask_M]
            confident_probs = confident_probs[mask_M]



            # 如果没有样本满足条件 (极端情况)，跳过更新
            if confident_logits.shape[0] < 2:
                continue
            

            # === 生成伪标签 (Pseudo-label) ===
            # y_hat = argmax pi(c)
            pseudo_labels = torch.argmax(confident_probs, dim=1)
            #print("pseudo_labels",pseudo_labels)

            confident_features, pseudo_labels = apply_slerp_augmentation(
                confident_features, 
                pseudo_labels, 
                t_range=(0.2, 0.8),
                target_classes= 1  # <--- 指定只增强 1   
            )


            # === 公式 (22): L_TTT ===
            # L_ce: Cross Entropy with Pseudo labels
            # 注意: CE Loss 输入通常是 logits，这里我们需要带温度的 logits
            #loss_ce = criterion_ce(confident_logits / T_scale, pseudo_labels)
                
            # L_info: InfoNCE among confident samples
            #loss_info = info_nce_loss(confident_features, pseudo_labels)
            confident_features = confident_features.unsqueeze(1)
            loss_info = criterion_supcon(confident_features, pseudo_labels)


            #loss_ce = criterion(confident_logits, pseudo_labels)
            #print(loss_ce)

            #loss = loss_ce
       

            loss = loss_info
            # 总 Loss
            #loss = loss_ce + loss_info

            # #print("loss",loss)
            # loss = loss+ loss1

        loss_value = loss.item()

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    #output = torch.cat(tensors_gather, dim=0)
    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')

    return output
