import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import warnings

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

#from datasets import build_dataset
from engine import train_one_epoch, evaluate
import models
import my_models
import torch.nn as nn

import utils

from video_dataset import VideoDataSet
from video_dataset_aug import get_augmentor, build_dataflow
from video_dataset_config import get_dataset_config, DATASET_CONFIG


# from network import xception


warnings.filterwarnings("ignore", category=UserWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    # TALL_SWIN      mamba_vision_B    dinov3_vitb16
    parser.add_argument('--model_name',default="dinov3_vitb16")
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    
    # Dataset parameters
    # 指定数据集文本文件的路径，通常用于存储有关数据集的元数据，比如文件名和标签。
    # /data/siyu.liu/datasets/Celeb-DF-v2/                   Celeb-DF-v2
    # /data/siyu.liu/datasets/DFDC/                          DFDC
    # /data/siyu.liu/datasets/FF++/                          FF++
    # /data/siyu.liu/datasets/FF++_c40/                      CFF++
    # /data/siyu.liu/datasets/Wilddeepfake/images/           Wilddeepfake
    # Face2Face,  Deepfakes, FaceSwap, NeuralTextures
    # UnFace2Face UnDeepfakes UnFaceSwap UnNeuralTextures

    # 指定数据集文本文件的路径，通常用于存储有关数据集的元数据，比如文件名和标签。
    parser.add_argument('--data_txt_dir', type=str,default='/data/siyu.liu/datasets/FF++/', help='path to text of dataset')
    # 数据集的实际存放路径，包括所有图像或视频文件。
    parser.add_argument('--data_dir', type=str,default="/data/siyu.liu/datasets/FF++/", help='path to dataset')
    # 选择的数据集名称，默认值为 Celeb-DF-v2，并且允许用户从预定义的选项中选择。
    parser.add_argument('--dataset', default='FaceSwap',
                        choices=list(DATASET_CONFIG.keys()), help='path to dataset file list')
    # 每个样本的持续时间（以帧数表示），默认值为 4，通常用于处理视频数据。  也就是tall的布局所需要的帧数
    parser.add_argument('--duration', default=4, type=int, help='number of frames')
    # 每组采样的帧数，默认值为 1。用于控制均匀采样或密集采样的频率。
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    # 布尔值，指示是否以 3D 卷积的格式加载数据，默认值为 False。
    parser.add_argument('--threed_data', default=False, help='load data in the layout for 3D conv')
    # 输入图像的大小，默认值为 224，用于调整输入图像到模型所需的尺寸。
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    # 布尔值，指示是否禁用缩放操作，直接裁剪到输入大小，默认值为 False。
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, directly crop the input_size')
    # 布尔值，指示是否进行确定性采样，默认值为 False。
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform determinstic sampling for data loader')
    # 布尔值，指示是否进行密集采样，默认值为 True，允许更细致的数据采样。
    parser.add_argument('--dense_sampling', default=True,
                        help='perform dense sampling for data loader')
    # 数据增强的版本，默认值为 v1，允许选择 v1 或 v2，分别对应不同的数据增强方法。
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    # 数据增强时使用的缩放范围，默认值为 [256, 320]，允许用户根据需要调整
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    # 数据的模态，默认为 rgb，可以选择 rgb 或 flow，通常用于处理不同类型的视频数据。
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow')
    # 布尔值，指示是否使用 LMDB 格式存储数据而不是 JPEG，默认值为 False。
    parser.add_argument('--use_lmdb', default=False, help='use lmdb instead of jpeg.')
    # 布尔值，指示是否直接使用 PyAV 库处理视频，默认值为 False。
    parser.add_argument('--use_pyav', default=False, help='use video directly.')

    # temporal module
    # 布尔值，用于指定是否从预训练的网络开始训练，默认值为指定的权重路径
    ######   需要改动
     
    # 是否启用与训练权重
    parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Start with pretrained version of specified network (if avail)')  

    # 用于选择应用的时间模块，默认值为 None，可选项包括 ResNet3d、TAM、TTAM、TSM、TTSM 和 MSA。
    # 该参数指定了模型在处理时间序列数据时使用的模块。
    parser.add_argument('--temporal_module_name', default=None, type=str, metavar='TEM', choices=['ResNet3d', 'TAM', 'TTAM', 'TSM', 'TTSM', 'MSA'],
                        help='temporal module applied. [TAM]')
    # 布尔值，指示在时间模块中是否仅使用注意力机制，默认值为 False。
    # 这意味着如果为 True，则可能不使用其他的卷积或池化操作。
    parser.add_argument('--temporal_attention_only', action='store_true', default=False,
                        help='use attention only in temporal module]')
    # 布尔值，指示是否应用令牌掩码，默认值为 False。令牌掩码用于控制哪些输入数据会被关注。
    parser.add_argument('--no_token_mask', action='store_true', default=False, help='do not apply token mask')
    # 用于调整空间头部数量的比例，默认值为 1.0。
    # 这个参数可以影响注意力机制中的头部数量，从而改变模型的复杂度。
    parser.add_argument('--temporal_heads_scale', default=1.0, type=float, help='scale of the number of spatial heads')
    # 用于调整空间 MLP（多层感知器）的比例，默认值为 1.0。
    # 类似于头部比例，这个参数可以影响 MLP 的结构和复杂性。
    parser.add_argument('--temporal_mlp_scale', default=1.0, type=float, help='scale of spatial mlp')
    # 布尔值，指示是否在时间模块中使用相对位置编码，默认值为 False。
    # 相对位置编码用于捕捉序列中元素之间的相对位置信息。
    parser.add_argument('--rel_pos', action='store_true', default=False,
                        help='use relative positioning in temporal module]')
    # 指定进行时间池化的方式，默认值为 None，可选项包括 avg（平均池化）、max（最大池化）、conv（卷积池化）
    # 和 depthconv（深度卷积池化）。
    # 该参数影响时间序列数据的汇聚方式。
    parser.add_argument('--temporal_pooling', type=str, default=None, choices=['avg', 'max', 'conv', 'depthconv'],
                        help='perform temporal pooling]')
    # 用于选择在时间注意力中使用的瓶颈结构，默认值为 None，
    # 可选项包括 regular（常规瓶颈）和 dw（深度卷积瓶颈）。
    # 这个参数可以影响模型的效率和性能。
    parser.add_argument('--bottleneck', default=None, choices=['regular', 'dw'],
                        help='use depth-wise bottleneck in temporal attention')
    # 表示时间序列的帧数，默认值为 14，通常用于控制输入序列的长度。
    parser.add_argument('--window_size', default=14, type=int, help='number of frames')
    # 指定每行显示的帧数，默认值为 2，用于可视化或处理图像时的布局设置。
    parser.add_argument('--thumbnail_rows', default=2, type=int, help='number of frames per row')
    # 布尔值，指示是否将中心位置嵌入添加到图像令牌中，默认值为 False。这个参数用于增强模型在处理图像时对位置信息的理解。
    parser.add_argument('--hpe_to_token', default=False, action='store_true',
                        help='add hub position embedding to image tokens')
    # Model parameters
    # 指定要训练的模型的名称，默认值为 'TALL_SWIN'。这个参数用于选择要使用的特定模型。
    # 有两个模型可以使用 TALL_SWIN      mamba_vision_B      dinov3_vitb16
    parser.add_argument('--model', default='dinov3_vitb16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # 指定输入图像的大小，默认值为 224。这个参数控制输入到模型中的图像尺寸，通常为正方形。
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    # 表示 dropout 率，默认值为 0.0。Dropout 是一种正则化技术，用于在训练时随机丢弃一定比例的神经元，以防止过拟合。
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    # 表示 drop path 率，默认值为 0.1。Drop path 是另一种正则化方法，随机丢弃网络中的某些路径，以增强模型的泛化能力。
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # 表示 drop block 率，默认值为 None。Drop block 是一种基于块的 dropout 方法，
    # 随机丢弃某些块而不是单个神经元。这有助于在某些情况下提高模型的鲁棒性。
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    # 布尔值，指示是否启用模型的指数移动平均（EMA），默认为 False。
    # 如果启用，模型会在训练期间计算并保存权重的移动平均，以提高模型的性能。
    parser.add_argument('--model-ema', action='store_true')
    # 与 --model-ema 参数相反，用于禁用模型的 EMA。当使用该选项时，model_ema 的值将被设置为 False。
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # 设置 model_ema 参数的默认值为 True。这意味着如果没有明确指定 --no-model-ema，模型将默认启用 EMA。
    parser.set_defaults(model_ema=True)
    # 指定 EMA 的衰减率，默认值为 0.99996。这个参数控制移动平均的更新速度，
    # 值越接近 1，更新越慢，模型参数变化对平均值的影响越小。
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    # 布尔值，指示是否强制在 CPU 上计算模型的 EMA，默认值为 False。
    # 如果为 True，即使在使用 GPU 时，也会将 EMA 计算强制在 CPU 上执行。
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters

    # 指定优化器的名称，默认值为 'adamw'。
    # 该参数决定了训练过程中使用的优化算法，adamw 是一种常用的变种 Adam 优化器，具有权重衰减。
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    
    # 指定优化器的 epsilon 值，默认值为 1e-8。
    # 该参数用于防止在优化过程中除以零的情况，增加数值稳定性。
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    
    # 指定优化器的 beta 值，默认为 None，这意味着将使用优化器的默认值。通常，
    # Adam 和 AdamW 优化器使用两个 beta 值来控制一阶和二阶矩的衰减率。
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    
    # 指定梯度裁剪的范数，默认值为 None，表示不进行裁剪。
    # 梯度裁剪是一种防止梯度爆炸的方法，可以限制梯度的最大范数。
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    # 指定 SGD 优化器的动量，默认值为 0.9。
    # 动量是优化算法中的一种技术，通过结合当前梯度和之前的梯度来加速收敛。
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    # 指定权重衰减的值，默认值为 1e-5。
    # 权重衰减是一种正则化技术，通过在损失函数中增加权重的 L2 范数来防止过拟合。
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    # 指定学习率调度器的类型，默认值为 'cosine'。
    # 学习率调度器用于调整训练过程中学习率的变化方式，cosine 调度器通常用于逐渐减小学习率。
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    # 指定初始学习率，默认值为 1.5e-5。学习率控制模型权重更新的幅度，影响模型的收敛速度和效果。
    parser.add_argument('--lr', type=float, default=1.5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    # 指定学习率噪声的开/关时期的百分比，默认为 None。这允许在训练期间引入噪声，以增加模型的鲁棒性。
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    # 指定学习率噪声的限制百分比，默认值为 0.67。这定义了在每个训练周期内噪声的应用范围。
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    # 指定学习率噪声的标准差，默认值为 1.0。标准差越大，噪声变化的幅度越大，从而增加学习率的不确定性。
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    # 指定预热阶段的学习率，默认值为 1.5e-8。预热学习率是训练初期使用的小学习率，旨在避免模型参数在初始阶段更新过快。
    parser.add_argument('--warmup-lr', type=float, default=1.5e-8, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    # 指定学习率的下限，默认值为 1.5e-7。对于使用周期性调度器的情况，当学习率减小到 0 时，确保学习率不会低于这个值。
    parser.add_argument('--min-lr', type=float, default=1.5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # 指定每隔多少个训练周期减少学习率，默认值为 10。这个参数用于定义学习率衰减的频率。
    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                        help='epoch interval to decay LR')
    # 指定预热阶段的训练周期数，默认值为 10。在这个阶段，学习率逐渐增大，以便模型逐步适应。
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    # 指定在达到最小学习率后，保持学习率不变的训练周期数，默认值为 10。这有助于模型在达到最低学习率后进行稳定的训练。
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # 指定 Plateau 学习率调度器的耐心周期数，默认值为 10。如果在这个周期内没有性能改善，则学习率会被调整。
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    # 指定学习率衰减的速率，默认值为 0.1。当学习率衰减时，它将乘以这个衰减率，以控制衰减的程度。
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    # 指定颜色抖动的因子，默认值为 0.4。该参数用于随机调整图像的亮度、对比度、饱和度等，以增强模型对颜色变化的鲁棒性。
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    # 指定使用的 AutoAugment 策略，默认值为 'rand-m9-mstd0.5'。
    # AutoAugment 是一种自动数据增强技术，使用预定义的增强策略来提高模型的泛化能力。
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    # 指定标签平滑的因子，默认值为 0.1。标签平滑通过减小目标标签的确定性，帮助模型更好地处理分类任务，从而减少过拟合。
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    # 指定训练时的插值方法，默认值为 'bicubic'。该参数定义在图像缩放时使用的插值算法，可以选择随机、双线性或双三次插值。
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # 使用此选项可以启用重复数据增强。默认情况下为 False。
    # 当设置为 True 时，训练过程中将对同一图像应用相同的增强策略多次，以增强模型的鲁棒性。
    parser.add_argument('--repeated-aug', action='store_true')
    # 此选项用于禁用重复数据增强，设置 dest='repeated_aug' 的值为 False。这与上一个参数相对，确保在训练过程中不会对图像进行重复增强。
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    # 这行代码确保在没有提供 --repeated-aug 选项时，repeated_aug 默认值为 False。
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    # 随机擦除的概率，默认值为 0.0。如果设置为大于 0 的值，模型将在训练期间随机擦除图像的一部分，以增强鲁棒性。
    # 改过开始为0.0
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    # 指定随机擦除的模式，默认值为 'pixel'。该参数定义了擦除时采用的具体方法，可以选择不同的策略，如按像素或区域擦除。
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    # 随机擦除的次数，默认值为 1。此参数定义在一张图像上应用随机擦除操作的次数。
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    # 启用此选项后，第一次增强（清洁）拆分时不会进行随机擦除，默认值为 False。此参数控制是否在特定增强步骤之前跳过随机擦除。
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    # 启用或禁用 Cutout 技术，默认值为 True。Cutout 是一种数据增强技术，通过遮挡图像的部分区域来提高模型的泛化能力。
    parser.add_argument('--cutout',default=True)
    # 指定 Mixup 的超参数，默认值为 0。
    # 当该值大于 0 时，将启用 Mixup，允许对输入图像进行线性组合，以增强模型的表现。
    # 改过初始为 0 
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    # 该过初始值为0
    # 指定 CutMix 的超参数，默认值为 0。
    # 与 Mixup 类似，当该值大于 0 时，将启用 CutMix，这是一种通过裁剪和混合不同图像来增强数据的方法。
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    # 指定 CutMix 的最小/最大比例，使用时覆盖 alpha 值并启用 CutMix。
    # 默认值为 None。此参数允许设置 CutMix 计算的比例范围。
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    # 指定执行 Mixup 或 CutMix 的概率，默认值为 1.0。
    # 该参数定义了在启用 Mixup 或 CutMix 时实际应用的概率。
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    # 当同时启用 Mixup 和 CutMix 时，切换到 CutMix 的概率，默认值为 0.5。该参数决定了在执行混合时的策略切换概率。
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # 指定如何应用 Mixup/CutMix 参数，默认值为 'batch'。可选值包括 'batch'（对整个批次应用）、
    # 'pair'（对成对样本应用）或 'elem'（对单个元素应用）
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    # 指定保存输出结果的路径，默认值为 "output/FF++"。如果该路径为空，则不会保存任何结果。
    parser.add_argument('--output_dir', default="/data/yiming.hao/code/TALL4Deepfake_raw_2_new/output/TALL_JPL/onlyFaceSwap",
                        help='path where to save, empty for no saving')
    # 指定用于训练或测试的设备，默认值为 'cuda'。可以根据需要选择 CPU 或 GPU。
    """
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    """
    parser.add_argument('--device', default='4',
                        help='device to use for training / testing')
    # 设置随机种子，默认值为 42。用于确保实验的可重复性。
    parser.add_argument('--seed', default=42, type=int)
    # 指定从检查点恢复训练的路径，默认值为空字符串 ""。如果提供此参数，训练将从指定的检查点开始。
    parser.add_argument('--resume', default="", help='resume from checkpoint')
    # 如果设置此参数，训练将不会恢复损失缩放器的状态。此参数用于防止在恢复时丢失缩放信息。
    parser.add_argument('--no-resume-loss-scaler', action='store_false', dest='resume_loss_scaler')
    # 如果设置此参数，将禁用自动混合精度（AMP）。AMP 是一种提高训练效率的技术，尤其在使用 GPU 时。
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='disable amp')
    # *****需要修改
    # 指定用于节省内存的检查点路径，默认值为 
    # "/data/siyu.liu/code/TALL4Deepfake/data_preparation/swin_base_patch4_window7_224_22k.pth"。通过使用检查点，可以减少内存消耗。
    # parser.add_argument('--use_checkpoint', default="/data/siyu.liu/code/TALL4Deepfake/data_preparation/swin_base_patch4_window7_224_22k.pth", help='use checkpoint to save memory')
    parser.add_argument('--use_checkpoint', default=True, help='use checkpoint to save memory')
    # 设置开始训练的 epoch 数，默认值为 0。
    # 如果从检查点恢复训练，可以指定此参数以从特定 epoch 开始。
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 启用此参数后，将只执行评估，而不进行训练。
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    # 指定用于数据加载的工作线程数，默认值为 8。
    # 增加此值可以加快数据加载速度，尤其是在处理大型数据集时。
    # 默认是八
    parser.add_argument('--num_workers', default=8, type=int)
    # 启用此参数后，会将 CPU 内存固定在 DataLoader 中，
    # 以提高数据传输到 GPU 的效率。默认值为 False。
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # 如果设置此参数，将禁用内存固定。
    # 用于控制 DataLoader 的内存管理。
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    # 将 pin_mem 的默认值设置为 True。
    parser.set_defaults(pin_mem=True)

    # for testing and validation
    # 指定在测试或验证阶段进行裁剪的数量，默认值为 1。可选择的值包括 1, 3, 5, 10。
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    # 指定在测试或验证阶段生成的剪辑数量，默认值为 8。
    parser.add_argument('--num_clips', default=8, type=int)

    # distributed training parameters
    # 指定分布式进程的数量，默认值为 1。在进行分布式训练时，该参数用于设置总的进程数。
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # 指定当前进程的本地排名，用于分布式训练。
    parser.add_argument("--local_rank", type=int)
    # 指定用于设置分布式训练的 URL，默认值为 'env://'。该参数用于配置不同进程之间的通信。
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # 启用自动恢复功能，默认值为 True。如果检测到上次未完成的训练，将自动恢复。
    parser.add_argument('--auto-resume', default=True, help='auto resume')
    # exp
    # parser.add_argument('--simclr_w', type=float, default=0., help='weights for simclr loss')
    # 启用此参数后，在对比学习中将不涉及 Mixup 数据增强。
    parser.add_argument('--contrastive_nomixup', action='store_true', help='do not involve mixup in contrastive learning')
    # 启用此参数后，将对模型进行微调，默认值为 False。
    parser.add_argument('--finetune', default=False, help='finetune model')
    
   
    # 指定预训练模型的路径，  训练时后不启动
    # 用于初始化模型权重。
    # parser.add_argument('--initial_checkpoint', type=str, default='', help='path to the pretrained model')
    parser.add_argument('--initial_checkpoint', type=str, default='', help='path to the pretrained model')
    # 启用此参数后，将使用 HEXA 方法进行对比学习。
    parser.add_argument('--hard_contrastive', action='store_true', help='use HEXA')

    # 此参数被注释掉，说明可能是一个可选的自我蒸馏权重参数，默认值为 0。如果启用，将用于设置自我蒸馏的权重。
    # parser.add_argument('--selfdis_w', type=float, default=0., help='enable self distillation')

    # 帮助分类个数
    
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')


    return parser


def main(args):
    # 这个函数主要用于初始化分布式训练环境。
    utils.init_distributed_mode(args)
    # 输出 args 对象的内容，便于调试和确认参数设置。
    print(args)
    # Patch
    # 检查 args 是否具有 hard_contrastive 属性。
    # 如果没有该属性，则将其设置为 False。
    if not hasattr(args, 'hard_contrastive'):
        args.hard_contrastive = False
    # 检查 args 是否具有 selfdis_w 属性。
    if not hasattr(args, 'selfdis_w'):
        # 如果没有该属性，则将其设置为 0.0。
        args.selfdis_w = 0.0

    # 此处有改动
    #is_imnet21k = args.data_set == 'IMNET21K'
    # 根据 args.device 设置当前的计算设备（如 CPU 或 GPU）
    #device = torch.device(args.device)
    
    # 这行代码将环境变量 CUDA_VISIBLE_DEVICES 设置为 args.device。
    # 通过设置这个变量，可以控制哪些 GPU 是可用的，只有在这个变量中指定的 GPU 会被 PyTorch 识别和使用。
    # 例如，如果 args.device 是 "0,1"，那么只会使用 GPU 0 和 1。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # 这行代码配置 PyTorch 的 CUDA 内存分配策略，设置 max_split_size_mb 为 128 MB。这意味着在分配 GPU 内存时，
    # PyTorch 将尝试避免分配超过 128 MB 的单个块，以减少内存碎片，提高内存利用率。
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #  该过的代码
    print(torch.cuda.current_device())

    # fix the seed for reproducibility
    # 计算随机种子，使用 args.seed 和当前进程的排名（通过 get_rank() 获取），以确保在分布式训练中每个进程有不同的种子。
    seed = args.seed + utils.get_rank()
    # 在 PyTorch 中设置随机种子，以确保模型训练的可重复性。
    torch.manual_seed(seed)
    # 在 NumPy 中设置随机种子，确保随机操作的可重复性。
    np.random.seed(seed)
    # random.seed(seed)
    
    # 启用 cudnn 的自动调优，使得在确定输入数据的大小时能优化卷积算法，适合固定输入大小的模型
    cudnn.benchmark = True
    
    # 调用 get_dataset_config 函数，获取与指定数据集相关的配置参数，并将其赋值给相应的变量。
    # num_classes:数据集中类别的数量 train_list_name:训练集文件名 val_list_name: 验证集文件名 test_list_name:测试集文件名
    # filename_seperator:文件名分隔符 image_tmpl:图像模板 例如 ***.jpg  filter_video：  label_file:
    # args.dataset = CFF++    args.use_lmdb =False
    """
    得到的结果例如：
        num_classes: 2                  train_list_name :  train.txt                val_list_name :   val.txt               test_list_name:   test.txt
        filename_seperator : " "(空格)   image_tmpl  :   {:03d}.jpg                  filter_video :   3                      label_file : None  (没有定义)
    """
    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset, args.use_lmdb)

    args.num_classes = num_classes
    # 检查输入的模态是否为 RGB。
    if args.modality == 'rgb':
        # 如果是 RGB 模态，设置输入通道为 3
        args.input_channels = 3
    # 检查输入的模态是否为光流（flow）
    elif args.modality == 'flow':
        # 如果是光流模态，设置输入通道为 10（假设使用了 5 对光流信息）。
        args.input_channels = 2 * 5

    # 初始化 mixup_fn 为 None。
    mixup_fn = None
    # 检查是否激活了 Mixup 或 Cutmix 数据增强方法，
    # 如果 args.mixup、args.cutmix 大于 0 
    # cutmix_minmax 不为 None，
    # 则 mixup_active 为 True。
    # Mixup 允许对输入图像进行线性组合，以增强模型的表现。
    # 启用 CutMix，这是一种通过裁剪和混合不同图像来增强数据的方法。
    # cutmix_minmax 指定 CutMix 的最小/最大比例
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
    # 打印正在创建的模型名称。
    print(f"Creating model: {args.model}")
    # 调用 create_model 函数，传入各种参数（包括模型名称、是否使用预训练权重、持续时间等）来创建模型实例。各参数的作用如下：
    """
    args.model: 指定模型架构。   TALL_SWIN
    pretrained=args.pretrained: 是否使用预训练模型。  /data/yiming.hao/code/TALL-SWIN/TALL4Deepfake/pre_pth/swin_base_patch4_window7_224_22k.pth
    duration=args.duration: 可能表示输入序列的持续时间。                           4
    hpe_to_token, rel_pos, window_size, thumbnail_rows: 
    这些参数可能与特定模型架构或数据处理方法有关。
    hpe_to_token: 布尔值，指示是否将中心位置嵌入添加到图像令牌中，默认值为 False         rel_pos: 布尔值，指示是否在时间模块中使用相对位置编码，默认值为 False
    window_size: 表示时间序列的帧数，默认值为 14 通常用于控制输入序列的长度              thumbnail_rows: 指定每行显示的帧数，默认值为 2,用于可视化或处理图像时的布局设置
    token_mask=not args.no_token_mask: 是否使用 token 掩码。
    online_learning=False: 指定是否为在线学习模式。
    num_classes=args.num_classes: 设置输出类别数。
    drop_rate, drop_path_rate, drop_block_rate: 控制不同层的丢弃率。
    drop                         drop_path                       drop_block
    use_checkpoint=args.use_checkpoint: 是否使用检查点。
    use_checkpoint', default="/data/yiming.hao/code/TALL-SWIN/TALL4Deepfake/pre_pth/swin_base_patch4_window7_224_22k.pth"
    """
    model =create_model(
        args.model,
        pretrained=args.pretrained,
        duration=args.duration,
        hpe_to_token = args.hpe_to_token,
        rel_pos = args.rel_pos,
        window_size=args.window_size,
        thumbnail_rows = args.thumbnail_rows,
        token_mask=not args.no_token_mask,
        online_learning = False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        #drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        use_checkpoint=args.use_checkpoint
    )

    # TODO: finetuning
    # 将模型移动到指定的设备（如 GPU 或 CPU），以便进行训练。
    model.to(device)
    # 初始化 model_ema 为 None，准备后续的 EMA（Exponential Moving Average）模型。
    model_ema = None
    # 检查是否启用 EMA 模型，如果是，创建一个新的 EMA 模型。
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # 实例化 EMA 模型，使用当前模型和指定的衰减因子。根据设置选择设备。
        # model_ema_decay
        # 指定 EMA 的衰减率，默认值为 0.99996。这个参数控制移动平均的更新速度，
        # 值越接近 1，更新越慢，模型参数变化对平均值的影响越小。
        # args.model_ema_force_cpu
        # 布尔值，指示是否强制在 CPU 上计算模型的 EMA，默认值为 False。
        # 如果为 True，即使在使用 GPU 时，也会将 EMA 计算强制在 CPU 上执行。
        # 指定是否从先前保存的模型状态中恢复 EMA 模型。如果 args.resume 为 True，
        # 那么在创建 ModelEma 时会尝试加载之前保存的 EMA 权重，以便继续训练。        
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)
    # 初始化 model_without_ddp 为当前模型，后续可能会用于不带分布式数据并行的情况。
    model_without_ddp = model
    # 检查是否启用分布式训练，如果是，进行以下处理：
    if args.distributed:
        # 将模型封装为分布式数据并行（DDP）模型，以支持在多个 GPU 上并行训练。
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # 将模型封装为分布式数据并行模型，以支持在多个 GPU 上进行并行训练。
        # device_ids=[args.gpu] 指定当前 GPU 的 ID，确保每个进程使用对应的 GPU。
        ##### 该过
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # 计算模型中所有可训练参数的总数。p.numel() 返回参数的元素总数，
    # p.requires_grad 确保只计算需要梯度的参数。这样可以帮助我们了解模型的复杂性。
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 打印出模型中可训练参数的数量，帮助用户了解模型的规模。
    print('number of params:', n_parameters)

    # 调用 create_optimizer 函数创建优化器。该函数使用传入的 args 和模型来设置优化器的类型和参数（如学习率）。
    optimizer = create_optimizer(args, model)
    # 初始化一个 NativeScaler 实例，用于在训练中进行损失缩放。
    # 这在混合精度训练中很常用，有助于避免数值不稳定性和梯度下溢。
    loss_scaler = NativeScaler()
    #print(f"Scaled learning rate (batch size: {args.batch_size * utils.get_world_size()}): {linear_scaled_lr}")
    # 调用 create_scheduler 函数创建学习率调度器，以动态调整学习率。
    # 返回的调度器 lr_scheduler 可以根据训练进度自动调整学习率，从而提高训练效果。
    lr_scheduler, _ = create_scheduler(args, optimizer)
    # 初始化损失函数，这里使用的是标签平滑交叉熵（Label Smoothing Cross Entropy）。
    # 这种损失函数通过对标签进行平滑处理，有助于提高模型的泛化能力并减少过拟合。
    criterion = LabelSmoothingCrossEntropy()
    
    # 检查 args.mixup 的值是否大于 0。这表示是否启用 Mixup 数据增强技术。
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        # 损失函数支持混合标签处理，适用于 Mixup 的标签平滑效果。
        criterion = SoftTargetCrossEntropy()
    # 检查 args.smoothing 是否被设置。这用于确定是否启用标签平滑。
    elif args.smoothing:
        # 如果启用标签平滑，使用 LabelSmoothingCrossEntropy 作为损失函数，并将平滑参数传入。
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # 如果都不满足上述条件，使用标准的交叉熵损失函数。
        criterion = torch.nn.CrossEntropyLoss() 
    # 检查是否在分布式训练模式下。
    if args.distributed:
        # 如果启用分布式训练且模型的配置中没有定义 mean，则使用默认的均值 (0.5, 0.5, 0.5)。如果定义了，则使用模型配置中的值。
        mean = (0.5, 0.5, 0.5) if 'mean' not in model.module.default_cfg else model.module.default_cfg['mean']
        std = (0.5, 0.5, 0.5) if 'std' not in model.module.default_cfg else model.module.default_cfg['std']
    else:
        # 类似地，如果模型的配置中没有定义 std，则使用默认的标准差 (0.5, 0.5, 0.5)。
        mean = (0.5, 0.5, 0.5) if 'mean' not in model.default_cfg else model.default_cfg['mean']
        std = (0.5, 0.5, 0.5) if 'std' not in model.default_cfg else model.default_cfg['std']
    
    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # create data loaders w/ augmentation pipeiine
    # 定义用于创建视频数据集的类。
    video_data_cls = VideoDataSet
    
    
    
    # 构建训练列表的路径，使用 args.data_txt_dir 作为基础路径和 train_list_name 作为文件名
    # 例如 /data/siyu.liu/datasets/Celeb-DF-v2/train.txt
    train_list = os.path.join(args.data_txt_dir, train_list_name)
    
    # 调用 get_augmentor 函数创建训练数据的增强器。
    # 传入的参数包括输入大小、均值、标准差、增强版本、缩放范围、cutout 设置和数据集类型。
    train_augmentor = get_augmentor(True, args.input_size, mean, std, threed_data=False,
                                    version=args.augmentor_ver, scale_range=args.scale_range, cut_out = args.cutout,dataset=args.dataset)
    
    dataset_train = video_data_cls(args.data_dir, train_list, args.duration, args.frames_per_group,
                                num_clips=args.num_clips,
                                modality=args.modality, image_tmpl=image_tmpl,
                                dense_sampling=args.dense_sampling,
                                transform=train_augmentor, is_train=True, test_mode=False,
                                seperator=filename_seperator, filter_video=filter_video,whole_video=False)
    
    # 获取当前进程的数量，通常用于分布式训练时。
    num_tasks = utils.get_world_size()
    
    # 函数创建训练数据加载器。
    # 传入训练数据集 dataset_train，设置 is_train=True，指定批量大小 batch_size，
    # 工作线程数 num_workers，并指明是否进行分布式训练。
    data_loader_train = build_dataflow(dataset_train, is_train=True, batch_size=args.batch_size,
                                    workers=args.num_workers, is_distributed=args.distributed)
    
    
    
    
    # 通过组合数据文本目录和验证列表名称，构建验证数据集的路径。
    # 例如 /data/siyu.liu/datasets/Celeb-DF-v2/val.txt
    val_list = os.path.join(args.data_txt_dir, val_list_name)
    # 使用 get_augmentor 函数获取验证数据的增强器。
    # 这里 is_train 设置为 False，因为验证过程中不需要进行数据扩充（如混合、翻转等）。
    # get_augmentor一个灵活的数据增强流水线，能够根据训练或验证模式、不同的数据集和增强版本等条件，自动选择和配置相应的图像处理操作
    val_augmentor = get_augmentor(False, args.input_size, mean, std, args.disable_scaleup,
                                threed_data=args.threed_data, version=args.augmentor_ver,
                                scale_range=args.scale_range, num_clips=args.num_clips, num_crops=args.num_crops,cut_out = False, dataset=args.dataset)
    
    # 创建验证数据集 dataset_val，使用之前定义的 video_data_cls 类。
    """
    传入相关参数，如数据目录、验证列表路径、样本持续时间、帧数、模态类型、数据变换（增强器）等。
    """
    dataset_val = video_data_cls(args.data_dir, val_list, args.duration, args.frames_per_group,
                                num_clips=args.num_clips,
                                modality=args.modality, image_tmpl=image_tmpl,
                                dense_sampling=args.dense_sampling,
                                transform=val_augmentor, is_train=False, test_mode=False,
                                seperator=filename_seperator, filter_video=filter_video)
    # 调用 build_dataflow 函数创建验证数据加载器。
    # 参数设置与训练数据加载器类似，只是将 is_train 设置为 False
    data_loader_val = build_dataflow(dataset_val, is_train=False, batch_size=args.batch_size,
                                    workers=args.num_workers, is_distributed=args.distributed)


    # 初始化 max_accuracy 为 0.0，用于记录模型的最高准确率
    max_accuracy = 0.0
    
    # 将输出目录转换为 Path 对象，便于后续操作
    output_dir = Path(args.output_dir)
    
    # 如果 initial_checkpoint 参数被提供，则加载该检查点，并将模型的状态加载到当前模型中。
    if args.initial_checkpoint:
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        utils.load_checkpoint(model, checkpoint['model'])
    
    # 如果 auto_resume 为真且 resume 参数为空
    if args.auto_resume:
        # 尝试设置 resume 为输出目录中的默认检查点路径。
        if args.resume == '':
            args.resume = str(output_dir / "checkpoint.pth")
            # 如果该路径不存在，则将 resume 设置为空。
            if not os.path.exists(args.resume):
                args.resume = ''
    # 如果 resume 参数非空，检查其是否为 URL：
    if args.resume:
        if args.resume.startswith('https'):
            # 如果是 URL，通过 torch.hub.load_state_dict_from_url 下载并加载检查点。
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 否则，直接从本地加载检查点。
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 使用 utils.load_checkpoint 函数将检查点中的模型状态加载到当前模型。
        utils.load_checkpoint(model, checkpoint['model'])
        # 这行代码检查当前是否不在评估模式（即是在训练模式），同时确保检查点中包含优化器、学习率调度器和当前的 epoch 信息。
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # 这两行代码从检查点中加载优化器和学习率调度器的状态，以便在恢复训练时继续使用相同的学习策略。
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # 将起始的 epoch 设置为从检查点恢复的 epoch 加一，以确保在恢复后从下一个 epoch 开始训练。
            args.start_epoch = checkpoint['epoch'] + 1
            # 如果检查点中包含损失缩放器的状态，并且用户选择恢复缩放器状态，则加载该状态。
            if 'scaler' in checkpoint and args.resume_loss_scaler:
                print("Resume with previous loss scaler state")
                loss_scaler.load_state_dict(checkpoint['scaler'])
            # 如果使用了模型的 EMA 机制，则从检查点中加载该状态。
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            # 恢复训练过程中的最大准确率，以便在后续的训练中进行比较和监控。
            max_accuracy = checkpoint['max_accuracy']
    
    # 如果用户指定了评估模式
    if args.eval:
        # 调用 evaluate 函数对验证数据集进行评估，并输出模型在测试集上的准确率，然后结束函数。
        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=args.distributed, amp=args.amp, num_crops=args.num_crops, num_clips=args.num_clips)
        # 输出模型在测试集上的准确率，len(dataset_val) 获取测试集的样本数量。
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    # 如果不在评估模式，输出开始训练的信息，并记录开始时间。
    print(f"Start training, currnet max acc is {max_accuracy:.2f}")
    # 记录当前时间，用于计算总训练时间
    start_time = time.time()
    # 循环遍历每个训练周期（epoch），从 start_epoch 到 epochs。
    for epoch in range(args.start_epoch, args.epochs):
        # 如果使用分布式训练
        if args.distributed:
            # 设置数据加载器的采样器以确保每个进程在每个周期使用不同的数据。
            data_loader_train.sampler.set_epoch(epoch)
        
        # 调用 train_one_epoch 函数进行一个周期的训练，返回训练统计信息 train_stats。
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,args.num_clips,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, num_tasks, True,
            amp=args.amp,
            contrastive_nomixup=args.contrastive_nomixup,
            hard_contrastive=args.hard_contrastive,
            finetune=args.finetune
        )
        
        # 更新学习率调度器的状态，根据当前周期调整学习率。
        lr_scheduler.step(epoch)
        # 再次调用 evaluate 函数评估当前模型在验证集上的表现。
        
        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=args.distributed, amp=args.amp, num_crops=args.num_crops, num_clips=args.num_clips)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        
        """
        torch.save({
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict() if use_amp else None,  # 混合精度可选
    'scheduler': scheduler.state_dict() if scheduler else None,
    'args': args  # 可以是 Namespace 或 dict
}, f"checkpoint_epoch{epoch}.pth")
        """
        # 检查是否指定了输出目录，如果是，则进行检查点的保存。
        if args.output_dir:
            # 创建检查点文件路径列表。如果当前模型的准确率等于最大准确率，则添加一个保存最佳模型的路径。
            checkpoint_paths = [output_dir / 'checkpoint{}.pth'.format(epoch)]
            if test_stats["acc1"] == max_accuracy:
                checkpoint_paths.append(output_dir / 'model_best.pth')
            for checkpoint_path in checkpoint_paths:
                state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'scaler': loss_scaler.state_dict(),
                    'max_accuracy': max_accuracy
                }
                if args.model_ema:
                    state_dict['model_ema'] = get_state_dict(model_ema)
                utils.save_on_master(state_dict, checkpoint_path)
                
        # 构建一个字典 log_stats，记录训练和测试的统计信息、当前周期和模型参数数量。
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        # 如果指定了输出目录且当前进程是主进程，则将日志信息写入 log.txt 文件
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # 计算总训练时间，并以可读的格式输出。
    print('Training time {}'.format(total_time_str))


# 1111111
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # 这行代码检查 args.output_dir 是否存在。如果用户指定了一个输出目录，代码将执行接下来的操作。
    if args.output_dir:
        # .mkdir() 是 Path 对象的方法，用于创建目录
        # parents=True 参数表示如果输出目录的父目录不存在，也会一并创建。
        # exist_ok=True 参数表示如果目录已经存在，不会抛出错误。
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
