import torch
import torch.nn as nn
import torch.nn.functional as F
from DINO.dinov3.models.vision_transformer import vit_base
import copy
import os
import numpy
from DINO.dinov3.hub.backbones import Weights, _make_dinov3_vit, dinov3_vits16
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from JPL2 import JPLModule
class Backbone(nn.Module):
    def __init__(self, base_model, in_planes, num_classes=2, thumbnail_rows=2,img_size = 224,
        duration=4, **kwargs):
        super().__init__()
        self.base = base_model
        self.in_planes = in_planes
        self.num_classes = num_classes          # ✅ 存起来，后面 view/分类会用
        self.img_size = img_size                # ✅ 存起来，create_thumbnail 会用
        
        self.default_cfg = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }
        
        self.jpl = JPLModule(in_planes,2,6)

        self.classifier = nn.Linear(in_planes, num_classes) if num_classes > 0 else nn.Identity()

        
        #下面是按照tall_swin增加的参数

        self.duration = duration
        self.num_clips = 8
        self.thumbnail_rows = thumbnail_rows
        self.image_mode = True
        self.ape = False
        self.patches_resolution = (56,56)
        self.frame_padding = self.duration % thumbnail_rows if self.image_mode is True else 0
        if self.frame_padding != 0:
            self.frame_padding = self.thumbnail_rows - self.frame_padding
            self.duration += self.frame_padding
        # absolute position embedding
#            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#            trunc_normal_(self.absolute_pos_embed, std=.02)
            # print("frame_pos_embed\n")
            self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.duration, 128))
            # print("self.frame_pos_embed", self.frame_pos_embed)
            trunc_normal_(self.frame_pos_embed, std=.02)  

    def pad_frames(self, x):
        """
        用于对输入的帧数据进行填充，使其符合特定的帧数量要求。
        具体来说，它会根据 self.duration 和 self.frame_padding 的值，
        确保输入的帧数据在时间维度上的形状符合预期。接下来我将逐行解释这段代码。

        self: 表示当前类的实例,说明该方法属于一个类。
        x: 输入的张量k,代表一组帧数据,通常形状是 [batch_size, num_frames, ...]，其中 num_frames 是帧的数量。
        frame_num: 计算得到有效的帧数, 即填充后的总帧数减去填充的部分
        """
        frame_num = self.duration - self.frame_padding
        """
        这里使用 view 方法改变输入张量 x 的形状。view 会返回一个具有新形状的张量，(-1, 3 * frame_num) 是前两维的大小，
        后面是输入张量 x 除了时间维度外的其他维度（通常是图像的高度、宽度或其他特征维度）。
        3 * frame_num 表示将输入帧数的维度扩展为 3 * frame_num,可能是因为每帧图像有 RGB 三个通道，或者其他类似的操作。
        例如，如果 x 的形状是 [batch_size, num_frames, height, width]，而 frame_num = 10,
        则 view 后的形状将会是 [batch_size, 30, height, width]（假设每帧有 3 个通道）。
        """
        x = x.view((-1,3*frame_num)+x.size()[2:])
        """
        torch.zeros:创建一个大小为 (x.shape[0], 3 * self.frame_padding) + x.size()[2:] 的全零张量。
        这个张量的形状与 x 相似，但时间维度（帧数）被扩展为填充的部分 3 * self.frame_padding。
        x.shape[0] 是批量大小(batch size)。
        3 * self.frame_padding 是填充帧的数量,3 是因为每个帧有三个通道。
        x.size()[2:] 保持原始图像的其他维度（例如高度和宽度）。
        .cuda()：将这个张量移动到 GPU 上，假设有可用的 CUDA 设备。
        """
        x_padding = torch.zeros((x.shape[0], 3*self.frame_padding) + x.size()[2:]).cuda()
        """
        torch.cat:在时间维度(即第 1 维,dim=1)上将原始张量 x 和填充张量 x_padding 进行拼接。
        拼接后的张量 x 的形状将变为 ([batch_size, 3 * (frame_num + frame_padding), ...]),
        即原始帧数和填充帧数的总和。
        """
        x = torch.cat((x, x_padding), dim=1)
        """
        assert:这行代码用于确保填充后的帧数量与目标帧数量一致。
        如果填充后的帧数不等于 3 * self.duration(self.duration 是目标的帧数)，就会抛出异常并打印错误信息。
        x.shape[1] 是拼接后的帧数（即时间维度的大小）。
        3 * self.duration 是期望的总帧数（目标帧数乘以 3,因为每帧有 3 个通道）。
        如果 assert 检查失败（即帧数不匹配），会抛出异常并显示相应的错误信息，帮助调试。
        """
        assert x.shape[1] == 3 * self.duration, 'frame number %d not the same as adjusted input size %d' % (x.shape[1], 3 * self.duration)

        return x
    def create_image_pos_embed(self):
        """
        假设图像已经被划分为 patch,这里得到的是图像 patch 的行数和列数。
        例如:如果图片是224*224,每个 patch 是16*16,
        那 patches_resolution 就是 (14, 14)。
        """
        img_rows, img_cols = self.patches_resolution
        # frame_pos_embed 是形状为 (1, 时间帧数, 嵌入维度 T) 的张量，表示每一帧的嵌入向量。
        _, _, T = self.frame_pos_embed.shape
        """
        这里的含义是：将整张图像按帧的数量划分成格子，每帧对应一个格子。
        thumbnail_rows 控制划分的纵向分辨率。
        self.duration = 4表示视频总帧数为16。
        self.thumbnail_rows = 2:你想在图像纵向上分4块。
        则会把图像分成 4 x 4 = 16 个格子(行列都为4)，每一帧嵌入对应一个格子区域。
        """
        rows = img_rows // self.thumbnail_rows
        cols = img_cols // (self.duration // self.thumbnail_rows)
        img_pos_embed = torch.zeros(img_rows, img_cols, T).cuda()
         #print (self.duration, T, img_rows, img_cols, rows, cols)
        for i in range(self.duration):
            """
            i // self.thumbnail_rows 和 i % self.thumbnail_rows 是将第 i 帧定位到图像网格中的行列。
            """
            r_indx = (i // self.thumbnail_rows) * rows
            c_indx = (i % self.thumbnail_rows) * cols
            """
            把每一帧的嵌入 self.frame_pos_embed[0, i] 
            复制到图像中的一块区域(大小为 rows x cols)
            实现把时间信息“铺”到图像上的空间区域。
            """
            img_pos_embed[r_indx:r_indx+rows,c_indx:c_indx+cols] = self.frame_pos_embed[0, i]
            #print (r_indx, r_indx+rows, c_indx, c_indx+cols)
        return img_pos_embed.reshape(-1, T)
    
    def create_thumbnail(self, x):
        # import pdb;pdb.set_trace()
        # print("x.shape",x.shape)
        # x.shape torch.Size([32, 12, 224, 224])

        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            # 使用 nn.functional.interpolate 函数调整 x 的大小。
            # 参数 size=self.img_size 指定目标尺寸。
            # mode='bilinear' 指定双线性插值，用于调整图像大小。
            x = nn.functional.interpolate(x, size=self.img_size,mode='bilinear')
        # print("input_size",input_size)
        # input_size torch.Size([224, 224])
        """
        使用了 einops.rearrange,一个强大的工具,用于灵活地重排张量的维度。
        解释字符串 'b (th tw c) h w -> b c (th h) (tw w)':
        输入的张量 x 形状为 [batch_size, th * tw * c, height, width]。
        [32, 12, 224, 224]    即 [32 , 2 * 2 *3 , 224, 224]
        th 和 tw 分别是缩略图行和列的数量。  2 
        c 是每个像素点的通道数，通常为 3(RGB 图像)。
        将输入张量重新排列为形状 [batch_size, c, th * height, tw * width]。
        [32, 3, 448, 448] 即 [32, 3, 2 * 224, 2 * 224]
        th 和 tw 决定缩略图的分块数，重新排列后将它们合并到高宽维度。
        """
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.thumbnail_rows, c=3)
        # x.shape torch.Size([32, 3, 448, 448])
        # print("x.shape" , x.shape)
        return x



    def forward(self, x, targets=None):
        
        # print("targets",targets)
        # targets tensor([1, 1, 0, 1], device='cuda:0')

        # print("x.shape",x.shape)
        # x.shape torch.Size([4, 96, 224, 224])
        if self.frame_padding > 0:
            x = self.pad_frames(x)
            # print("x.shape",x.shape)
        else:
            # -1：自动计算这一维的大小以保持元素总数一致。
            # duration = 4是指每个视频样本 选择 4 帧
            # 3 * self.duration：新的通道数，表示每个时序块有 3 * self.duration 个通道。 也就是 12个通道
            # 经过view后 也就是 [32, 12, 224, 224]
            x = x.view((-1, 3 * self.duration) + x.size()[2:])
            # print("x.shape",x.shape)
            # x.shape torch.Size([32, 12, 224, 224])

        # 如果使用 tall 模块
        if self.image_mode:
            x = self.create_thumbnail(x)
            # print("cratethumbnail",x.shape)
            # cratethumbnail torch.Size([32, 3, 448, 448])
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
            # print("x.shpe_inter",x.shape)
            # x.shpe_inter torch.Size([32, 3, 224, 224])
            if targets is not None:
                targets = targets.repeat_interleave(self.num_clips)
        else:
            x = rearrange(x, 'b (n t c) h w -> (b n t) c h w', t=self.duration, c=3)
            if  targets is not None:
                targets = targetss.repeat_interleave(self.num_clips*self.duration)
        # print("x.shape",x.shape)
        # x.shape torch.Size([32, 3, 224, 224])

        

        feature = self.base(x)

        # print("feature.shape",feature.shape)
        # feature.shape torch.Size([32, 197, 768])

        cls_token = feature[:, 0, :]   # [B, 768]
        # print("cls_token.shape",cls_token.shape)
        # cls_token.shape torch.Size([32, 768])

        patch_tokens = feature[:, 1:, :]   # [B, 196, 768]
        # print("patch_tokens.shape",patch_tokens.shape)
        
        

        loss = torch.tensor(0.0, device=cls_token.device)

        
        #启用jpl loss时候打开
        cls_token = F.normalize(cls_token, p=2, dim=-1)  # [B, D]

        jpl =self.jpl(features = cls_token,labels = targets)

        jpl_loss = jpl["proxy_loss"]

        cls_token1 = jpl["class_similarities"]
        loss = jpl_loss+ loss
        
        # 启动jpl loss时候关闭
        # cls_token1 = self.classifier(cls_token)

        # print("cls_token.shape",logits.shape)
        # cls_token.shape torch.Size([4, 197, 2])
        if not self.image_mode:
            cls_token1 = cls_token1.view(-1, self.duration, self.num_classes)
            cls_token1 = torch.mean(cls_token1, dim=1)
        
        
        return cls_token1, cls_token, loss

        # if self.training:
        #     return logits, feature
        # else:
        #     return feature,loss


@register_model
def dinov3_vitb16(pretrained=False, check_hash: bool = False, **kwargs) -> torch.nn.Module:
    """
    加载DINOv3 ViT-Base (16x16 patch)模型及权重
    
    参数:
        weights_path_or_url: 权重文件的本地路径或URL
        check_hash: 是否验证权重文件的哈希值
    
    返回:
        加载好权重的DINOv3 ViT-Base模型
    """
    # if not hasattr(dinov3_vitb16, "_printed"):
    #     print("[dinov3_vitb16] incoming kwargs keys:", list(kwargs.keys()))
    
    num_classes = kwargs.pop("num_classes", 2)  # 默认=0
    

    weights_path_or_url = "/data/yiming.hao/Dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    
    # 调用底层构建函数，指定ViT-Base配置
    model = _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,          # ViT-Base特征维度
        depth=12,               # ViT-Base层数
        num_heads=12,           # ViT-Base注意力头数
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=True,
        weights=weights_path_or_url,  # 可以是本地路径或URL
        compact_arch_name="vitb",     # 指定ViT-Base架构
        check_hash=check_hash,
        **kwargs
    )
    # return model
    return Backbone(model, in_planes=768, num_classes=num_classes)


@register_model
def dinov3_vits16(weights_path_or_url: str, check_hash: bool = False, **kwargs) -> torch.nn.Module:
    """
    加载DINOv3 ViT-Small (16x16 patch)模型及权重
    ViT-Small 是 DINOv3 中轻量级架构,参数约21M,适合显存有限的场景
    
    参数:
        weights_path_or_url: 权重文件的本地路径或URL(需对应ViT-Small版本,如 dinov3_vits16_pretrain_lvd1689m.pth)
        check_hash: 是否验证权重文件的哈希值（确保权重完整性，建议开启）
    
    返回:
        加载好权重的DINOv3 ViT-Small模型
    """

    num_classes = kwargs.pop("num_classes", 2)  # 默认=0
    
    weights_path_or_url = "/data/yiming.hao/Dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    


    # 调用底层构建函数，指定ViT-Small核心配置（关键参数已按官方vits16标准调整）
    model = _make_dinov3_vit(
        img_size=224,               # 输入图像尺寸（与ViT-Base一致，DINOv3默认224x224）
        patch_size=16,              # Patch大小（16x16，与ViT-Base一致）
        in_chans=3,                 # 输入通道数（RGB图像，固定为3）
        # RoPE位置编码参数（与ViT-Base保持一致，确保预训练权重兼容）
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        # -------------------------- ViT-Small 核心架构参数 --------------------------
        embed_dim=384,              # 特征维度：ViT-Small为384（ViT-Base为768，减半）
        depth=12,                   #  transformer层数：ViT-Small为12（与Base一致，但每层参数更少）
        num_heads=6,                # 注意力头数：ViT-Small为6（Base为12，减半，确保每个头维度仍为64）
        # ----------------------------------------------------------------------------
        ffn_ratio=4,                # FFN层维度扩张比（与Base一致，4倍于embed_dim）
        qkv_bias=True,              # QKV投影层是否加偏置（官方默认开启）
        drop_path_rate=0.0,         # 随机深度概率（推理时设为0，训练时可调整）
        layerscale_init=1.0e-05,    # LayerScale初始化值（与Base一致，稳定训练）
        norm_layer="layernormbf16", # 归一化层类型（官方默认layernormbf16，适配bf16精度）
        ffn_layer="mlp",            # FFN层类型（默认MLP，也可改为swiglu）
        ffn_bias=True,              # FFN层是否加偏置（官方默认开启）
        proj_bias=True,             # 注意力输出投影层是否加偏置（官方默认开启）
        n_storage_tokens=4,         # 存储token数量（与Base一致，DINOv3核心设计，辅助特征存储）
        mask_k_bias=True,           # 掩码偏置（官方默认开启，提升掩码推理稳定性）
        pretrained=True,            # 启用预训练权重加载（必须设为True，否则不加载weights参数）
        weights=weights_path_or_url,# ViT-Small版本的权重路径/URL
        compact_arch_name="vits",   # 架构名称标识：ViT-Small对应"vits"（Base为"vitb"，Large为"vitl"）
        check_hash=check_hash,      # 权重哈希校验（建议设为True，避免权重损坏）
        **kwargs
    )
    #return model
    return Backbone(model, in_planes=384, num_classes=num_classes)


@register_model
def dinov3_vitl16(weights_path_or_url: str, check_hash: bool = False, **kwargs) -> torch.nn.Module:
    """
    加载DINOv3 ViT-Large (16x16 patch)模型及权重
    ViT-Large 是 DINOv3 中的中大型架构，参数规模更大，特征提取能力更强
    
    参数:
        weights_path_or_url: 权重文件的本地路径或URL(需对应ViT-Large版本,如 dinov3_vitl16_pretrain_lvd1689m.pth)
        check_hash: 是否验证权重文件的哈希值（确保权重完整性）
    
    返回:
        加载好权重的DINOv3 ViT-Large模型
    """

    num_classes = kwargs.pop("num_classes", 2)  # 默认=0

    
    weights_path_or_url = "/data/yiming.hao/Dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    

    # 构建ViT-Large模型，关键参数遵循DINOv3官方配置
    model = _make_dinov3_vit(
        img_size=224,               # 输入图像尺寸（DINOv3默认224x224）
        patch_size=16,              # Patch大小（16x16，与Small/Base保持一致）
        in_chans=3,                 # 输入通道数（RGB图像，固定为3）
        
        # RoPE位置编码参数（与DINOv3官方保持一致，确保预训练权重兼容）
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        
        # -------------------------- ViT-Large 核心架构参数 --------------------------
        embed_dim=1024,             # 特征维度：ViT-Large为1024（Small=384，Base=768）
        depth=24,                   # Transformer层数：24层（Small/Base为12层，深度翻倍）
        num_heads=16,               # 注意力头数：16头（确保每个头维度=64，1024/16=64）
        # ----------------------------------------------------------------------------
        
        ffn_ratio=4,                # FFN层维度扩张比（4倍于embed_dim，与Small/Base一致）
        qkv_bias=True,              # QKV投影层带偏置（官方默认配置）
        drop_path_rate=0.0,         # 推理阶段关闭随机深度
        layerscale_init=1.0e-05,    # LayerScale初始化值（稳定训练的关键参数）
        norm_layer="layernormbf16", # 归一化层类型（bf16精度的LayerNorm，官方默认）
        ffn_layer="mlp",            # FFN层类型（默认MLP，可根据权重文件调整为swiglu）
        ffn_bias=True,              # FFN层带偏置
        proj_bias=True,             # 注意力输出投影层带偏置
        n_storage_tokens=4,         # 存储token数量（DINOv3核心设计，4个寄存器token）
        mask_k_bias=True,           # 掩码偏置（提升掩码训练稳定性）
        pretrained=True,            # 启用预训练权重加载
        weights=weights_path_or_url,# ViT-Large版本的权重路径/URL
        compact_arch_name="vitl",   # 架构标识（ViT-Large对应"vitl"）
        check_hash=check_hash,       # 权重哈希校验（建议开启）
        **kwargs
    )
    # return model
    return Backbone(model, in_planes=1024, num_classes=num_classes)



