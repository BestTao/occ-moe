# 双分支的MLP，原图使用MLP，遮挡图经过softmoe
import math
from functools import partial
from itertools import repeat
from typing import Any, ClassVar, Iterable, Mapping, Optional, Tuple, Type, Union,Dict
# from .moe_pytorch import sparse_moe_spmd
# from routing_pytorch import NoisyTopExpertsPerItemRouter
# from .routing_test import NoisyTopItemsPerExpertRouter
import numpy as np
import torch
import torch.nn as nn
# from timm.models import vision_transformer
import torch.nn.functional as F
# from torch._six import container_abcs
import collections.abc  # 导入 collections.abc 模块
from itertools import repeat
from .soft_moe2 import SoftMoE5
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):  # 使用 collections.abc.Iterable 替代
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, mlp_dim, hidden_features=3072, out_features=None, act_layer=nn.GELU, dropout_rate=0.):
        super().__init__()
        out_features = out_features or mlp_dim
        hidden_features = hidden_features or mlp_dim
        # print(f"测试mlp_dim：{mlp_dim}")
        # print(f"测试hidden_features：{hidden_features}")
        self.fc1 = nn.Linear(mlp_dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    
    
class MlpBlock(Mlp):
    def __init__(self, deterministic: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        self.deterministic = deterministic


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # print(f"进入MlpBlock")
        # 强制使用初始化时设定的 deterministic 值
        return super().forward(inputs)
    

class Attention(nn.Module):
    def __init__(self, dim, num_patches=128, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.dp = nn.Parameter(torch.ones((self.num_heads, self.num_patches)))

    def forward(self, x, attn_weight=None, alpha=0.1):
        B, N, C = x.shape # 64, 129, 768
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        patch_attn =  attn[:, :, 0, 1:]
        if attn_weight is not None:
            temp = attn.clone() #[64,8,129,129]
            # simple
            # w = attn_weight.unsqueeze(1).unsqueeze(2).expand(B, 1, N, self.num_patches)
            w = attn_weight.unsqueeze(1).unsqueeze(2)
            # dynamic
            # attn_weight_d = attn_weight.unsqueeze(1).expand(B, self.num_heads, self.num_patches) * self.dp.unsqueeze(0)
            # w = attn_weight_d.unsqueeze(2).expand(B, self.num_heads, N, self.num_patches)
            # w = attn_weight_d.unsqueeze(2)

            # simple fix
            # attn[:, :, :, 2:] = attn[:, :, :, 2:] * w * 0.9
            # weak fix
            ones = torch.ones_like(w)
            temp[:, :, 0:1, 1:] = attn[:, :, 0:1, 1:] * (w * alpha + ones * (1-alpha))
            attn = temp.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # cls_attn = attn[:, :, 0, -128:].detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        return x, patch_attn

class MoeBlock(nn.Module): 
    """Encoder block with a Sparse MoE of MLPs."""
    
    def __init__(self, mlp_block,dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-06)
                 ):
        super(MoeBlock, self).__init__()
        # if isinstance(mlp_block, MlpMoeBlock):
        #     self.mlp_block = mlp_block  # MoE层使用mlp_block前缀
        # else:
        #     self.mlp = mlp_block  # 普通层保持原mlp命名
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_block = mlp_block

    def forward(self, inputs,attn_weight=None, alpha=0.1,is_occluded=False):
        # print("进入MoeBlock")
        # Attention Block
        x = self.norm1(inputs)
        x_, patch_attn = self.attn(x, attn_weight=attn_weight, alpha=alpha)
        x = inputs + self.drop_path(x_)
        y = self.norm2(x)
        y_output = self.mlp_block(y)
        if isinstance(y_output, tuple):
            y, block_metrics = y_output
        else:
            y = y_output
            block_metrics = {}
        x = x + self.drop_path(y)
        return x, patch_attn,block_metrics
  
class MoeDualBlock(nn.Module): 
    """Encoder block with a Sparse MoE of MLPs."""
    
    def __init__(self, mlp_block, moe_mlp,dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-06)
                 ):
        super(MoeDualBlock, self).__init__()
        # if isinstance(mlp_block, MlpMoeBlock):
        #     self.mlp_block = mlp_block  # MoE层使用mlp_block前缀
        # else:
        #     self.mlp = mlp_block  # 普通层保持原mlp命名
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 双路径MLP
        self.mlp_block = mlp_block  # 原图路径
        self.moe_mlp = moe_mlp            # 遮挡图路径

    def forward(self, inputs,attn_weight=None, alpha=0.1,is_occluded=False):
        # print("进入MoeBlock")
        # Attention Block
        x = self.norm1(inputs)
        x_, patch_attn = self.attn(x, attn_weight=attn_weight, alpha=alpha)
        x = inputs + self.drop_path(x_)
        y = self.norm2(x)
        original_feat = self.mlp_block(y)  # 原图路径特征
        moe_feat = self.moe_mlp(y)        # 遮挡路径特征
        # y_output = self.mlp_block(y)
        if is_occluded:
            y_output = self.moe_mlp(y)
        else:
            y_output = self.mlp_block(y)

        if isinstance(y_output, tuple):
            y, block_metrics = y_output
        else:
            y = y_output
            block_metrics = {}
        x = x + self.drop_path(y)
        return x, patch_attn,block_metrics
        # return x, patch_attn,            {
        #         'original_feat': original_feat.detach(),  # 用于监督
        #         'moe_feat': moe_feat[0] if isinstance(moe_feat, tuple) else moe_feat,
        #         'aux_loss': moe_feat[1].get('auxiliary_loss', 0) if isinstance(moe_feat, tuple) else 0
        #     }


# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, attn_weight=None, alpha=0.1):
#         x_, patch_attn = self.attn(self.norm1(x), attn_weight=attn_weight, alpha=alpha)
#         x = x + self.drop_path(x_)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x, patch_attn

#     def forward(self, x, attn_weight=None, alpha=0.1):
#         x_, patch_attn = self.attn(self.norm1(x), attn_weight=attn_weight, alpha=alpha)
#         x = x + self.drop_path(x_)
#         mlp_out = self.mlp(self.norm2(x))
#         if isinstance(mlp_out, tuple):
#             mlp_out, moe_metrics = mlp_out  # 解包 MoE 输出和指标
#         else:
#             moe_metrics = {}
        
#         x = x + self.drop_path(mlp_out)
#         # x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x, patch_attn,moe_metrics

class old_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-06)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(mlp_dim=768, act_layer=act_layer, dropout_rate=drop)

    def forward(self, x, attn_weight=None, alpha=0.1):
        x_, patch_attn = self.attn(self.norm1(x), attn_weight=attn_weight, alpha=alpha)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, patch_attn

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):  # using this one
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():  # initialize
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x
    

class TransReID(nn.Module):  # as backbone, build_transformer_local.base
    """
        Transformer-based Object Re-Identification
        modified by zzw:
            1.add occlusion token
            2.output middle feature
            3. Moe
    """
    def __init__(self, encoder: dict,img_size=(256, 128), patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,mlp_dim: int=768,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,moe: Optional[dict] = None,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), local_feature=False, sie_xishu =1.0,deterministic: bool = False,attention_qk_norm: bool = False,
                dtype: torch.dtype = torch.float32,occ_aware=True, occ_block_depth=0, fix_alpha=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        self.encoder = encoder
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        self.patch_num = self.patch_embed.num_patches
        # print(moe)
        self.moe = moe or {'layers': (5,7,9,11),
                            'num_experts': 16,
                            'capacity_factor': 1.0,
                            'noise_std': 0.01, 
                            'compute_metrics': True,
                            'group_size': 4}
        self.occ_aware = occ_aware
        self.occ_block_depth = occ_block_depth
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.occ_aware:
            self.occ_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_occ_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # occ token for occ token but not load pretrain state
            self.fix_alpha = fix_alpha
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_num + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))


        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.moe_blocks = nn.ModuleList()
        self.moe_occ_blocks = nn.ModuleList()
        # self.moe_occ_blocks = nn.ModuleList()
        dense_mlp_params = dict(
            mlp_dim=mlp_dim,
            dropout_rate=drop_rate,
        )
        # print(f"self.moe:{self.moe}")
        moe_mlp_params = {**dense_mlp_params, 
                          **(self.moe)}
        moe_mlp_layers = moe_mlp_params.pop('layers', ())  
        for block_idx in range(depth):
            # ✅ 动态实例化 MLP
            # print(moe_mlp_layers)
            if block_idx in moe_mlp_layers:
                # print("使用MOE")
                # original_mlp = MlpBlock(**dense_mlp_params)  # 原始MLP
                original_mlp =  SoftMoE5(
                        dim=embed_dim,  # 必须与Transformer维度一致
                        num_experts=self.moe['num_experts'],
                        capacity_factor=self.moe.get('capacity_factor', 1.0),
                        noise_std=self.moe.get('noise_std', 0.01),
                        compute_metrics=self.moe.get('compute_metrics', True)
                    )

                moe_mlp = SoftMoE5(
                        dim=embed_dim,  # 必须与Transformer维度一致
                        num_experts=self.moe['num_experts'],
                        capacity_factor=self.moe.get('capacity_factor', 1.0),
                        noise_std=self.moe.get('noise_std', 0.01),
                        compute_metrics=self.moe.get('compute_metrics', True)
                    )
                block = MoeDualBlock(
                    mlp_block=original_mlp,
                    moe_mlp=moe_mlp,
                    dim=embed_dim,
                    num_heads=num_heads,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],)
            else:
                # print("不使用MOE") 
                mlp = MlpBlock(**dense_mlp_params)
                block = MoeBlock(
                mlp_block=mlp,  # 传入实例
                num_heads=num_heads,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[block_idx],
            )

            self.moe_blocks.append(block)
        if not self.occ_aware:
            self.old_blocks = nn.ModuleList([
                old_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-06))
                for i in range(depth)])
        
        if self.occ_block_depth != 0:
            dpr_occ = [x.item() for x in torch.linspace(0, drop_path_rate, occ_block_depth)]  # stochastic depth decay rule
            for block_idx2 in range(occ_block_depth):
                if block_idx2 in moe_mlp_layers:
                    mlp = SoftMoE5(
                        dim=embed_dim,  # 必须与Transformer维度一致
                        num_experts=self.moe['num_experts'],
                        capacity_factor=self.moe.get('capacity_factor', 1.0),
                        noise_std=self.moe.get('noise_std', 0.1),
                        compute_metrics=self.moe.get('compute_metrics', True)
                    )
                else:
                    mlp = MlpBlock(**dense_mlp_params)
                # ✅ 创建编码块实例
                moe_block = MoeBlock(
                    mlp_block=mlp,  # 传入实例
                    num_heads=num_heads,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr_occ[block_idx2],
                )
                self.moe_occ_blocks.append(moe_block)

            # self.moe_occ_blocks =  nn.ModuleList([
            #     old_Block(
            #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_occ[i], norm_layer=norm_layer)
            #     for i in range(occ_block_depth)])
        if self.occ_aware:
            self.occ_pred = nn.Linear(embed_dim, 2*self.patch_num, bias=True)
            # self.occ_pred = nn.Linear(embed_dim, 3*self.patch_num, bias=True)


        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id, mid_feature=0, final_depth=11, occ_fix=False):
        """

        Args:
            x:
            camera_id:
            view_id:
            mid_feature:  output middle feat after n blocks n in [1, depth-1]
            final_depth: final feat after n blocks

        Returns:

        """
        B = x.shape[0]
        encoder_kwargs = dict(self.encoder)
        if encoder_kwargs.get('position_emb', {}).get('name') == 'sincos2d':
            encoder_kwargs['position_emb'] = dict(encoder_kwargs['position_emb'])
            encoder_kwargs['position_emb']['h'] = x.shape[2]
            encoder_kwargs['position_emb']['w'] = x.shape[3]
        # x切割为小块x
        # print(x.shape)
        x = self.patch_embed(x)
        if self.occ_aware: # 添加遮挡预测的编码
            occ_tokens = self.occ_token.expand(B, -1, -1)
            x = torch.cat((occ_tokens, x), dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.occ_aware: # 添加位置编码
            pos_embed_ = torch.cat([self.pos_embed[:, :1], self.pos_embed_occ_token, self.pos_embed[:, 1:]], dim=1)
        else:
            pos_embed_ = self.pos_embed

        if self.cam_num > 0 and self.view_num > 0:
            x = x + pos_embed_ + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + pos_embed_ + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + pos_embed_ + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + pos_embed_

        x = self.pos_drop(x)

        if self.occ_aware and self.occ_block_depth != 0:
            x_ = x  # backup
            for i,blk in enumerate(self.moe_occ_blocks):
                x, _ ,_= blk(x)
            print(f"原始的Transreid的self.occ_blocks的x形状：{x.shape}")
            # print(x.shape) # torch.Size([8, 130, 768])
            # tt = x[:, 1, :] 
            # print(tt.shape) # torch.Size([8, 768])
            # t2 = self.occ_pred(x[:, 1, :]) # torch.Size([8, 256])
            # print(t2.shape)
             # print(x.shape):torch.Size([8, 130, 768]);x[:, 1, :]shape为torch.Size([8, 768]);
            # print("进行了self.occ_pred")
            occ_pred = self.occ_pred(x[:, 1, :]).reshape((-1, self.patch_num, 2))
            # print(f"occ_pred:{occ_pred.shape}") # torch.Size([8, 128, 2])
            occ_score = occ_pred.detach().softmax(dim=-1)  
            # print(f"occ_score:{occ_score.shape}") # torch.Size([64, 128, 2])
            attn_weight = occ_score[:, :, 0]
            # print(f"attn_weight:{attn_weight.shape}")# torch.Size([64, 128])
            x = torch.cat([x_[:, 0:1], x_[:, 2:]], dim=1)
            fix_alpha = [x.item() for x in torch.linspace(0, self.fix_alpha, final_depth)]
            metrics={}
            # for i, blk in enumerate(self.moe_blocks[:final_depth]):
            for i, blk in enumerate(self.moe_blocks[:final_depth]):    
                if occ_fix:
                    # print(occ_fix)
                    # print(f"输入x:{x.shape}")
                    x, attn,blk_metrics= blk(x, attn_weight, fix_alpha[i],is_occluded=occ_fix)
                    # print(x.shape)
                    # print(f"中间打印损失：{blk_metrics}")
                    # print(f"循环中的metrics{blk_metrics}")
                else:
                    # print(occ_fix)
                    # print(f"可能错误的输入形状：{x.shape}")
                    x, attn,blk_metrics = blk(x)
                if isinstance(blk_metrics, dict) and blk_metrics:  # 假设 MoE 块返回指标
                    # print("执行copy")
                    metrics[f'encoderblock_{i}'] = blk_metrics.copy()
            
            metrics['auxiliary_loss'] = sum(m['auxiliary_loss'] for m in metrics.values())
            # print(f"主循环后的metrics：{metrics}")
        elif self.occ_aware:
            assert mid_feature < final_depth <= len(self.moe_blocks), "mid or final depth out of range!"
            print("走错了")
            for blk in self.moe_blocks[:mid_feature]:
                x, attn,metrics = blk(x)
                                                                    
            if mid_feature > 0:
                x_mid = x
                occ_pred = self.occ_pred(x_mid[:, 1, :]).reshape((-1, self.patch_num, 2))
                occ_score = occ_pred.detach().softmax(dim=-1)
                attn_weight = occ_score[:, :, 0]
            else:
                occ_pred = None
                attn_weight = None

            fix_alpha = [x.item() for x in torch.linspace(0, 0.1, final_depth-mid_feature)]

            for i, blk in enumerate(self.moe_blocks[mid_feature:final_depth]):
                if occ_fix:
                    x, attn,metrics = blk(x, attn_weight, fix_alpha[i])
                else:
                    x, attn,metrics = blk(x)
            x = torch.cat([x[:, 0:1], x[:, 2:]], dim=1)  # exclude occ token which is useless for loss compute
        else:
            print("走错了old_blocks")
            for blk in self.old_blocks[:final_depth]:
                x, attn = blk(x)
            occ_pred = None

        # attn_maps = torch.cat([a.unsqueeze(1) for a in attn_list], dim=1)

        return x, occ_pred, attn,metrics['auxiliary_loss']

    def forward(self, x, cam_label=None, view_label=None, mid_feature=0, final_depth=11, occ_fix=False):
        x, occ_pred, attn,metrics = self.forward_features(x, cam_label, view_label, mid_feature, final_depth, occ_fix)
        return x, occ_pred, attn,metrics
    # def load_param(self, model_path):
    #     param_dict = torch.load(model_path, map_location='cpu')
    #     if 'model' in param_dict:
    #         param_dict = param_dict['model']
    #     if 'state_dict' in param_dict:
    #         param_dict = param_dict['state_dict']
    #     for k, v in param_dict.items():
    #         if 'head' in k or 'dist' in k:
    #             continue
    #         if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
    #             # For old models that I trained prior to conv based patchification
    #             O, I, H, W = self.patch_embed.proj.weight.shape
    #             v = v.reshape(O, -1, H, W)
    #         elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
    #             # To resize pos embedding when using model at different size from pretrained weights
    #             if 'distilled' in model_path:
    #                 print('distill need to choose right cls token in the pth')
    #                 v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
    #             v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
    #         try:
    #             self.state_dict()[k].copy_(v)
    #         except:
    #             print('===========================ERROR=========================')
    #             print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        # 键名转换规则（注意全部使用小写）
        key_mapping = [
            ('blocks.', 'moe_blocks.'),       # 主模块前缀
            ('.mlp.fc1.', '.mlp_block.fc1.'),  # 普通层MLP路径
            ('.mlp.fc2.', '.mlp_block.fc2.')
        ]
        
        model_dict = self.state_dict()
        updated_keys = []
        skipped_keys = []
        
        for raw_k, v in param_dict.items():
            if 'head' in raw_k or 'dist' in raw_k:
                continue
                
            # 执行键名转换 -------------------------------------------------
            k = raw_k
            for old, new in key_mapping:
                k = k.replace(old, new)
            # Step 2: 处理特殊参数形状
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
                
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)            
            # 特殊处理MoE层 -------------------------------------------------
            if 'moe_blocks.' in k:
                # 提取层号 (e.g. "moe_blocks.3.norm1" -> 3)
                layer_idx = int(k.split('.')[1])
                # 如果该层是MoE层且参数属于原始MLP部分
                if layer_idx in self.moe_blocks and 'mlp_block.mlp.' in k:
                    print(f'[Skip] MoE层参数: {raw_k} -> {k}')
                    skipped_keys.append(k)
                    continue
                    
            # 处理特殊参数形状 -----------------------------------------------
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
                
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            
            # 参数加载 -----------------------------------------------------
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k].copy_(v)
                    updated_keys.append(k)
                else:
                    print(f'[Mismatch] {k}: pretrained {v.shape} vs model {model_dict[k].shape}')
                    skipped_keys.append(k)
            else:
                print(f'[Missing] {k} (原始键名: {raw_k})')
                skipped_keys.append(k)
        
        # 打印统计信息 -----------------------------------------------------
        print(f'\n===== 参数加载统计 =====')
        print(f'成功加载: {len(updated_keys)}/{len(param_dict)}')
        print(f'跳过参数: {len(skipped_keys)}')
        print(f'匹配率: {len(updated_keys)/len(param_dict):.1%}')
        
        # 可选：打印前5个成功加载的键示例
        print('\n----- 成功加载示例 -----')
        for k in updated_keys[:5]:
            print(f'+ {k}')
        
        # 可选：打印前5个跳过的键示例
        if skipped_keys:
            print('\n----- 跳过键示例 -----')
            for k in skipped_keys[:5]:
                print(f'- {k}')


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, occ_aware=True, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, \
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, occ_aware=occ_aware, **kwargs)

    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate = drop_path_rate,\
        camera=camera, view=view,  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def deit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



# batch_size = 64
# channels = 3  # RGB 图像
# height = 256
# width = 128
# patch_size = (16, 16)
# num_classes = None  # 假设有 10 个类别
# hidden_size = 768  # ViT-base 使用 768 维隐藏层
# num_layers = 12
# num_heads = 12
# mlp_dim = 768

# # 构造一个随机的输入
# inputs = torch.randn(batch_size, channels, height, width)

# # 定义 ViT MoE 的参数
# encoder_config = {
#     'num_layers': 12,
#     'mlp_dim': 8,
#     'num_heads': 12,
#     # 'dropout_rate': 0.1,
#     'attention_dropout_rate': 0.1
# }
# moe={'layers': (3, 6, 9),  # (2,5,8), (3,5,7,9), (4,7,10)
#     'num_experts': 8, # 小规模数据集（<50k图像）：4-8个专家, 中大规模（50k-200k）：8-16个专家
#     'group_size': 4, # 2-8
#     'router': {
#         'num_selected_experts': 1,  # 1-2
#         'noise_std': 1e-3, # 1e-4 到 1e-2
#         'importance_loss_weight': 0.02, # 0.01 到 0.1
#         'load_loss_weight': 0.02, # 0.01 到 0.1
#         'dispatcher': {
#             'name': 'einsum',
#             'capacity': 2,
#             'batch_priority': False,
#             'bfloat16': False,
#         }
#     },
#             }  # 指定 MoE 层

# img_size=(256, 128)
# stride_size=16 
# drop_rate=0.0
# attn_drop_rate=0.0
# drop_path_rate=0.1
# camera=0
# view=0
# local_feature=False
# sie_xishu=1.5
# occ_aware=True
# occ_block_depth=3

# test_model = TransReID(encoder=encoder_config,
#         img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, \
#         camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,moe = moe,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, occ_aware=occ_aware,occ_block_depth=occ_block_depth)

# out,occ_pred,atten,metrics = test_model(inputs,occ_fix=True)
# print(test_model)
# print("打印结果：")
# print(f"out.shape:{out.shape}") # out.shape:torch.Size([8, 129, 768])
# print(f"occ_pred:{occ_pred.shape}") # occ_pred:torch.Size([8, 128, 2])
# print(f"atten:{atten.shape}") # atten:torch.Size([8, 12, 128])
# print(f"metrics_loss:{metrics}")


# from omegaconf import OmegaConf
# from functools import partial

# # 加载配置文件
# with open("/data/zht/learn_pytorch/megaz-reid-master/megaz-reid-master/model/backbones/test.yml") as f:
#     cfg = OmegaConf.load(f)

# # 转换元组类型参数（OmegaConf 默认会将列表转换为ListConfig）
# cfg.model.img_size = tuple(cfg.model.img_size)
# cfg.moe.layers = tuple(cfg.moe.layers)

# # 构建模型
# test_model = TransReID(
#     encoder=cfg.encoder,
#     img_size=cfg.model.img_size,
#     patch_size=cfg.model.patch_size,
#     stride_size=cfg.model.stride_size,
#     embed_dim=cfg.model.embed_dim,
#     depth=cfg.model.depth,
#     num_heads=cfg.model.num_heads,
#     mlp_ratio=cfg.model.mlp_ratio,
#     qkv_bias=cfg.model.qkv_bias,
#     camera=cfg.model.camera,
#     view=cfg.model.view,
#     drop_path_rate=cfg.model.drop_path_rate,
#     drop_rate=cfg.model.drop_rate,
#     attn_drop_rate=cfg.model.attn_drop_rate,
#     sie_xishu=cfg.model.sie_xishu,
#     local_feature=cfg.model.local_feature,
#     occ_aware=cfg.model.occ_aware,
#     occ_block_depth=cfg.model.occ_block_depth,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     moe=cfg.moe
# )

# # 构造输入
# inputs = torch.randn(
#     cfg.training.batch_size,
#     cfg.training.channels,
#     cfg.training.height,
#     cfg.training.width
# )

# # 前向传播
# out, occ_pred, atten, metrics = test_model(inputs, occ_fix=True)
# print("打印结果：")
# print(f"out.shape:{out.shape}") # out.shape:torch.Size([8, 129, 768])
# print(f"occ_pred:{occ_pred.shape}") # occ_pred:torch.Size([8, 128, 2])
# print(f"atten:{atten.shape}") # atten:torch.Size([8, 12, 128])
# print(f"metrics_loss:{metrics}")


# def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, occ_aware=False, **kwargs):
#     model = TransReID(
#         img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, \
#         camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, occ_aware=occ_aware, **kwargs)

#     return model

# def my_vit_base_patch16_224_TransReID(cfg=None, **kwargs):
#     """
#     构建 TransReID 模型的工厂函数，支持配置参数和动态覆盖
    
#     参数：
#     cfg - OmegaConf 配置对象（包含模型参数）
#     **kwargs - 显式覆盖参数（优先级高于cfg）
#     """
#     # 默认参数（当cfg未提供时使用）
#     default_args = {
#         "img_size": (256, 128),
#         "patch_size": 16,
#         "stride_size": 16,
#         "embed_dim": 768,
#         "depth": 12,
#         "num_heads": 12,
#         "mlp_ratio": 4,
#         "qkv_bias": True,
#         "camera": 0,
#         "view": 0,
#         "drop_path_rate": 0.1,
#         "drop_rate": 0.0,
#         "attn_drop_rate": 0.0,
#         "sie_xishu": 1.5,
#         "local_feature": False,
#         "occ_aware": False,
#         "occ_block_depth": 3,
#         "norm_layer": partial(nn.LayerNorm, eps=1e-6)
#     }
    
#     # 合并配置参数（优先级：显式参数 > cfg > 默认参数）
#     final_args = default_args.copy()
    
#     if cfg is not None:
#         # 处理需要类型转换的参数
#         cfg_params = OmegaConf.to_container(cfg.model, resolve=True)
#         cfg_params["img_size"] = tuple(cfg_params["img_size"])
        
#         # 合并配置参数
#         final_args.update(cfg_params)
    
#     # 应用显式覆盖参数（最高优先级）
#     final_args.update(kwargs)
    
#     # 处理特殊参数
#     if "encoder" not in final_args:
#         final_args["encoder"] = OmegaConf.to_container(cfg.encoder, resolve=True)
        
#     if "moe" in cfg:
#         final_args["moe"] = OmegaConf.to_container(cfg.moe, resolve=True)
    
#     # 构建模型
#     model = TransReID(
#         encoder=final_args.get("encoder"),
#         img_size=final_args["img_size"],
#         patch_size=final_args["patch_size"],
#         stride_size=final_args["stride_size"],
#         embed_dim=final_args["embed_dim"],
#         depth=final_args["depth"],
#         num_heads=final_args["num_heads"],
#         mlp_ratio=final_args["mlp_ratio"],
#         qkv_bias=final_args["qkv_bias"],
#         camera=final_args["camera"],
#         view=final_args["view"],
#         drop_path_rate=final_args["drop_path_rate"],
#         drop_rate=final_args["drop_rate"],
#         attn_drop_rate=final_args["attn_drop_rate"],
#         sie_xishu=final_args["sie_xishu"],
#         local_feature=final_args["local_feature"],
#         occ_aware=final_args["occ_aware"],
#         occ_block_depth=final_args["occ_block_depth"],
#         norm_layer=final_args["norm_layer"],
#         moe=final_args.get("moe")  # 可选参数使用 get()
#     )
    
#     return model

# cfg = OmegaConf.load("/data/zht/learn_pytorch/megaz-reid-master/megaz-reid-master/model/backbones/test.yml")
# model1 = my_vit_base_patch16_224_TransReID(cfg=cfg)
# out, occ_pred, atten, metrics = model1(inputs, occ_fix=True)
# print("打印结果：")
# print(f"out.shape:{out.shape}") # out.shape:torch.Size([8, 129, 768])
# print(f"occ_pred:{occ_pred.shape}") # occ_pred:torch.Size([8, 128, 2])
# print(f"atten:{atten.shape}") # atten:torch.Size([8, 12, 128])
# print(f"metrics_loss:{metrics}")