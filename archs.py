import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from fast_kan import FastKANLayer
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from torch.nn import init

import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

__all__ = ['UKANCls']

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_min: float = -2.
        grid_max: float = 2.
        num_grids: int = 8
        use_base_update: bool = True
        base_activation = F.silu
        spline_weight_init_scale: float = 0.1

        if not no_kan:
            self.fc1 = FastKANLayer(
                        in_features, 
                        hidden_features,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale
                    )
            self.fc2 = FastKANLayer(
                        hidden_features,
                        out_features,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale
                    )
            self.fc3 = FastKANLayer(
                        hidden_features,
                        out_features,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale
                    )
            # # TODO   
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )   

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)


        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# 添加cbam注意力机制（效果没变化）
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
# 添加注意力模块
import torch
from torch import nn
    
# 添加ss2d块

class UKANCls(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],num_cls_classes=2, **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(input_channels, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.class_head = FastKANLayer(
            embed_dims[2],num_cls_classes,
            grid_min=-2.,
            grid_max=2.,
            num_grids=8,
            use_base_update=True,
            base_activation=F.silu,
            spline_weight_init_scale=0.1
        )

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim =1)
        self.cbam = CBAM(channel=16)
        self.cbam1 = CBAM(channel=32)
        self.cbam2 = CBAM(channel=128)

    
# put shape: torch.Size([8, 3, 256, 256])
# After Stage 1 (encoder1) shape: torch.Size([8, 16, 128, 128])
# After Stage 2 (encoder2) shape: torch.Size([8, 32, 64, 64])
# After Stage 3 (encoder3) shape: torch.Size([8, 128, 32, 32])
# After Stage 4 (patch_embed3) shape: torch.Size([8, 256, 160]), H: 16, W: 16
# After norm3 and reshape shape: torch.Size([8, 160, 16, 16])
# After Bottleneck (patch_embed4) shape: torch.Size([8, 64, 256]), H: 8, W: 8
# After norm4 and reshape shape: torch.Size([8, 256, 8, 8])
# After decoder1 shape: torch.Size([8, 160, 16, 16])
# After add t4 shape: torch.Size([8, 160, 16, 16])
# After dblock1 shape: torch.Size([8, 256, 160])
# After dnorm3 and reshape shape: torch.Size([8, 160, 16, 16])
# After decoder2 shape: torch.Size([8, 128, 32, 32])
# After add t3 shape: torch.Size([8, 128, 32, 32])
# After dblock2 shape: torch.Size([8, 1024, 128])
# After dnorm4 and reshape shape: torch.Size([8, 128, 32, 32])
# After decoder3 shape: torch.Size([8, 32, 64, 64])
# After add t2 shape: torch.Size([8, 32, 64, 64])
# After decoder4 shape: torch.Size([8, 16, 128, 128])
# After add t1 shape: torch.Size([8, 16, 128, 128])
# After decoder5 shape: torch.Size([8, 16, 256, 256])

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        B = x.shape[0]

        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        # print(f"After Stage 1 (encoder1) shape: {out.shape}")
        t1 = out
        t1 = self.cbam(t1)
        
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        # print(f"After Stage 2 (encoder2) shape: {out.shape}")
        t2 = out
        t2 = self.cbam1(t2)

        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        # print(f"After Stage 3 (encoder3) shape: {out.shape}")
        t3 = out
        t3 = self.cbam2(t3)

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        # print(f"After Stage 4 (patch_embed3) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After norm3 and reshape shape: {out.shape}")
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        # print(f"After Bottleneck (patch_embed4) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)

        class_out = self.class_head(F.adaptive_avg_pool1d(out.transpose(1,2),1).squeeze(-1))
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After norm4 and reshape shape: {out.shape}")

        ### Decoder
        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder1 shape: {out.shape}")
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        # print(f"After add t4 shape: {out.shape}")

        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        # print(f"After dblock1 shape: {out.shape}")

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After dnorm3 and reshape shape: {out.shape}")
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder2 shape: {out.shape}")
        out = torch.add(out, t3)
        # print(f"After add t3 shape: {out.shape}")
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        # print(f"After dblock2 shape: {out.shape}")

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After dnorm4 and reshape shape: {out.shape}")

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder3 shape: {out.shape}")
        out = torch.add(out, t2)
        # print(f"After add t2 shape: {out.shape}")
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder4 shape: {out.shape}")
        out = torch.add(out, t1)
        # print(f"After add t1 shape: {out.shape}")
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder5 shape: {out.shape}")

        return self.final(out), class_out
            