import sys
import os 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
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
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st
from kan import KANLinear
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

__all__ = ['UKANClsSSPScaleMLP']

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # X축, Y축 각각 Pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 채널 압축 비율 설정 (최소 8 채널 보장)
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() # ReLU보다 성능이 좋은 최신 Activation

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Height 방향 Pooling -> [N, C, H, 1]
        x_h = self.pool_h(x)
        # Width 방향 Pooling -> [N, C, 1, W] -> permute -> [N, C, W, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenation & Convolution
        y = torch.cat([x_h, x_w], dim=2) # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # Split for X and Y
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) # Restore shape

        # Attention Map Generation
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Attention Apply
        return identity * a_h * a_w

# ---------------------------------------------------------
# 2. Spatial Pyramid Pooling (SPP)
#    다양한 크기의 Grid로 나누어 특징을 추출 (Flatten 포함)
# ---------------------------------------------------------
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[1, 4, 8]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        pooled_features = []
        
        for size in self.pool_sizes:
            # size x size 그리드로 Pooling
            tensor = F.adaptive_avg_pool2d(x, output_size=(size, size))
            # 벡터로 펼쳐서 리스트에 저장
            pooled_features.append(tensor.view(batch_size, -1))
            
        # 모든 레벨의 특징 벡터를 이어 붙임
        return torch.cat(pooled_features, dim=1)

# ---------------------------------------------------------
# 3. Final Optimized Classifier Head
#    (Reducer -> CA -> SPP -> Linear)
# ---------------------------------------------------------
class MultiTask_Classifier_Head(nn.Module):
    def __init__(self, in_channels=256, reduced_channels=128, num_classes=2,pooling_sizes=[1,2,4],reduction=16):
        super(MultiTask_Classifier_Head, self).__init__()
        
        # print(f"[Init] Classifier Head: Input({in_channels}) -> Reduce({reduced_channels}) -> SPP[1,4,8]")

        # Step 1: Channel Reduction (파라미터 절약)
        self.reducer = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )
        self.meta_net = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Linear(32, reduced_channels * 2) # Gamma(128) + Beta(128)
                )
        
        # Step 2: Coordinate Attention (위치 정보 강화)
        # reduction=16: 내부에서 128->8로 압축했다가 복구 (효율적)
        self.ca = CoordAtt(reduced_channels, reduced_channels, reduction=reduction)
        
        # Step 3: SPP (세밀한 공간 정보 수집)
        self.spp = SpatialPyramidPooling(pool_sizes=pooling_sizes)
        
        # Step 4: Classification Layers
        # Input Dim 계산: 128 * (1 + 16 + 64) = 128 * 81 = 10,368
        spp_output_dim = reduced_channels * (sum([size * size for size in pooling_sizes]))
        print(f"[Init] SPP Output Dim: {spp_output_dim},{reduced_channels} ")
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(spp_output_dim),
            KANLinear(spp_output_dim, spp_output_dim//2),
            nn.BatchNorm1d(spp_output_dim//2),
            nn.Dropout(0.2),
            KANLinear(spp_output_dim//2, num_classes)      
        )
        # self.classifier = nn.Sequential(
        #     # ----------------------------------------------------------
        #     # TIP: KANLinear 사용 시 아래 nn.Linear 대신 교체하세요.
        #     # ex) KANLinear(spp_output_dim, 64)  <-- Hidden Dim 줄일 것!
        #     # ----------------------------------------------------------
        #     nn.Linear(spp_output_dim, 256), 
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5), # 파라미터가 5M 정도 되므로 Dropout 필수
        #     nn.Linear(256, num_classes)
        # )

    def forward(self, x, spacing):
            # x: [Batch, 256, 32, 32]
            # spacing: [Batch, 1] (단위: mm/pixel 등)
            
            # (1) Reduce Channels
            x = self.reducer(x) # [B, 128, 32, 32]
            
            # (2) Feature Modulation (핵심!) ---------------------------
            # Spacing 값으로 Scale(gamma)과 Shift(beta)를 예측
            stats = self.meta_net(spacing) # [B, 256]
            gamma, beta = stats.chunk(2, dim=1) # 각각 [B, 128]
            
            # 차원 맞추기: [B, 128] -> [B, 128, 1, 1]
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
            # Affine Transformation 적용: (Feature * Scale) + Shift
            # 1.0을 더해주는 이유는 초기 학습 시 gamma가 0 근처일 때 특징이 사라지는 것 방지
            x = x * (1.0 + gamma) + beta 
            # ----------------------------------------------------------
            
            # (3) Attention & Pooling & Classification
            x = self.ca(x)
            x = self.spp(x)
            x = self.classifier(x)
            
            return x


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]


        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
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

class UKANClsSSPScaleMLP(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],num_cls_classes=2,pooling_sizes=[1,2,4],reduction=16, **kwargs):
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

        # --- 기존 class_head 삭제 ---
        # self.class_head = nn.Sequential(...)

        # --- 새로운 CNN 기반 class_head 정의 ---
        # 입력: t3 (B, embed_dims[0], H/8, W/8)
        # 예: (B, 128, 128, 128)
        self.clssfn_head = MultiTask_Classifier_Head(
            in_channels=embed_dims[2],
            reduced_channels=embed_dims[1] // 2,
            num_classes=num_cls_classes,
            pooling_sizes=pooling_sizes,
            reduction=reduction
        )
        # --- (수정 끝) ---

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
        
        
    def forward(self, x, spacing):
        B = x.shape[0]

        ### Encoder
        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out

        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        # print(f"After Stage 3 (encoder3) shape: {t3.shape}")

        # --- (수정) 새로운 분류 분기 ---
        # t3 (CNN 인코더의 최종 출력)에서 분기
        # --- (수정 끝) ---

        ### Tokenized KAN Stage
        ### Stage 4
        
        # segmentation 경로는 t3가 아닌 'out' (t3와 동일)을 사용합니다.
        # (참고: t3 = self.cbam2(out)이므로, KAN 경로는 CBAM 적용된 것을 사용합니다)
        out, H, W = self.patch_embed3(t3) 
        # print(f"After Stage 4 (patch_embed3) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        # print(f"After Bottleneck (patch_embed4) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)

        # --- (수정) 기존 분류 헤드 제거 ---
        # pooled_features = out.mean(dim=1)
        # class_out = self.class_head(pooled_features)
        # --- (수정 끝) ---
        
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        cls_out = self.clssfn_head(out,spacing)  # CNN 기반 분류 헤드 사용
        # print(f"Classification output shape: {cls_out.shape}")
        # print(f"After norm4 and reshape shape: {out.shape}")

        ### Decoder
        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        # class_out은 이미 위에서 계산되었습니다.
        return self.final(out), cls_out

if __name__ == "__main__":
    import time

    # 1. 장치 설정
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. 모델 초기화 및 장치로 이동
    model = UKANClsSSPScaleMLP(num_classes=2, input_channels=1, img_size=1024, patch_size=16, in_chans=3,
      embed_dims=[128, 160, 256], num_cls_classes=2, pooling_sizes=[1,2,4], reduction=16)
    model.to(device)
    model.eval() # 추론 모드로 설정 (Dropout 등 비활성화)

    # 3. 파라미터 수 계산 (학습 가능한 파라미터만)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1_000_000:.2f} M")

    # 4. 더미 입력 데이터 생성 (장치로 이동)
    # 참고: (8, 3, 1024, 1024) 배치는 VRAM이 많이 필요할 수 있습니다.
    try:
        x = torch.randn((1, 1, 1024, 1024)).to(device)
        spancing = torch.tensor([[1.0,1.0]]).to(device)  # Dummy spacing tensor
        print(f"Input shape: {x.shape}")
    except RuntimeError as e:
        print(f"Error creating input tensor (likely OOM): {e}")
        print("Batch size 8 may be too large for 1024x1024. Exiting.")
        exit()

    # 5. 웜업(Warm-up) 실행 (정확한 시간 측정을 위해)
    print("Running warm-up pass...")
    with torch.no_grad():
        _ = model(x, spancing)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 6. 추론 시간 측정
    print("Running timed inference...")
    
    start_time = time.time()
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize() # CUDA 커널 실행 동기화 (시작)
            
        seg_out, cls_out = model(x, spancing)
        
        if device.type == 'cuda':
            torch.cuda.synchronize() # CUDA 커널 실행 동기화 (종료)
            
    end_time = time.time()
    
    duration = end_time - start_time
    batch_size = x.shape[0]
    fps = batch_size / duration # Frames Per Second

    print("\n--- Results ---")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Inference time for batch of {batch_size}: {duration:.4f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")