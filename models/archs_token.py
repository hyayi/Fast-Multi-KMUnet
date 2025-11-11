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
# from utils import * # utils 임포트는 주석 처리 (정의되지 않음)
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st
from kan import KANLinear  # 'kan' 라이브러리 설치 필요 (pip install pykan)
from torch.nn import init

import time
import math
from functools import partial
from typing import Optional, Callable
from archs import KANLayer, KANBlock  # 'archs'는 KANLayer/Block가 정의된 로컬 파일로 가정
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

# Mamba/SSM 관련 임포트
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except ImportError:
    pass

__all__ = ['UKANClsToken'] # 오타 수정 UKANCls -> UKANClsToken

# -------------------------------------------------------------------
# 1. DepthWise-Conv 헬퍼 모듈
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# 2. KAN 관련 모듈 (수정됨)
# -------------------------------------------------------------------

class KANLayerToken(KANLayer):
    """
    KANLayer를 상속받아 [CLS] 토큰을 처리하는 custom forward를 구현한 레이어.
    __init__에서 super()를 호출해 fc1, fc2를 상속받고,
    dwconv와 fc3를 추가로 정의합니다.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        
        # 1. KANLayer (부모)의 __init__을 호출
        # (이때 fc1, fc2, act, drop 등이 정의된다고 가정)
        super().__init__(in_features, hidden_features, out_features, act_layer, drop, no_kan)

        # 2. 이 클래스의 forward에서만 사용할 레이어 추가 정의
        dim = in_features
        self.dwconv_1 = DW_bn_relu(dim)
        self.dwconv_2 = DW_bn_relu(dim)
        
        # fc3 추가 정의 (KANLayer에는 fc1, fc2만 있다고 가정)
        # KANLinear 또는 nn.Linear를 사용 (no_kan 플래그에 따라)
        if no_kan:
            self.fc3 = nn.Linear(dim, dim)
        else:
            # KANLinear의 시그니처를 부모 클래스와 맞춘다고 가정
            self.fc3 = KANLinear(dim, dim, grid_size=5, spline_order=3) # 예시, KANLinear 시그니처에 맞춰야 함

        self.dwconv_3 = DW_bn_relu(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # (N-1)은 H*W와 같아야 합니다.

        # --- Block 1: fc1 -> dwconv_1 ---
        # (B, N, C) -> (B*N, C) -> (B*N, C) -> (B, N, C)
        # 참고: 부모 클래스의 self.act와 self.drop을 의도적으로 생략 (제공된 forward 로직 기준)
        x = self.fc1(x.reshape(B*N, C)).reshape(B, N, C)

        # CLS 분리
        cls_token = x[:, 0:1, :]    # (B, 1, C)
        patch_tokens = x[:, 1:, :]  # (B, N-1, C)

        # DWConv 적용 (패치 토큰에만)
        patch_tokens = self.dwconv_1(patch_tokens, H, W) 

        # 다시 결합
        x = torch.cat((cls_token, patch_tokens), dim=1) # (B, N, C)

        # --- Block 2: fc2 -> dwconv_2 ---
        x = self.fc2(x.reshape(B*N, C)).reshape(B, N, C)

        # CLS 분리
        cls_token = x[:, 0:1, :]    # (B, 1, C)
        patch_tokens = x[:, 1:, :]  # (B, N-1, C)
        
        # DWConv 적용 (패치 토큰에만)
        patch_tokens = self.dwconv_2(patch_tokens, H, W)
        
        # 다시 결합
        x = torch.cat((cls_token, patch_tokens), dim=1) # (B, N, C)
        
        # --- Block 3: fc3 -> dwconv_3 ---
        x = self.fc3(x.reshape(B*N, C)).reshape(B, N, C)

        # CLS 분리
        cls_token = x[:, 0:1, :]    # (B, 1, C)
        patch_tokens = x[:, 1:, :]  # (B, N-1, C)

        # DWConv 적용 (패치 토큰에만)
        patch_tokens = self.dwconv_3(patch_tokens, H, W)

        # 최종 결합
        x = torch.cat((cls_token, patch_tokens), dim=1) # (B, N, C)
    
        return x

class KANBlockEndocer(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim) # KANLayerToken은 in/out이 동일하다고 가정

        # KANLayerToken을 MLP 레이어로 사용
        self.layer = KANLayerToken(in_features=dim, hidden_features=mlp_hidden_dim, 
                                   act_layer=act_layer, drop=drop, no_kan=no_kan)

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
        # KANLayerToken의 forward(x, H, W)를 호출
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x

# -------------------------------------------------------------------
# 3. U-Net / ViT 구성 요소 (PatchEmbed, ConvLayer, CBAM)
# -------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # H, W 계산을 stride 기준으로 수정 (ViT/PVT 스타일)
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        
        # Padding 계산 수정 (stride > 1 일 때)
        padding = (patch_size[0] // 2, patch_size[1] // 2)
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
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

# --- CBAM 모듈 ---
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

# -------------------------------------------------------------------
# 4. 메인 모델 (UKANClsToken) - 수정된 로직 적용
# -------------------------------------------------------------------

class UKANClsToken(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], num_cls_classes=2, **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        # Encoder (Conv)
        self.encoder1 = ConvLayer(input_channels, kan_input_dim//8)  # 3 -> 16
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  # 16 -> 32
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim) # 32 -> 128 (embed_dims[0])

        # Norm Layers
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder (Token)
        self.block1 = nn.ModuleList([KANBlockEndocer(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlockEndocer(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        # Classification Head
        self.class_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dims[2], embed_dims[2] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dims[2] // 2, num_cls_classes)
        )

        # --- CLS Token 관련 수정 ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        
        # Stage 4(C1) -> Bottleneck(C2) 차원 프로젝션 레이어 추가
        self.cls_proj = nn.Linear(embed_dims[1], embed_dims[2])
        # -------------------------

        # Decoder (Token)
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        # Patch Embed (Downsampling)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # Decoder (Conv)
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        # Final Segmentation Head
        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        
        # CBAM
        self.cbam = CBAM(channel=kan_input_dim//8) # 16
        self.cbam1 = CBAM(channel=kan_input_dim//4) # 32
        self.cbam2 = CBAM(channel=kan_input_dim) # 128

    def forward(self, x):
        B = x.shape[0]

        ### Encoder
        ### Conv Stage
        
        # Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = self.cbam(out)
        
        # Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = self.cbam1(out)

        # Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = self.cbam2(out) # (B, C0, H/8, W/8) - C0=embed_dims[0]

        ### Tokenized KAN Stage
        ### Stage 4

        # (B, C0, H/8, W/8) -> (B, N1, C1) - N1=(H/16*W/16), C1=embed_dims[1]
        out, H, W = self.patch_embed3(out)
        
        # CLS 토큰 추가: (B, N1, C1) -> (B, N1+1, C1)
        out = torch.cat([self.cls_token.expand(B, -1, -1), out], dim=1) 
        
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W) # KANLayerToken의 forward(x, H, W) 호출
        out = self.norm3(out)
        
        # --- 💥 CLS / Patch 분리 로직 (수정됨) 💥 ---

        # 1. CLS 토큰과 패치 토큰 분리
        cls_token_out = out[:, 0:1, :]    # (B, 1, C1)
        patch_tokens_out = out[:, 1:, :]  # (B, N1, C1)

        # 2. 패치 토큰 -> Decoder Skip Connection (t4) 생성 (Decoder는 CLS 제외)
        t4 = patch_tokens_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, C1, H/16, W/16)

        ### Bottleneck

        # 3. CLS 토큰은 C1 -> C2 차원으로 프로젝션
        cls_token_2 = self.cls_proj(cls_token_out) # (B, 1, C2) - C2=embed_dims[2]

        # 4. 패치 토큰은 Conv로 공간적 다운샘플링 (t4 사용)
        patch_tokens_2, H, W = self.patch_embed4(t4) # (B, N2, C2) - N2=(H/32*W/32)

        # 5. Bottleneck KANBlock (block2)을 위해 다시 결합
        out = torch.cat((cls_token_2, patch_tokens_2), dim=1) # (B, N2+1, C2)

        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)

        # 6. Classification Head: 최종 CLS 토큰만 사용 (요청사항)
        cls_final = out[:, 0]                 # (B, C2)
        class_out = self.class_head(cls_final) # 👈 Classification 결과

        # 7. Decoder Input: 최종 패치 토큰만 사용 (요청사항)
        patches_final = out[:, 1:, :]         # (B, N2, C2)
        out = patches_final.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, C2, H/32, W/32)
        
        # --- 💥 로직 수정 끝 💥 ---


        ### Decoder
        ### Stage 4
        # 'out'은 (B, C2, H/32, W/32)
        # 't4'는 (B, C1, H/16, W/16)
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4) # Skip connection (CLS 토큰 없음)
        _, _, H, W = out.shape # (H/16, W/16)

        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W) # (B, N1, C1)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, C1, H/16, W/16)
        
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3) # Skip connection (t3)
        _, _, H, W = out.shape # (H/8, W/8)
        
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W) # (B, N0, C0)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, C0, H/8, W/8)

        # Conv Decoder
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        # final: Segmentation 결과
        return self.final(out), class_out