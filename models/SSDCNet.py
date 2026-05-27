import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import scipy.io as sio
from torch.autograd import Variable
import numbers
from einops import rearrange
import os, math, gc, importlib
from torch.utils.cpp_extension import load

from models.SCFM import SCFM
from models.GAWS import GAWS

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_HEAD_SIZE_A"] = str(64)


# ======================
# 基础工具函数
# ======================

def to_3d(x):
    # x: [B, C, H, W] -> [B, H*W, C]
    b, c, h, w = x.shape
    return x.view(b, c, h * w).transpose(1, 2)


def to_4d(x, h, w):
    # x: [B, H*W, C] -> [B, C, H, W]
    b, n, c = x.shape
    return x.transpose(1, 2).view(b, c, h, w)


class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, hidden_dim=256, top_k=2, tau=1.3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = tau    # ★ 新增：softmax 温度

        # 路由器
        self.router = nn.Linear(input_dim, num_experts)

        # 专家网络（每个都是一个简单 MLP）
        self.experts = nn.ModuleList([
            MLP(input_dim, output_dim, [hidden_dim])
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: [T, C_in]
        return: [T, C_out]
        """
        # ------------------------------
        # 1) 路由器 logits
        # ------------------------------
        logits = self.router(x)  # [T, E]

        # ------------------------------
        # 2) 温度 softmax
        # scores = softmax(logits / tau)
        # tau < 1 → 更均匀，top-k 更稳定
        # tau > 1 → 更尖锐，让专家更分工明确
        # ------------------------------
        scores = F.softmax(logits / self.tau, dim=-1)   # ★ 使用 tau

        # ------------------------------
        # 3) top-k 稀疏 MoE
        # ------------------------------
        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)  # [T, k]

        # 归一化，让 top-k 和为 1
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-8)

        T = x.size(0)
        outputs = torch.zeros(T, self.experts[0].layers[-1].out_features,
                              device=x.device, dtype=x.dtype)

        # ------------------------------
        # 4) 根据专家 ID 聚合
        # ------------------------------
        for i in range(self.top_k):
            expert_id = top_idx[:, i]               # [T]
            weight = top_scores[:, i].unsqueeze(-1) # [T, 1]

            for e in range(self.num_experts):
                mask = (expert_id == e)
                if mask.any():
                    x_e = x[mask]                   # [T_e, C_in]
                    out_e = self.experts[e](x_e)    # [T_e, C_out]
                    outputs[mask] += weight[mask] * out_e

        return outputs

# ======================
# Prior：空间–光谱多尺度 backbone（方向 B）
# ======================

class Prior(nn.Module):
    def __init__(self, hsi_channels):
        super(Prior, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels,
                      kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=hsi_channels, out_channels=hsi_channels,
                      kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_channels=hsi_channels * 3, out_channels=hsi_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, H):
        """
        H: [B, C, H, W]  （这里 C = hsi_channels）
        """
        out1 = self.conv1(H)
        out2 = self.conv2(H)
        out3 = self.conv3(H)
        out_cat = torch.cat([out1, out2, out3], dim=1)  # [B, 3C, H, W]
        out = self.cat_conv(out_cat)                    # [B, C, H, W]
        result = H + out                                # 残差连接
        return result


# ======================
# SpeUB / SpaUB & LN
# ======================
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# 通道注意力（用于 SpeUB）
# -----------------------
class ChannelSE(nn.Module):
    """
    Squeeze-and-Excitation 式的通道注意力
    """
    def __init__(self, channels, reduction=4):
        super(ChannelSE, self).__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))       # [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -----------------------
# 光谱上采样 SpeUB_v2
# -----------------------
class SpeUB_v2(nn.Module):
    """
    光谱上采样改进版：
    1) 采用 1x1 conv 做光谱映射（逐像素 MLP）
    2) 用 1x1 conv 回到 MSI 空间做残差
    3) 用 depthwise 3x3 卷积做轻量空间平滑
    4) 引入通道注意力（SE）
    """
    def __init__(self, hsi_channels, msi_channels, hidden_ratio=2):
        super(SpeUB_v2, self).__init__()
        hidden = max(int(msi_channels * hidden_ratio), hsi_channels)

        # MSI -> HSI 的光谱映射（像素级 MLP）
        self.spec_map1 = nn.Sequential(
            nn.Conv2d(msi_channels, hidden, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hsi_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # HSI -> MSI 的反向映射（估计 MSI，用于构造残差）
        self.spec_back = nn.Sequential(
            nn.Conv2d(hsi_channels, msi_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 对残差做进一步的光谱+空间映射
        # 先用 1x1 做光谱，后用 depthwise 3x3 做空间平滑
        self.residual_refine = nn.Sequential(
            nn.Conv2d(msi_channels, hsi_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=3, stride=1, padding=1,
                      groups=hsi_channels),  # depthwise
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.se = ChannelSE(hsi_channels)

    def forward(self, HRMS):
        # HRMS: [B, C_msi, H, W]
        Y1 = self.spec_map1(HRMS)   # 初始光谱上采样到 HSI 空间

        # 估计回 MSI，构造光谱残差
        Y2 = self.spec_back(Y1)     # 预测 MSI
        mid = HRMS - Y2             # MSI 空间的残差

        # 把残差映射回 HSI，并做轻量空间平滑
        Y3 = self.residual_refine(mid)

        out = Y1 + Y3               # 残差修正后的 HSI 表达
        out = self.se(out)          # 通道注意力再重标定一次
        return out


# -----------------------
# 轻量空间注意力（用于 SpaUB）
# -----------------------
class SpatialAttention(nn.Module):
    """
    只做简单的 spatial attention：avg + max pooling -> 7x7 conv -> sigmoid
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


# -----------------------
# 空间上采样 SpaUB_v2
# -----------------------
class SpaUB_v2(nn.Module):
    """
    空间上采样改进版：
    1) 使用双线性插值 + 3x3 conv 做上采样，避免转置卷积棋盘伪影
    2) 保留“低分辨率重构 + 残差修正”思想
    3) 引入简单的空间注意力
    """
    def __init__(self, hsi_channels, scale=4):
        super(SpaUB_v2, self).__init__()
        self.scale = scale

        # 主干：上采样到高分辨率
        self.up_main = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 从高分辨率回到低分辨率，用于重构 LR 并计算残差
        self.down_for_res = nn.Sequential(
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=3,
                      stride=scale, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 把残差再上采样到高分辨率作为修正项
        self.up_residual = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 空间注意力（在高分辨率特征上）
        self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, LRHS):
        # LRHS: [B, C_hsi, h, w]
        # 1) 主干上采样到高分辨率
        X_up = self.up_main(LRHS)          # [B, C, H, W], H = h*scale

        # 2) 从高分辨率重构回低分辨率，构造残差
        X_low_hat = self.down_for_res(X_up)  # [B, C, h, w]
        # 尺度对得上的前提：scale 和 conv 参数整除，这里与 stride=scale 对应
        mid = LRHS - X_low_hat

        # 3) 残差上采样到高分辨率，作为精细修正
        X_res = self.up_residual(mid)

        out = X_up + X_res
        out = self.spatial_att(out)       # 对高分辨率的纹理结构做注意力加权
        return out



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)

class Fusion_Frequency(nn.Module):
    def __init__(self, channels, cutoff_ratio=0.18, mask_sharpness=12.0):
        super(Fusion_Frequency, self).__init__()
        self.cutoff_ratio = cutoff_ratio
        self.mask_sharpness = mask_sharpness

        # Detail injector
        self.injector = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )

        # Lightweight feature alignment
        self.align = nn.Conv2d(channels * 2, channels, 1)

    def _build_lowpass_mask(self, h, w_half, device, dtype):
        fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1)
        fx = torch.fft.rfftfreq((w_half - 1) * 2, d=1.0, device=device).view(1, w_half)
        radius = torch.sqrt(fy.pow(2) + fx.pow(2))
        cutoff = self.cutoff_ratio * radius.max().clamp_min(1e-6)
        lowpass = torch.sigmoid((cutoff - radius) * self.mask_sharpness)
        return lowpass.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    def _frequency_decompose(self, x):
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        lowpass = self._build_lowpass_mask(
            h=x.shape[-2],
            w_half=x_freq.shape[-1],
            device=x.device,
            dtype=x.real.dtype
        )
        low_freq = torch.fft.irfft2(
            x_freq * lowpass,
            s=x.shape[-2:],
            dim=(-2, -1),
            norm='ortho'
        )
        high_freq = x - low_freq
        return low_freq, high_freq

    def forward(self, Fspa, Fspe):
        _, high_freq_spa = self._frequency_decompose(Fspa)
        texture_residue = self.injector(high_freq_spa)

        out = Fspe + texture_residue

        concat = torch.cat([out, Fspa], dim=1)
        out = out + 0.1 * self.align(concat)

        return out


class SSDCNet(nn.Module):
    def __init__(self, args, niter=2):
        super(SSDCNet, self).__init__()
        self.niter = niter
        self.hsi_channels = args.n_bands
        self.msi_channels = args.n_select_bands
        self.scale_ratio = args.scale_ratio

        self.base_filter = args.n_bands
        self.stride = 1
        self.patch_size = 1
        self.embed_dim = self.base_filter * self.stride * self.patch_size

        self.SpeUB = SpeUB_v2(self.hsi_channels, self.msi_channels)
        self.SpaUB = SpaUB_v2(self.hsi_channels, scale=self.scale_ratio)

        self.GAWS = GAWS(
            scale_ratio=self.scale_ratio,
            n_select_bands=self.msi_channels,
            n_bands=args.n_bands,
            image_size=args.image_size,
            feat_dim=64,
            guide_dim=64,
            H=64,
            W=64,
            mlp_dim=[256, 128],
            NIR_dim=33
        )

        self.SCFM = SCFM(
            in_channels=2 * self.hsi_channels,
            out_channels=self.hsi_channels,
            n_experts=16,
            gating_hidden=256,
            expert_hidden=64
        )

        self.Fusion = Fusion_Frequency(self.hsi_channels)

    def forward(self, LRHS, HRMS):
        # LRHS: [B, C_hsi, h, w]
        # HRMS: [B, C_msi, H, W]
        LRHS_UP = self.SpaUB(LRHS)
        HRMS_UP = self.SpeUB(HRMS)
        Fspa = self.GAWS(LRHS, LRHS_UP, HRMS_UP)
        Fspe = self.SCFM(torch.cat([LRHS_UP, HRMS_UP], dim=1))
        Fspe = LRHS_UP + Fspe
        result = self.Fusion(Fspa, Fspe)

        return result
