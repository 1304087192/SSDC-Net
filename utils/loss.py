import torch
import torch.nn as nn

l1 = nn.L1Loss()
def ADUFnet_loss_l1_kkt(output, GT, Z1, Z2, A, A_prev, rho1, rho2, beta=0.1, gamma=0.1):
    # 1) 重建监督（L1）
    loss_recon = l1(output, GT)

    # 2) 原始残差（primal）
    loss_primal = 0.5 * (l1(Z1, A) + l1(Z2, A))

    # 3) 对偶残差（dual）
    if A_prev is not None:
        loss_dual_1 = l1(rho1 * (A - A_prev), torch.zeros_like(A))
        loss_dual_2 = l1(rho2 * (A - A_prev), torch.zeros_like(A))
        loss_dual = 0.5 * (loss_dual_1 + loss_dual_2)
    else:
        loss_dual = torch.zeros((), device=A.device)

    total = loss_recon + beta * loss_primal + gamma * loss_dual
    return total, loss_recon, beta * loss_primal, gamma * loss_dual,{'recon': loss_recon.item(),
                   'primal': (loss_primal).item(),
                   'dual': (loss_dual).item()}

def SSIMLoss(output, target):
    return 1 - ssim(output, target, data_range=1.0, size_average=True)

class SAMLoss(nn.Module):
    def forward(self, output, target):
        EPS = 1e-8
        dot_product = torch.sum(output * target, dim=1)
        norm1 = torch.norm(output, dim=1)
        norm2 = torch.norm(target, dim=1)
        sam = torch.acos(torch.clamp(dot_product / (norm1 * norm2 + EPS), -1, 1))
        return torch.mean(sam)

def compute_spatial_spectral_loss(spat_edge1, spat_edge2, spec_edge, GT):
    """
    输入:
        spat_edge1: (B, C, H-1, W) - 纵向边缘预测图
        spat_edge2: (B, C, H, W-1) - 横向边缘预测图
        spec_edge : (B, C-1, H, W) - 光谱边缘预测图
        GT        : (B, C, H, W)   - Ground Truth 高光谱图像
    返回:
        L_spat1, L_spat2, L_spec, L_total
    """
    B, C, H, W = GT.shape

    # 构造参考边缘图像
    spat1_hat = GT[:, :, :-1, :] - GT[:, :, 1:, :]       # (B, C, H-1, W)
    spat2_hat = GT[:, :, :, :-1] - GT[:, :, :, 1:]       # (B, C, H, W-1)
    spec_hat  = GT[:, :-1, :, :] - GT[:, 1:, :, :]       # (B, C-1, H, W)

    # 计算每个损失项
    L_spat1 = torch.sum((spat_edge1 - spat1_hat) ** 2) / (2 * B * C * (H - 1) * W)
    L_spat2 = torch.sum((spat_edge2 - spat2_hat) ** 2) / (2 * B * C * H * (W - 1))
    L_spec  = torch.sum((spec_edge - spec_hat) ** 2)    / (2 * B * (C - 1) * H * W)

    # 合并损失
    L_spat = 0.5 * L_spat1 + 0.5 * L_spat2
    L_total = L_spat + L_spec

    return L_total

def L1Loss(output, target):
    PLoss = nn.L1Loss(size_average=True).cuda()
    Pixelwise_Loss =PLoss(output, target)
    return Pixelwise_Loss

class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (x[:, :, 1:, :].numel())
        count_w = (x[:, :, :, 1:].numel())
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w)