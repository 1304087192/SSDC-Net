import torch
import numpy as np

import cv2

from PIL import Image
import scipy

# 确保能导入 scipy.misc（新版本里只有部分函数，但模块还在）
import scipy.misc

# ====== 补丁 1：给 numpy 补 np.int，修复 "no attribute 'int'" ======
if not hasattr(np, "int"):
    np.int = int   # 或者：setattr(np, "int", int)
    
def _imresize(img, size, interp='bicubic', mode='F'):
    """
    替代 scipy.misc.imresize 的简易实现，供 skvideo.measure.niqe 调用。
    img: 2D numpy 数组 (H, W)，float 或 uint8
    size: float 比例（例如 0.5）或 (new_h, new_w)
    """
    # 1) 计算新尺寸
    if isinstance(size, (float, int)):
        new_h = int(img.shape[0] * size)
        new_w = int(img.shape[1] * size)
    else:
        # size 为 (new_h, new_w)
        new_h, new_w = size

    # 2) 归一化到 [0, 255] 做成灰度图
    img = np.asarray(img)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)

    img_uint8 = (img_norm * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(img_uint8, mode='L')

    # 3) 插值方式映射到 Pillow
    if interp == 'bicubic':
        resample = Image.BICUBIC
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    else:
        resample = Image.NEAREST

    pil_resized = pil_img.resize((new_w, new_h), resample=resample)

    # 返回 float32，和 skvideo 原实现期望的类型靠近
    return np.array(pil_resized, dtype=np.float32)


# 把自定义函数挂到 scipy.misc.imresize 上，让 skvideo 使用它
scipy.misc.imresize = _imresize

import skvideo.measure as skm     # NIQE 实现

def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    # img_max = np.max(img_tgt)
    # img_max = 1
    img_max = 255
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    """
    Calculate the Spectral Angle Mapper (SAM) between two hyperspectral images.
    
    Args:
        img_tgt: numpy.ndarray or torch.Tensor, shape (bands, H, W) or (H, W, bands)
        img_fus: numpy.ndarray or torch.Tensor, same shape as img_tgt

    Returns:
        sam: float, average SAM in degrees
    """
    eps = 1e-8  # 防止除0的小量

    # 保证通道数在第一个维度 (bands, H, W)
    if isinstance(img_tgt, np.ndarray):
        if img_tgt.shape[-1] == img_fus.shape[-1]:
            img_tgt = np.moveaxis(img_tgt, -1, 0)  # (H, W, bands) -> (bands, H, W)
            img_fus = np.moveaxis(img_fus, -1, 0)
        img_tgt = np.squeeze(img_tgt)
        img_fus = np.squeeze(img_fus)
        
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
        img_fus = img_fus.reshape(img_fus.shape[0], -1)
        
        A = np.linalg.norm(img_tgt, axis=0) + eps
        B = np.linalg.norm(img_fus, axis=0) + eps
        AB = np.sum(img_tgt * img_fus, axis=0)
        
        sam = AB / (A * B)
        sam = np.clip(sam, -1.0, 1.0)  # 防止超界
        sam = np.arccos(sam)
        sam = np.mean(sam) * 180 / np.pi

    elif isinstance(img_tgt, torch.Tensor):
        if img_tgt.shape[-1] == img_fus.shape[-1]:
            img_tgt = img_tgt.permute(2, 0, 1)  # (H, W, bands) -> (bands, H, W)
            img_fus = img_fus.permute(2, 0, 1)
        img_tgt = img_tgt.squeeze()
        img_fus = img_fus.squeeze()
        
        img_tgt = img_tgt.view(img_tgt.shape[0], -1)
        img_fus = img_fus.view(img_fus.shape[0], -1)
        
        A = torch.norm(img_tgt, dim=0) + eps
        B = torch.norm(img_fus, dim=0) + eps
        AB = torch.sum(img_tgt * img_fus, dim=0)
        
        sam = AB / (A * B)
        sam = torch.clamp(sam, -1.0, 1.0)
        sam = torch.acos(sam)
        sam = torch.mean(sam) * 180 / np.pi

        sam = sam.item()  # 转成 float

    else:
        raise TypeError("Input must be numpy.ndarray or torch.Tensor")

    return sam

def compute_mae(img1, img2):
    """
    Compute Mean Absolute Error (MAE) per pixel between two hyperspectral images img1 and img2.
    img1, img2: numpy arrays of shape (H, W, C) with the same shape.
    Returns a 2D array (H, W) of MAE values for each pixel.
    """
    # Ensure the inputs are float numpy arrays
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    # Mean across the spectral channel dimension (axis=2)
    mae_map = diff.mean(axis=2)
    return mae_map

def normalize(image):
    """
    Normalize a numpy image (grayscale or multi-channel) to [0, 1] range.
    """
    img = image.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val < 1e-8:
        # Avoid division by zero (e.g., if image is constant)
        return np.zeros_like(img)
    norm_img = (img - min_val) / (max_val - min_val)
    return norm_img

def get_rgb(hsi_image, band_indices):
    """
    Extract a pseudo-RGB image from a hyperspectral image using the specified band indices.
    - hsi_image: numpy array of shape (H, W, C) (C >= 3).
    - band_indices: list or tuple of length 3 indicating [R_band_index, G_band_index, B_band_index].
    Returns a 3-channel image (H, W, 3) normalized to [0,1].
    """
    H, W, C = hsi_image.shape
    # Clip band indices to valid range
    band_indices = [min(max(0, b), C-1) for b in band_indices]
    # Extract the three bands
    rgb = hsi_image[:, :, band_indices].astype(np.float32)  # shape (H, W, 3)
    # Normalize the 3-channel image to [0,1]
    rgb_norm = normalize(rgb)
    return rgb_norm

# --------------------------------------------------------------------

def to_gray_from_array(img: np.ndarray) -> np.ndarray:
    """
    接受 (H,W) 或 (H,W,C) 的 numpy，返回灰度图 float32 [0,1].
    - 若是 3 通道：按 RGB 转灰度
    - 若是其他通道数（例如高光谱）：取所有通道的平均作为灰度
    """
    if img.ndim == 2:
        gray = img.astype(np.float32)
    elif img.ndim == 3:
        h, w, c = img.shape
        img_f = img.astype(np.float32)

        if c == 3:
            # 假定是 RGB
            gray = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
        else:
            # 高光谱：简单做一个所有波段的平均 (你也可以改成选某几个波段)
            gray = img_f.mean(axis=2)
    else:
        raise ValueError(f"img ndim should be 2 or 3, got {img.ndim}")

    # 归一化到 [0,1]
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray.astype(np.float32)


def local_std(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """计算局部标准差，用于判定活动区域。"""
    mean = cv2.blur(img, (ksize, ksize))
    mean_sq = cv2.blur(img ** 2, (ksize, ksize))
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var)


def block_view(img: np.ndarray, block_size: int = 16):
    """把图像切成不重叠小块的视图。"""
    h, w = img.shape
    h_crop = h - h % block_size
    w_crop = w - w % block_size
    img = img[:h_crop, :w_crop]
    new_shape = (h_crop // block_size, w_crop // block_size, block_size, block_size)
    return img.reshape(new_shape)


def piqe_simplified_from_array(
    img: np.ndarray,
    block_size: int = 16,
    activity_thresh: float = 0.01,
) -> float:
    """
    简化版 PIQE：输入为 numpy 数组 (H,W) / (H,W,C)，返回分数 [0,100]，越小质量越好。
    """
    gray = to_gray_from_array(img)   # -> (H,W) float32 [0,1]
    std_map = local_std(gray, ksize=7)

    img_blocks = block_view(gray, block_size)
    std_blocks = block_view(std_map, block_size)

    Bh, Bw, _, _ = img_blocks.shape
    block_scores = []
    block_weights = []

    for i in range(Bh):
        for j in range(Bw):
            std_block = std_blocks[i, j]
            mean_std = std_block.mean()

            if mean_std < activity_thresh:  # 平坦块不参与
                continue

            block = img_blocks[i, j]
            gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            thr = np.percentile(grad_mag, 90)
            artifact_ratio = (grad_mag > thr).mean()

            score_block = artifact_ratio * 100.0
            weight_block = mean_std + 1e-6

            block_scores.append(score_block)
            block_weights.append(weight_block)

    if not block_scores:
        return 10.0

    block_scores = np.array(block_scores)
    block_weights = np.array(block_weights)

    score = np.sum(block_scores * block_weights) / np.sum(block_weights)
    score = float(np.clip(score, 0.0, 100.0))
    return score


# 若你已经装好了 scikit-video，可用这个数组版 NIQE：
def niqe_from_array(img: np.ndarray) -> float:
    gray = to_gray_from_array(img)      # (H,W), float32
    niqe_arr = skm.niqe(gray)          # 有时会返回标量或 (1,)
    return float(np.mean(niqe_arr))