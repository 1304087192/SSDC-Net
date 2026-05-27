import os
import h5py
import torch
from torch.utils.data import Dataset

class HSIData(Dataset):
    """
    Dataset for hyperspectral image fusion from HDF5 files.
    It handles different dataset formats (WashingtonDC, PaviaC, Chikusei, Houston) by checking dataset name.
    Returns high-res HSI (GT), low-res HSI, and high-res MSI (or RGB) as torch Tensors.
    """
    def __init__(self, file_path, dataset_name):
        super(HSIData, self).__init__()
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.h5 = None    # will hold the h5py File object (opened on first access)

        # Determine data orientation and dimensions by inspecting the file
        with h5py.File(self.file_path, 'r') as f:
            # Identify which keys exist for ground truth and multi-spectral data
            if "GT" in f:
                gt_shape = f["GT"].shape
            else:
                gt_shape = f["HRHSI"].shape  # Houston uses "HRHSI" instead of "GT"

            # -------- 1. 调整各数据集的存储维度方向 ---------------------------------
            # PaviaC / Chikusei: (N, C, H, W)
            # Houston:           (N, H, W, C)
            # 新 WashingtonDC:    (N, C, H, W)
            if dataset_name in ["PaviaC", "Chikusei", "WashingtonDC", "Botswana"]:
                self.orientation = "NCHW"   # Channels before spatial
            elif dataset_name in ["Houston"]:
                self.orientation = "NHWC"   # Channels last
            else:
                # Fallback: assume channels last if last dim is likely spectral bands
                self.orientation = "NHWC" if gt_shape[-1] > 3 else "NCHW"

            # Get dataset length (number of samples)
            if "GT" in f:
                self.length = gt_shape[0]
            else:
                self.length = f["HRHSI"].shape[0]

            # -------- 2. 统计高光谱与多光谱通道数 -----------------------------------
            if self.orientation == "NCHW":
                num_hyper = gt_shape[1]     # number of HSI bands

                # Multi-spectral (MS) bands: check key "HRMSI" or "RGB" or "PAN"
                if "HRMSI" in f:
                    ms_shape = f["HRMSI"].shape
                elif "PAN" in f:  # 新 WashingtonDC 使用 PAN 作为高分辨率多光谱/全色
                    ms_shape = f["PAN"].shape
                else:
                    ms_shape = f["RGB"].shape
                num_ms = ms_shape[1]
            else:  # NHWC
                if "GT" in f:
                    num_hyper = gt_shape[-1]
                else:
                    num_hyper = f["HRHSI"].shape[-1]
                # Multi-spectral channels:
                if "HRMSI" in f:
                    ms_shape = f["HRMSI"].shape
                elif "PAN" in f:
                    ms_shape = f["PAN"].shape
                else:
                    ms_shape = f["RGB"].shape
                num_ms = ms_shape[-1]

            self.num_hyper = num_hyper      # number of spectral bands in HSI (GT)
            self.num_ms = num_ms            # number of bands in multi-spectral (MSI/RGB) input

            # -------- 3. 高分辨率图像空间尺寸 --------------------------------------
            if self.orientation == "NCHW":
                self.hr_h = gt_shape[2]
                self.hr_w = gt_shape[3]
            else:  # NHWC
                if "GT" in f:
                    self.hr_h = gt_shape[1]
                    self.hr_w = gt_shape[2]
                else:
                    self.hr_h = f["HRHSI"].shape[1]
                    self.hr_w = f["HRHSI"].shape[2]

            # -------- 4. 低分辨率 HSI 尺寸（用于 scale 计算）------------------------
            # 旧数据集里是 LRHSI，新 WashingtonDC 里变为 MS
            if "LRHSI" in f:
                lr_shape = f["LRHSI"].shape
            elif "MS" in f:  # <- 新增：WashingtonDC 用 MS 作为低分辨率 HSI
                lr_shape = f["MS"].shape
            else:
                # If LRHSI key not present (unlikely), assume scale factor known
                self.lr_h = self.hr_h // 4
                self.lr_w = self.hr_w // 4
                lr_shape = None

            if lr_shape is not None:
                if self.orientation == "NCHW":
                    self.lr_h = lr_shape[2]
                    self.lr_w = lr_shape[3]
                else:
                    self.lr_h = lr_shape[1]
                    self.lr_w = lr_shape[2]

            # Compute scale factor (assuming integer scale)
            if self.lr_h > 0:
                self.scale = self.hr_h // self.lr_h
            else:
                self.scale = 1  # default

    def __len__(self):
        return self.length

    def _init_h5(self):
        # Open the HDF5 file on first access (to avoid reopening for each sample, especially when using multiprocessing)
        if self.h5 is None:
            self.h5 = h5py.File(self.file_path, 'r')

    def __getitem__(self, idx):
        self._init_h5()

        # Load data for index
        # Depending on the dataset, use the appropriate keys and orientation
        if self.dataset_name == "Houston":
            # Houston uses keys HRHSI (GT), LRHSI, HRMSI, 形状为 (H, W, C)
            hr_hsi = self.h5["HRHSI"][idx]  # (H, W, C)
            lr_hsi = self.h5["LRHSI"][idx]  # (h, w, C)
            hr_msi = self.h5["HRMSI"][idx]  # (H, W, C_ms)

            GT = torch.from_numpy(hr_hsi.astype('float32')).permute(2, 0, 1)
            LRHSI = torch.from_numpy(lr_hsi.astype('float32')).permute(2, 0, 1)
            HRMSI = torch.from_numpy(hr_msi.astype('float32')).permute(2, 0, 1)

        elif self.dataset_name == "WashingtonDC":
            # -------- 5. 新 WashingtonDC 读取方式 ------------------------------
            # 现在文件里：
            # GT: (B, C, H, W)
            # MS: (B, C, h, w)  -> 低分辨率 HSI
            # PAN: (B, C_ms, H, W) -> 高分辨率多光谱/全色
            gt = self.h5["GT"][idx]     # (C, H, W)
            lr = self.h5["MS"][idx]     # (C, h, w)
            pan = self.h5["PAN"][idx]   # (C_ms, H, W)

            GT = torch.from_numpy(gt.astype('float32'))      # 已经是 (C, H, W)，无需 permute
            LRHSI = torch.from_numpy(lr.astype('float32'))   # (C, h, w)
            HRMSI = torch.from_numpy(pan.astype('float32'))  # (C_ms, H, W)

        elif self.dataset_name in ["PaviaC", "Chikusei"]:
            # PaviaC and Chikusei use keys GT, LRHSI, RGB, stored as (N, C, H, W)
            # print("dataset name:",self.dataset_name)
            GT = torch.from_numpy(self.h5["GT"][idx].astype('float32'))       # (C, H, W)
            LRHSI = torch.from_numpy(self.h5["LRHSI"][idx].astype('float32')) # (C, h, w)
            HRMSI = torch.from_numpy(self.h5["RGB"][idx].astype('float32'))   # (C_ms, H, W)

        else:
            # Fallback: try keys "GT" and "RGB" or "HRMSI" with orientation check
            if "GT" in self.h5:
                data_gt = self.h5["GT"][idx]
            else:
                data_gt = self.h5["HRHSI"][idx]
            # 低分辨率：优先 LRHSI，其次 MS
            if "LRHSI" in self.h5:
                data_lr = self.h5["LRHSI"][idx]
            elif "MS" in self.h5:
                data_lr = self.h5["MS"][idx]
            else:
                data_lr = None

            if "HRMSI" in self.h5:
                data_ms = self.h5["HRMSI"][idx]
            elif "PAN" in self.h5:
                data_ms = self.h5["PAN"][idx]
            else:
                data_ms = self.h5["RGB"][idx]

            if self.orientation == "NHWC":
                GT = torch.from_numpy(data_gt.astype('float32')).permute(2, 0, 1)
                if data_lr is not None:
                    LRHSI = torch.from_numpy(data_lr.astype('float32')).permute(2, 0, 1)
                else:
                    LRHSI = None
                HRMSI = torch.from_numpy(data_ms.astype('float32')).permute(2, 0, 1)
            else:
                GT = torch.from_numpy(data_gt.astype('float32'))
                if data_lr is not None:
                    LRHSI = torch.from_numpy(data_lr.astype('float32'))
                else:
                    LRHSI = None
                HRMSI = torch.from_numpy(data_ms.astype('float32'))

        return GT, LRHSI, HRMSI
