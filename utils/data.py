import torch
import torch.utils.data as data
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        self.file_path = file_path
        self.dataset = None  # 用于 worker 初始化后再打开文件

        # 仅为了获取数据长度、形状
        with h5py.File(self.file_path, 'r') as f:
            # 获取所有键名
            keys = list(f.keys())
            print(f"发现的键: {keys}")
            # self.length = f["HRHSI"].shape[0]
            # print("HRHSI shape:", f["HRHSI"].shape)
            # print("LRHSI shape:", f["LRHSI"].shape)
            # print("HRMSI shape:", f["HRMSI"].shape)

            # self.length = f["GT"].shape[0]
            # print("GT shape:", f["GT"].shape)
            # print("LRHSI shape:", f["LRHSI"].shape)
            # print("RGB shape:", f["RGB"].shape)

            # print("MS shape:", f["MS"].shape)
            # print("PAN shape:", f["PAN"].shape)

            # self.length = f["gt"].shape[0]
            # print("GT shape:", f["gt"].shape)
            # print("LRHSI shape:", f["ms"].shape)
            # print("RGB shape:", f["pan"].shape)

            self.length = f["GT"].shape[0]
            print("GT shape:", f["GT"].shape)
            print("LRHSI shape:", f["LRHSI"].shape)
            print("HRMSI shape:", f["HRMSI"].shape)


    def __len__(self):
        return self.length

    def _init_dataset(self):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')  # 每个 worker 各自打开一次，保持连接

    def __getitem__(self, index):
        self._init_dataset()
        # CAVE,PaviaC
        # GT = torch.from_numpy(self.dataset["GT"][index]).float()
        # LRHSI = torch.from_numpy(self.dataset["LRHSI"][index]).float()
        # RGB = torch.from_numpy(self.dataset["RGB"][index]).float()

        # Botswana
        # MS = torch.from_numpy(self.dataset["MS"][index]).float()
        # PAN = torch.from_numpy(self.dataset["PAN"][index]).float()

        # WV3
        # GT = torch.from_numpy(self.dataset["gt"][index]).float()
        # LRHSI = torch.from_numpy(self.dataset["ms"][index]).float()
        # RGB = torch.from_numpy(self.dataset["pan"][index]).float()

        # DC
        # GT = torch.from_numpy(self.dataset["GT"][index]).float().permute(2, 0, 1)
        # LRHSI = torch.from_numpy(self.dataset["LRHSI"][index]).float().permute(2, 0, 1)
        # RGB = torch.from_numpy(self.dataset["RGB"][index]).float().permute(2, 0, 1)

        # Houston
        # HRHSI = torch.from_numpy(self.dataset["HRHSI"][index]).float().permute(2, 0, 1)
        # LRHSI = torch.from_numpy(self.dataset["LRHSI"][index]).float().permute(2, 0, 1)
        # HRMSI = torch.from_numpy(self.dataset["HRMSI"][index]).float().permute(2, 0, 1)

        #LN01,LN02
        GT = torch.from_numpy(self.dataset["GT"][index]).float()
        LRHSI = torch.from_numpy(self.dataset["LRHSI"][index]).float()
        HRMSI = torch.from_numpy(self.dataset["HRMSI"][index]).float()

        # return GT, LRHSI, RGB
        # return HRHSI, LRHSI, HRMSI
        # return GT, MS, PAN
        return GT, LRHSI, HRMSI
        
class DatasetFromHdf5Real(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5Real, self).__init__()
        self.file_path = file_path
        self.dataset = None  # 用于 worker 初始化后再打开文件
        with h5py.File(self.file_path, 'r') as f:
            # 获取所有键名
            keys = list(f.keys())
            print(f"发现的键: {keys}")
            # 确认 shape
            self.length = f["ms"].shape[0]
            print("ms shape:", f["ms"].shape)
            print("pan shape:", f["pan"].shape)

    def __len__(self):
        return self.length

    def _init_dataset(self):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')  # 每个 worker 各自打开一次，保持连接

    def __getitem__(self, index):
        self._init_dataset()
        ms = self.dataset["ms"][index]
        pan = self.dataset["pan"][index]
        return ms, pan

class DatasetFromPt(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromPt, self).__init__()
        self.file_path = file_path
        self.data = torch.load(self.file_path)

        # 确认 shape
        self.length = self.data["GT"].shape[0]
        print("GT shape:", self.data["GT"].shape)
        print("LRHSI shape:", self.data["LRHSI"].shape)
        print("RGB shape:", self.data["RGB"].shape)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        GT = self.data["GT"][index]     # already in shape (C, H, W)
        LRHSI = self.data["LRHSI"][index]
        RGB = self.data["RGB"][index]
        return GT, LRHSI, RGB
