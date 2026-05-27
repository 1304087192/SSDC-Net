import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
# from muon import SingleDeviceMuonWithAuxAdam
import warnings
# 屏蔽所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 屏蔽特定的 meshgrid 警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*"
)

# from models.DCTransformer import DCTransformer
# from models.PSTUN import PSTUN
# from models.AMSFNet import AMSFNet
from models.QIS_GAN import QIS_GAN
from models.MSSTNet import MSSTNet
from models.MCTNet.MCT_LN02 import MCTNet
from models.MDCFusformer_LN import MDCFusformer
# from models.PSRT import PSRT
# from models.SCIAUNet import V19Net
# from models.ADMMHFNet import ADMMHFNet
from models.AELF.AELF import AELF
# from models.MCIFNet import MCIFNet
from models.SMGUNet import SMGUNet
from models.SSDCNet import SSDCNet

from utils.data import *

from utils import *
from valid import validate
from train import *
from utils.scheduler import scheduler_is
import args_parser

from torch.nn import functional as F
# from tensorboard_logger import configure, log_value
# import pandas as pd

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log_dir = "TensorboardLog/new/" + args.arch + '_' +args.dataset
# configure(log_dir)

def main(training_data_loader, validate_data_loader, test_data_loader):
    print (args)
    if args.dataset == 'PaviaC':
      args.n_bands = 92
      args.image_size = 64
    elif args.dataset == 'Botswana':
      args.n_bands = 145
      args.n_select_bands = 1   
    elif args.dataset == 'WashingtonDC':
      args.n_bands = 191  
      # args.image_size = 64
    elif args.dataset == 'Houston':
      args.n_bands = 50
      args.n_select_bands = 13
    elif args.dataset == 'CAVE':
      args.n_bands = 31
      args.n_select_bands = 3
    elif args.dataset == 'XiongAn':
      args.n_bands = 256
      args.n_select_bands = 4
    elif args.dataset == 'Chikusei':
      args.n_bands = 128
      args.n_select_bands = 3
    elif args.dataset == 'Botswana':
      args.n_bands = 145
      args.n_select_bands = 1
    elif args.dataset == 'WV3':
      args.n_bands = 8
      args.n_select_bands = 1
    elif (args.dataset == 'LN02') | (args.dataset == 'LN01'):
      args.n_bands = 144
      args.n_select_bands = 8
      args.image_size = 60
      args.scale_ratio = 3

    # Build the models
    if args.arch == 'SCIAUNet':
        model = V19Net(args).cuda()
    elif args.arch == 'SMGUNet':
        model = SMGUNet(args).cuda()
    elif args.arch == 'fusformer':
        model = FusformerNet(args).cuda()
    elif args.arch == 'QIS_GAN':
        model = QIS_GAN(args).cuda()
    elif args.arch == 'MSSTNet':
        model = MSSTNet(args).cuda()
    elif args.arch == 'MCTNet':
        model = MCTNet(args).cuda()
    elif args.arch == 'AMSFNet':
        model = AMSFNet(args).cuda()
    elif args.arch == 'DCTransformer':
        model = DCTransformer(args).cuda()
    elif args.arch == 'MDCFusformer':
        model = MDCFusformer(args).cuda()
    elif args.arch == 'PSRT':
        model = PSRT(args).cuda()
    elif args.arch == 'MCTNet':
        model = MCTNet(args).cuda()
    elif args.arch == 'SFIGNet':
        model = SFIGNet(args).cuda()
    elif args.arch == 'PSTUN':
        model = PSTUN(args).cuda()
    elif args.arch == 'AELF':
        model = AELF(args).cuda()
    elif args.arch == 'SDAGE':
        model = SDAGE(args).cuda()
    elif args.arch == 'ADMMHFNet':
        model = ADMMHFNet(args,rank=8,stages=11,ratio=4.0).cuda()
    elif args.arch == 'MCIFNet':
        model = MCIFNet(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'SSDCNet':
        model = SSDCNet(args).cuda()

    # 设置 GPU 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载模型到 GPU1
    model.to(device)

    # Loss and optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    elif args.optimizer == 'Muon':
        # 1. 高维权重（比如 Linear/Conv 的权重矩阵）— 使用 Muon
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]

        # 2. 低维参数（例如 bias、LayerNorm 权重），使用辅助 AdamW
        hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
        # 3. 按官方教程要求创建参数组
        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=0.02,
                weight_decay=0.01
            ),
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=args.lr,               # 建议保持与 AdamW 相同
                betas=(0.9, 0.95),
                weight_decay=0.01
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    # # 手动为每个参数组设置 initial_lr
    # for param_group in optimizer.param_groups:
    #     param_group.setdefault('initial_lr', 0.000005)
    
    logging.basicConfig(
        filename='Logs/train/' + args.arch + '_' + args.dataset + '.log',  # 修改日志存储路径
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    start_epoch = 0
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(validate_data_loader, 
                            args.arch,
                            model,
                            device,
                            optimizer,
                            0)
        print ('psnr: ', recent_psnr)
        # print(checkpoint.keys())
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = 4

    # scheduler
    scheduler = scheduler_is(args.scheduler, optimizer, start_epoch)


    best_psnr = 0
    best_psnr = validate(validate_data_loader,
                    args.arch, 
                    model,
                    device,
                    optimizer,
                    0)
    print ('psnr: ', best_psnr)

    # ==========================
    # Early Stopping 初始化
    # ==========================
    patience = 400
    no_improve = 0
    best_epoch = 0

    # Epochs
    print ('Start Training: ')
    for epoch in range(start_epoch, args.n_epochs):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        train(training_data_loader,
            model,
            model_path,
            optimizer,
            scheduler,
            epoch,
            args.n_epochs,
            device,
            args.arch,
            args.dataset
        )

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(validate_data_loader, 
                        args.arch,
                        model,
                        device,
                        optimizer,
                        epoch)
        print ('psnr: ', recent_psnr)

        # ==========================
        # Early Stopping 判定
        # ==========================
        if recent_psnr > best_psnr:
            best_psnr = recent_psnr
            best_epoch = epoch
            no_improve = 0
            print(f"Epoch {epoch}: validation improved → {best_psnr:.4f}, saving checkpoint...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        else:
            no_improve += 1
            print(f"Epoch {epoch}: no improvement ({no_improve}/{patience})")

        # ==========================
        # 判断是否触发 Early Stop
        # ==========================
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}! Best epoch = {best_epoch} with PSNR={best_psnr:.4f}")
            break
  
    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    train_set = DatasetFromHdf5(args.dataset_root + '/' + args.dataset + '_Train.h5')  # creat data for training
    # train_set = DatasetFromPt(args.dataset_root + '/' + args.dataset + '_Train.pt')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = DatasetFromHdf5(args.dataset_root + '/' + args.dataset + '_Valid.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=8, batch_size=1, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put validate data to DataLoader for batches

    test_set = DatasetFromHdf5(args.dataset_root + '/' + args.dataset + '_Test.h5')  # creat data for test
    test_data_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put test data to DataLoader for batches
    main(training_data_loader, validate_data_loader, test_data_loader)
