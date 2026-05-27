import logging
# from tensorboard_logger import configure, log_value
import torch
import numpy as np

from utils.loss import *
from utils.metrics import calc_ergas, calc_psnr, calc_rmse, calc_sam
import matplotlib.pyplot as plt 
from utils.spectra_metric import spectra_metric

epoch_val_loss = []

def validate(validate_data_loader, arch, model, device, optimizer, epoch):
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(validate_data_loader, 1):
            GT, LRHSI, HRMSI = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output_HRHSI = model(LRHSI, HRMSI)
            val_loss = nn.L1Loss()(output_HRHSI, GT)
            epoch_val_loss.append(val_loss.item())
        ref = GT.detach().cpu().numpy()
        out = output_HRHSI.detach().cpu().numpy()
        ref = ref.reshape(-1, ref.shape[-1])  # 展平为 (N, B)
        out = out.reshape(-1, out.shape[-1])  # 展平为 (N, B)
        metric = spectra_metric(ref, out, scale=4)
        psnr = metric.PSNR()
        rmse = metric.RMSE()
        ergas = metric.ERGAS()
        sam = metric.SAM()
        ssim = metric.SSIM()
        recent_psnr=psnr
        print('RMSE:   {:.4f};'.format(rmse))
        print('PSNR:   {:.4f};'.format(psnr))
        print('ERGAS:   {:.4f};'.format(ergas))
        print('SAM:   {:.4f}.'.format(sam))
        print('SSIM:   {:.4f}.'.format(ssim))
        v_loss = np.nanmean(np.array(epoch_val_loss))
        logging.info(f'Validation Loss: {v_loss:.7f}')
        logging.info(f'Validation PSNR: {psnr:.2f}')
        print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        print('             validate loss: {:.7f}'.format(v_loss))
    return psnr