import time
import logging
# from tensorboard_logger import configure, log_value
import numpy as np

from utils.loss import *
from utils.metrics import calc_ergas, calc_psnr, calc_rmse, calc_sam 
from utils.spectra_metric import spectra_metric
import torch

@torch.no_grad()
def compute_P_from_LRHS(Y: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Y: [B, C, h, w]  (NCHW)
    return P: [B, C, rank]
    """
    B, C, h, w = Y.shape
    # 展平成 [B, N, C]，N=h*w
    X = Y.permute(0, 2, 3, 1).reshape(B, h*w, C)  # [B, N, C]

    # 中心化（建议做，PCA/SVD更稳定）
    X = X - X.mean(dim=1, keepdim=True)

    # SVD: X = U S Vh，Vh: [B, C, C]
    # 取前 rank 个右奇异向量作为谱基
    # torch.linalg.svd 返回 U,S,Vh
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    P = Vh.transpose(-2, -1)[:, :, :rank]  # [B, C, rank]
    return P

def train(training_data_loader,
        model,
        model_path,
        optimizer,
        scheduler,
        epoch,
        epochs,
        device,
        arch,
        dataset

 ):   
    time_s = time.time()
    epoch += 1
    epoch_train_loss = []
    epoch_psnr = []
    alpha = 1e-8
    # ============Epoch Train=============== #
    model.train()
    best_psnr = 0
    print("MODEL:", model_path)
    logging.info("MODEL:" + model_path)
    for iteration, batch in enumerate(training_data_loader, 1):
        GT, LRHSI, HRMSI = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # time_loadData = time.time()
        # print("time_loadData: ",time_loadData-time_s)
        optimizer.zero_grad()  # fixed

        if arch == "ADMMHFNet":
            P = compute_P_from_LRHS(LRHSI, rank=model.rank)
            output_HRHSI = model(LRHSI, HRMSI, P)
        elif arch == 'SMGUNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'ARG1Q2S':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'fusformer':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'QIS_GAN':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'AMSFNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'DCTransformer':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'MDCFusformer':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'PSRT':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'MCTNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'MSSTNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'MCTNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'SFIGNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'PSTUN':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'AELF':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'MoE_INR':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'SDAGE':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'MCIFNet':
            output_HRHSI = model(LRHSI,HRMSI)
        elif arch == 'SSDCNet':
            output_HRHSI = model(LRHSI,HRMSI)
     
        loss = L1Loss(output_HRHSI, GT)
        loss_library = model.SpectralIGM.get_library_reg_loss()

        smooth_loss = model.SpectralIGM.get_last_smooth_loss()
        loss = loss + loss_library + smooth_loss

        optimizer.zero_grad()
        loss.backward()
        # ✅ 在这里把所有 None 的 grad 补成 0，避免 muon_update 里收到 None
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None and p.requires_grad:
                    p.grad = torch.zeros_like(p)

        optimizer.step()
        epoch_train_loss.append(loss.item())

        ref = GT.detach().cpu().numpy()
        out = output_HRHSI.detach().cpu().numpy()
        # psnr = calc_psnr(ref, out)
        ref = ref.reshape(-1, ref.shape[-1])  # 展平为 (N, B)
        out = out.reshape(-1, out.shape[-1])  # 展平为 (N, B)
        metric = spectra_metric(ref, out, scale=4)
        psnr = metric.PSNR()
        epoch_psnr.append(psnr)
        if iteration % 100 == 0:
            # log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
            # print("===> Epoch[{}]({}/{}): Loss: {:.6f} | Loss_rec: {:.6f} | Loss_primal: {:.6f} | Loss_dual: {:.6f} | PSNR: {:.2f}".format(epoch, iteration, len(training_data_loader),loss.item(),loss_recon.item(),loss_primal.item(),loss_dual.item(),psnr))
            print("===> Epoch[{}]({}/{}): Loss: {:.6f} | PSNR: {:.2f}".format(epoch, iteration, len(training_data_loader),loss.item(), psnr))
            logging.info(f"Epoch[{epoch}]({iteration}/{len(training_data_loader)}): Loss: {loss.item():.6f} | PSNR: {psnr:.2f}")
    print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
    scheduler.step()  # update lr

    t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
    t_psnr = np.nanmean(np.array(epoch_psnr))
    print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
    print("PSNR: {:.2f}".format(t_psnr))
    logging.info(f'Epoch: {epoch}/{epochs} Training Loss: {t_loss:.7f}')
    # log_value('train_loss', t_loss, epoch)
    # log_value('lr', optimizer.param_groups[0]['lr'], epoch)
    # log_value('psnr', t_psnr, epoch)
    time_e = time.time()
    # print("train_left: ",time_e-time_train)    
    print("train_totaltime: ",time_e-time_s)