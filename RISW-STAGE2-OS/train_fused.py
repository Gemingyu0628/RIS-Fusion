
"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""
import os
import sys
import time
import datetime
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import kornia

from net.restormer import Restormer_Encoder, Restormer_Decoder, preFuse_Decoder
from utils.loss import Fusionloss, L_color, LpLssimLossweight
from utils.logger import Logger1
from utils.dataset import H5Dataset_nofusion
import CRIS.utils.config as config
from CRIS.model import build_segmenter


"""
------------------------------------------------------------------------------
Environment setup and random seed for reproducibility
------------------------------------------------------------------------------
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]
torch.backends.cudnn.deterministic = True  # 保证结果可复现
torch.backends.cudnn.benchmark = True      # 提升运行效率
seed_value = 3407
os.environ["PYTHONHASHSEED"] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


"""
------------------------------------------------------------------------------
Training hyperparameters
------------------------------------------------------------------------------
"""
num_epochs = 101
lr = 1e-4
weight_decay = 0
batch_size = 2
print("batchsize", batch_size)
print("num_epochs", num_epochs)

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


"""
------------------------------------------------------------------------------
Data loader
------------------------------------------------------------------------------
"""
datasetname = "IVT_final"
trainloader = DataLoader(
    H5Dataset_nofusion(r"data/add_imgsize_1024x1280_stride_0_text_multi.h5"),
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=0,
)
loader = {"train": trainloader}
print(datasetname)

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

"""
------------------------------------------------------------------------------
Model initialization and checkpoint loading
------------------------------------------------------------------------------
"""
def get_parser():
    parser = argparse.ArgumentParser(description="Pytorch Referring Expression Segmentation")
    parser.add_argument(
        "--config",
        default="CRIS/config/refcoco/cris_r50.yaml",
        type=str,
        help="config file",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="override some settings in the config.",
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

args = get_parser()
segmodel, _ = build_segmenter(args)

Feature_Encoder_IR = nn.DataParallel(Restormer_Encoder()).to(device)
Feature_Encoder_VIS = nn.DataParallel(Restormer_Encoder(inp_channels=3)).to(device)
Restorm_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
Pre_Decoder = nn.DataParallel(preFuse_Decoder()).to(device)
segmodel = nn.DataParallel(segmodel).to(device)

# 加载融合模型权重
checkpoint = torch.load("TRIS_stage2_epoch_260_11-03-14-12.pth")
Feature_Encoder_IR.load_state_dict(checkpoint["Feature_Encoder_IR"], strict=True)
Feature_Encoder_VIS.load_state_dict(checkpoint["Feature_Encoder_VIS"], strict=True)
Restorm_Decoder.load_state_dict(checkpoint["Restorm_Decoder"], strict=True)
Pre_Decoder.load_state_dict(checkpoint["Pre_Decoder"], strict=True)
segmodel.load_state_dict(checkpoint["segmodel"], strict=True)

# 加载分割模型权重
checkpoint = torch.load("CRIS/exp/refcoco/CRIS_R50/best_model.pth")
segmodel.load_state_dict(checkpoint["state_dict"], strict=True)


"""
------------------------------------------------------------------------------
Optimizers and learning rate schedulers
------------------------------------------------------------------------------
"""
optimizer1 = torch.optim.Adam(Feature_Encoder_IR.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Feature_Encoder_VIS.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(Restorm_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(Pre_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(segmodel.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)


"""
------------------------------------------------------------------------------
Loss functions
------------------------------------------------------------------------------
"""
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction="mean")
smoothl1loss = nn.SmoothL1Loss()

fusion_loss = Fusionloss(coeff_grad=10, device=device)
cbcr_loss = L_color().to(device)
criterion = LpLssimLossweight().to(device)

transform = transforms.Grayscale(num_output_channels=1)
transform_resize = transforms.Resize((416, 416))


"""
------------------------------------------------------------------------------
Training loop
------------------------------------------------------------------------------
"""
step = 0
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (image_IR, image_VIS, mask, Text) in enumerate(loader["train"]):
        image_IR, image_VIS, mask, Text = (
            image_IR.cuda(),
            image_VIS.cuda(),
            mask.cuda(),
            Text.cuda(),
        )

        # 切换模型训练模式
        Feature_Encoder_IR.train()
        Feature_Encoder_VIS.train()
        Restorm_Decoder.train()
        Pre_Decoder.train()
        segmodel.train()

        # 清零梯度
        Feature_Encoder_IR.zero_grad()
        Feature_Encoder_VIS.zero_grad()
        Restorm_Decoder.zero_grad()
        Pre_Decoder.zero_grad()
        segmodel.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()

        # 提取特征
        Feature_IR = Feature_Encoder_IR(image_IR)
        Feature_VIS = Feature_Encoder_VIS(image_VIS)

        # 预融合
        pre_fuse = Pre_Decoder(Feature_IR.detach(), Feature_VIS.detach())
        fuseforseg = transform_resize(pre_fuse.detach())

        # 语义分割推断及损失
        seg, _, seg_loss = segmodel(fuseforseg.detach(), Text, mask)
        if seg.shape[-2:] != image_IR.shape[-2:]:
            seg = F.interpolate(seg, size=image_IR.shape[-2:], mode="bicubic", align_corners=True)

        # 分离加权特征
        Feature_IR_emphasized = Feature_IR * seg
        Feature_IR_unemphasized = Feature_IR * (1 - seg)
        Feature_VIS_emphasized = Feature_VIS * seg
        Feature_VIS_unemphasized = Feature_VIS * (1 - seg)

        Feature_Fuse_emphasized = torch.cat((Feature_IR_emphasized, Feature_VIS_emphasized), dim=1)
        Feature_Fuse_unemphasized = torch.cat((Feature_IR_unemphasized, Feature_VIS_unemphasized), dim=1)

        # 融合特征解码
        Fused_final = Restorm_Decoder(Feature_Fuse_emphasized, Feature_Fuse_unemphasized)

        image_VIS_gray = transform(image_VIS)
        Fused_final_gray = transform(Fused_final)
        pre_fuse_gray = transform(pre_fuse)

        # 计算预融合损失
        loss_total_prefuse, loss_in_prefuse, loss_grad_prefuse = fusion_loss(image_IR, image_VIS_gray, Fused_final_gray)
        loss_total_prefuse *= 2  # 权重放大

        IR_ssim_loss_pre_fuse = Loss_ssim(pre_fuse_gray, image_IR)
        VIS_ssim_loss_pre_fuse = Loss_ssim(pre_fuse, image_VIS)
        ssim_loss_pre_fuse = IR_ssim_loss_pre_fuse + VIS_ssim_loss_pre_fuse
        ssim_loss_pre_fuse *= 4

        IR_mse_loss_pre_fuse = MSELoss(pre_fuse_gray, image_IR)
        VIS_mse_loss_pre_fuse = MSELoss(pre_fuse, image_VIS)
        mse_loss_pre_fuse = IR_mse_loss_pre_fuse + VIS_mse_loss_pre_fuse
        mse_loss_pre_fuse *= 4

        colorloss_pre_fuse = cbcr_loss(image_VIS, pre_fuse)
        colorloss_pre_fuse *= 50

        lossALL_pre_fuse = loss_total_prefuse + ssim_loss_pre_fuse + mse_loss_pre_fuse + colorloss_pre_fuse

        # 计算最终融合损失
        loss_total, loss_in, loss_grad = fusion_loss(image_IR, image_VIS_gray, Fused_final_gray)
        loss_total *= 2

        IR_ssim_loss = Loss_ssim(Fused_final_gray, image_IR)
        VIS_ssim_loss = Loss_ssim(Fused_final, image_VIS)
        ssim_loss = IR_ssim_loss + 1.5 * VIS_ssim_loss
        ssim_loss *= 4

        IR_mse_loss = MSELoss(Fused_final_gray, image_IR)
        VIS_mse_loss = MSELoss(Fused_final, image_VIS)
        mse_loss = IR_mse_loss + 5 * VIS_mse_loss
        mse_loss *= 4

        colorloss = cbcr_loss(image_VIS, Fused_final)
        colorloss *= 50

        lossALL_final_fuse = loss_total + ssim_loss + mse_loss + colorloss

        # 关注区域损失计算
        Fused_seg_in_interest = Fused_final * mask
        Fused_seg_Y_interest = Fused_final_gray * mask
        image_IR_seg_interest = image_IR * mask
        image_vis_seg_interest = image_VIS * mask

        loss_y_l1 = L1Loss(Fused_seg_Y_interest, image_IR_seg_interest)
        loss_grad_l1 = L1Loss(
            kornia.filters.SpatialGradient()(Fused_seg_in_interest),
            kornia.filters.SpatialGradient()(image_vis_seg_interest),
        )
        loss_cbcr = cbcr_loss(Fused_seg_in_interest, image_vis_seg_interest)

        loss_interest = loss_y_l1 + 5 * loss_grad_l1 + 2 * loss_cbcr

        # 总损失反向传播
        lossALL = lossALL_final_fuse + lossALL_pre_fuse + 1500 * loss_interest + 100*seg_loss
        lossALL.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(Feature_Encoder_IR.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(Feature_Encoder_VIS.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(Restorm_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(Pre_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(segmodel.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

        # 优化器更新
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()

        # 估计剩余时间并打印训练信息
        batches_done = epoch * len(loader["train"]) + i
        batches_left = num_epochs * len(loader["train"]) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [lossALL: %.6f] [lossALL_pre_fuse: %.6f] [lossALL_final_fuse: %.6f] [loss_interest: %.6f] [seg_loss: %.6f] ETA: %.6s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader["train"]),
                lossALL.item(),
                lossALL_pre_fuse.item(),
                lossALL_final_fuse.item(),
                loss_interest.item(),
                seg_loss.item(),
                time_left,
            )
        )
        sys.stdout.flush()

    # 学习率调整
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()
    scheduler5.step()

    # 学习率下限限制，防止过低
    if optimizer1.param_groups[0]["lr"] <= 1e-6:
        optimizer1.param_groups[0]["lr"] = 1e-6
    if optimizer2.param_groups[0]["lr"] <= 1e-6:
        optimizer2.param_groups[0]["lr"] = 1e-6
    if optimizer3.param_groups[0]["lr"] <= 1e-6:
        optimizer3.param_groups[0]["lr"] = 1e-6
    if optimizer4.param_groups[0]["lr"] <= 1e-6:
        optimizer4.param_groups[0]["lr"] = 1e-6
    if optimizer5.param_groups[0]["lr"] <= 1e-6:
        optimizer5.param_groups[0]["lr"] = 1e-6

    # 定期保存模型
    if epoch % 20 == 0:
        checkpoint = {
            "Feature_Encoder_IR": Feature_Encoder_IR.state_dict(),
            "Feature_Encoder_VIS": Feature_Encoder_VIS.state_dict(),
            "Restorm_Decoder": Restorm_Decoder.state_dict(),
            "Pre_Decoder": Pre_Decoder.state_dict(),
            "state_dict": segmodel.state_dict(),
        }
        if not os.path.exists("models/"):
            os.mkdir("models/")
        torch.save(
            checkpoint,
            os.path.join(
                "models/",
                "TRIS_stage2_epoch_" + str(epoch) + "_" + timestamp + ".pth",
            ),
        )
