import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] ='1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.utils
from net.restormer import Restormer_Encoder, Restormer_Decoder, preFuse_Decoder
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from torchvision.utils import save_image
import time
import clip as clip
from skimage.io import imread
import skimage
from skimage import morphology
import argparse
import CRIS.utils.config as config
from CRIS.model import build_segmenter
from torch.nn import functional as F
import cv2
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='CRIS/config/refcoco/cris_r50.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


args = get_parser()
segmodel, _ = build_segmenter(args)

def convert(imgA,imgB):
    transform = transforms.Resize((256,256))
    imgB = torch.from_numpy(imgB)
    if not isinstance(imgB, torch.FloatTensor):
        imgB = imgB.float()
    imgB.div_(255.)
    # imgB = transform(imgB)

    imgA = torch.from_numpy(imgA)
    if not isinstance(imgA, torch.FloatTensor):
        imgA = imgA.float()
    imgA.div_(255.)
    # imgA = transform(imgA)

    return imgA,imgB
transform = transforms.Resize((416,416))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"TRIS_stage2_epoch_260_11-03-14-12.pth" 
modelname = "TRIS_stage2_epoch_260"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Feature_Encoder_IR = nn.DataParallel(Restormer_Encoder()).to(device)
Feature_Encoder_VIS = nn.DataParallel(Restormer_Encoder(inp_channels=3)).to(device)
Restorm_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
Pre_Decoder = nn.DataParallel(preFuse_Decoder()).to(device)
segmodel = nn.DataParallel(segmodel).to(device)


Feature_Encoder_IR.load_state_dict(torch.load(ckpt_path)['Feature_Encoder_IR'], strict=True)
Feature_Encoder_VIS.load_state_dict(torch.load(ckpt_path)['Feature_Encoder_VIS'], strict=True)
Restorm_Decoder.load_state_dict(torch.load(ckpt_path)['Restorm_Decoder'], strict=True)
Pre_Decoder.load_state_dict(torch.load(ckpt_path)['Pre_Decoder'], strict=True)
segmodel.load_state_dict(torch.load(ckpt_path)['segmodel'], strict=True)

Feature_Encoder_IR.eval()
Feature_Encoder_VIS.eval()
Restorm_Decoder.eval()
Pre_Decoder.eval()
segmodel.eval()

import os
from pathlib import Path
from skimage.io import imread

# ---------------------------------------------------------------------
# 1. 路径与通用配置
# ---------------------------------------------------------------------
root_ir   = Path("Dataset/IVT_final/IVT_ir")
root_vis  = Path("Dataset/IVT_final/IVT_vis")
root_out  = Path("Dataset/IVT_final/fusion_ris")
root_out.mkdir(parents=True, exist_ok=True)          # 创建顶层输出文件夹
img_exts = {".png", ".jpg", ".jpeg", ".bmp"}         # 支持的图片后缀

all_time, img_cnt = 0.0, 0

# ---------------------------------------------------------------------
# 2. 遍历 IR 目录，保持相对层级
# ---------------------------------------------------------------------
with torch.no_grad():
    for ir_dir, _, files in os.walk(root_ir):
        ir_dir  = Path(ir_dir)
        rel_dir = ir_dir.relative_to(root_ir)        # 相对于 IR 根目录的路径
        vis_dir = root_vis / rel_dir                 # 对应的 VIS 子目录
        out_dir = root_out / rel_dir                 # 输出子目录
        out_dir.mkdir(parents=True, exist_ok=True)

        # 遍历当前子目录下的 IR 图
        for fname in files:
            if Path(fname).suffix.lower() not in img_exts:
                continue

            ir_path  = ir_dir / fname
            vis_path = vis_dir / fname               # 假设文件名一致
            if not vis_path.exists():
                print(f"[WARN] 可见光缺失: {vis_path}")
                continue

            # -----------------------------------------------------------------
            # 3. 数据读取 & 预处理   ← 保持你原先的逻辑即可
            # -----------------------------------------------------------------
            imageA = imread(str(ir_path)).astype(np.float32)[None, None, :, :]
            imageB = image_read_cv2(str(vis_path), mode='RGB').transpose(2, 0, 1)[np.newaxis, ...]

            # 若没有对应文本描述，可给一个空 token（或者按你需求调整）
            T_text = torch.zeros((1, 17), dtype=torch.long, device=device)

            imageA, imageB = convert(imageA, imageB)

            # -----------------------------------------------------------------
            # 4. 前向推理 & 融合
            # -----------------------------------------------------------------
            start = time.time()

            Feature_IR  = Feature_Encoder_IR(imageA)
            Feature_VIS = Feature_Encoder_VIS(imageB)

            pre_fuse = Pre_Decoder(Feature_IR, Feature_VIS)
            
    
            end = time.time()

            # -----------------------------------------------------------------
            # 5. 保存 (保留子目录层级)
            # -----------------------------------------------------------------
            save_image(pre_fuse, str(out_dir / fname))

            # 6. 统计时间
            all_time += (end - start)
            img_cnt  += 1

# ---------------------------------------------------------------------
# 7. 平均耗时
# ---------------------------------------------------------------------
if img_cnt:
    print(f"Processed {img_cnt} image pairs, average time per pair: {all_time / img_cnt:.4f} s")
else:
    print("No image pairs were processed.")






