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

ckpt_path=r"model_stage2.pth" 
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
segmodel.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

Feature_Encoder_IR.eval()
Feature_Encoder_VIS.eval()
Restorm_Decoder.eval()
Pre_Decoder.eval()
segmodel.eval()

for dataset_name in ["one_img_test"]:# M3FD_test, LLVIP_TEST
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder_img=os.path.join('test_results',dataset_name)
    if os.path.exists(test_out_folder_img) == False:
        os.mkdir(test_out_folder_img)

img_name = "2.png"
with torch.no_grad():
    
    imageA = image_read_cv2(os.path.join(test_folder,"ir",img_name), mode='RGB').transpose(2,0,1)[np.newaxis,0:1, ...] # imread(os.path.join(test_folder,"ir",img_name)).astype(np.float32)[None,None, :, :]
    imageB = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='RGB').transpose(2,0,1)[np.newaxis, ...]
    name=img_name[:-4]
    print(name)
    # with open(os.path.join(test_folder,"text",str(name)+'.txt'), 'r') as file:
    #     T_text = file.read()
    #     print(T_text)
    #     T_text = clip.tokenize(T_text).squeeze(0)[:17] 
    #     T_text = np.array(T_text)
    
    T_text = 'The person on the first left of all persons.'
    print(T_text)
    T_text = clip.tokenize(T_text).squeeze(0)[:17] 
    T_text = np.array(T_text)
    

    T_text = torch.from_numpy(T_text[None, :])
    
    imageA, imageB = convert(imageA, imageB)

    Feature_IR = Feature_Encoder_IR(imageA)
    Feature_VIS = Feature_Encoder_VIS(imageB)

    pre_fuse = Pre_Decoder(Feature_IR, Feature_VIS)
    pre_fuse = transform(pre_fuse)

    seg = segmodel(pre_fuse, T_text)
    
    if seg.shape[-2:] != imageA.shape[-2:]:
        seg = F.interpolate(seg, size=imageA.shape[-2:], mode='bicubic', align_corners=True)


    seg = seg>0.35
    seg = seg.float()
    

    Feature_IR_emphasized = Feature_IR * seg
    Feature_IR_unemphasized = Feature_IR * (1-seg)

    Feature_VIS_emphasized = Feature_VIS * seg
    Feature_VIS_unemphasized = Feature_VIS * (1-seg)

    Feature_Fuse_emphasized = torch.cat((Feature_IR_emphasized, Feature_VIS_emphasized), dim=1)
    Feature_Fuse_unemphasized = torch.cat((Feature_IR_unemphasized, Feature_VIS_unemphasized), dim=1)
    Fused_final = Restorm_Decoder(Feature_Fuse_emphasized, Feature_Fuse_unemphasized)

    save_image(Fused_final, test_out_folder_img +'/'+name+'.bmp')








