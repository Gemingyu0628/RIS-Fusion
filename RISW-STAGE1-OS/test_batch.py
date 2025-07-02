import os
import torch
# from dataset.transform import get_transform
from args import get_parser
from model.model_stage1 import TRIS 
import cv2 
import numpy as np 
import CLIP.clip as clip 
from PIL import Image 
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mmengine.logging import MMLogger
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config
from mmengine.structures import InstanceData
from torchvision.utils import save_image
import re
from tqdm import tqdm
# 定义关键词及其近义词
keywords = {
    'left': ['left', 'leftmost', 'left-hand', 'west'],
    'right': ['right', 'rightmost', 'right-hand', 'east'],
    'top': ['top', 'upper', 'topmost', 'north'],
    'bottom': ['bottom', 'lower', 'bottommost', 'south'],
    'all': ['all of'],
    'whole': ['whole', 'entire'],
    'person': ['person', 'people', 'persons'],
    'car': ['car', 'vehicle', 'automobile', 'auto', 'motorcar']

}


def detect_keywords(text):
    detected_keywords = {
        'left': False,
        'right': False,
        'top': False,
        'bottom': False,
        'all': False,
        'whole': False,
        'person': False,
        'car': False,
    }
    
    for key, synonyms in keywords.items():
        for synonym in synonyms:
            if re.search(r'\b' + synonym + r'\b', text, re.IGNORECASE):
                detected_keywords[key] = True
                break
    
    return detected_keywords



def determine_image_range(detected_keywords,x_range, y_range):
    # 原点是左上角点
    if detected_keywords['left']:
        x_range[1] = width // 2
    if detected_keywords['right']:
        x_range[0] = width // 2
    if detected_keywords['top']:
        y_range[1] = height // 2
    if detected_keywords['bottom']:
        y_range[0] = height // 2
    return x_range, y_range


def get_transform(size=None):
    if size is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def visualize_cam(normalized_heatmap, original=None, root=None):
    map_img = np.uint8(normalized_heatmap * 255)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
    if original is not None:
        original_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(heatmap_img, .6, original_img, 0.4, 0)
    else:
        img = heatmap_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if root is None:
        return img 
    plt.imsave(root, img)

def get_norm_cam(cam):
    cam = torch.clamp(cam, min=0)
    cam_t = cam.unsqueeze(0).unsqueeze(0).flatten(2)
    cam_max = torch.max(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    cam_min = torch.min(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)
    norm_cam = norm_cam.squeeze(0).squeeze(0).cpu().numpy()
    return norm_cam

def prepare_data(img_path_ir,img_path_vi, text, max_length=20):
    img1 = cv2.imread(img_path_ir)
    img2 = cv2.imread(img_path_vi)

    word_ids = []
    split_text = text.split(',')
    tokenizer = clip.tokenize

    for text in split_text:
        word_id = tokenizer(text).squeeze(0)[:max_length]
        word_ids.append(word_id.unsqueeze(0))
    word_ids = torch.cat(word_ids, dim=-1)

    h, w = img1.shape[0],img1.shape[1]


    img1 = Image.fromarray(img1)
    transform = get_transform()
    img1 = transform(img1)

    img2 = Image.fromarray(img2)
    transform = get_transform() # size=img_size
    img2 = transform(img2)


    return img1, img2, word_ids, h, w

def compute_overlap_response(score_map, masks):
    response_scores = []
    for mask in masks:
        # 计算重叠区域中的非零像素数量
        overlap_nonzero = torch.sum((score_map > 0) & (mask > 0))

        # 计算mask中的非零像素数量
        mask_nonzero = torch.sum(mask > 0)

        # 计算重叠响应分数
        if mask_nonzero > 0:
            overlap_ratio = overlap_nonzero / mask_nonzero
        else:
            overlap_ratio = 0
        response_scores.append(overlap_ratio)
    return response_scores

if __name__ == '__main__':
    import os 
    from natsort import natsorted
    os.environ['CUDA_ENABLE_DEVICES'] = '0'
    parse=get_parser()
    args=parse.parse_args()
    img_size = 320 
    max_length = 20 

    model=TRIS(args)
    model.cuda()

    model_path = 'weights/stage1/refcoco_train/no10/ckpt_320_epoch_13_best.pth' 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    img_path_ir = os.listdir('data/LLVIP_TEST_RENAME/IR')
    img_path_ir = natsorted(img_path_ir)
    image_path_vi = os.listdir('data/LLVIP_TEST_RENAME/VI')
    image_path_vi = natsorted(image_path_vi)
    text_path = os.listdir('data/LLVIP_TEST_RENAME/TEXT')
    text_path = natsorted(text_path)

    model_cfg = Config.fromfile('app/configs/m2_convl.py')
    segmodel = MODELS.build(model_cfg.model)
    segmodel = segmodel.eval().cuda()
    segmodel.init_weights()
    for p in segmodel.parameters():
        p.requires_grad = False


    for i in tqdm(range(0, len(img_path_ir), 1)):
        img_path_ir_single = 'data/LLVIP_TEST_RENAME/IR/' + img_path_ir[i]
        image_path_vi_single = 'data/LLVIP_TEST_RENAME/VI/' + image_path_vi[i]
        text_path_single = 'data/LLVIP_TEST_RENAME/TEXT/' + text_path[i]
        with open(text_path_single, 'r') as f:
            text = f.read()
        
        detected_keywords = detect_keywords(text)

        img_ir,img_vi, word_id, h, w = prepare_data(img_path_ir_single, image_path_vi_single, text, max_length)


        width = w
        height = h
        
        x_range = [0, width]
        y_range = [0, height]

        img_ir = img_ir.unsqueeze(0)
        img_vi = img_vi.unsqueeze(0)
        word_id = word_id#.view(-1) #20
        B,_,H,W = img_ir.size()  #  b 3 h w     b 3 1 h w   b 3 num h w
        im_w = W if W % 32 == 0 else W // 32 * 32 + 32
        im_h = H if H % 32 == 0 else H // 32 * 32 + 32
        instance_masks = []        
        with torch.no_grad(): # person 0 car 2
            # for i in range(B):
            img_tensor = F.pad(img_ir[0:1], (0, im_w - W, 0, im_h -H), 'constant', 0).cuda()
            img_tensor2 = F.pad(img_vi[0:1], (0, im_w - W, 0, im_h -H), 'constant', 0).cuda()
            batch_data_samples = [DetDataSample()]
            batch_data_samples[0].data_tag = 'coco'
            batch_data_samples[0].set_metainfo(dict(batch_input_shape=(im_h, im_w)))
            batch_data_samples[0].set_metainfo(dict(img_shape=(H, W)))
            results = segmodel.predict( img_tensor, batch_data_samples, rescale=False) # img_tensor*0.5+
            masks_temp = results[0]
            masks_temp['ins_results'] = masks_temp['ins_results'][masks_temp['ins_results'].scores > .8] # .2
            outpath = "./pseudomask_r8/"
            # masks_ins = []
            # if detected_keywords['person']:
            #     for j in range(masks_temp['ins_results'].scores.shape[0]):
            #         if masks_temp['ins_results'].labels[j] in [0,]:
            #             masks_ins.append(True)
            #         else:
            #             masks_ins.append(False)

            # if detected_keywords['car']:
            #     for j in range(masks_temp['ins_results'].scores.shape[0]):
            #         if masks_temp['ins_results'].labels[j] in [2,]:
            #             masks_ins.append(True)
            #         else:
            #             masks_ins.append(False)

            # if len(masks_ins) == 0:
            #     continue
            # masks_temp['ins_results'] = masks_temp['ins_results'][masks_ins] # false会否定掉仅存的mask

            

            masks = masks_temp['ins_results'].masks.to(torch.float)

            # outputs = model(img_ir.cuda(),img_vi.cuda(), # 
            #                 word_id.cuda()) #  outputs  1 1 320 320

            # outputs = (outputs>0.05).to(torch.float)

            # response_scores = compute_overlap_response(outputs, masks)


            # threshold = 0  # 你可以调整这个阈值
            # selected_masks = [mask for score, mask in zip(response_scores, masks) if score > threshold]
            selected_masks = masks
            # selected_masks = torch.cat(selected_masks, dim=0)

            if len(selected_masks) == 0:
                continue

            if detected_keywords['whole']:
                x_range, y_range = determine_image_range(detected_keywords,x_range, y_range)
                for mask in selected_masks:

                    coords = mask.nonzero(as_tuple=False) 
                    y_center, x_center = coords.float().mean(dim=0) 
                    if y_center >= y_range[0] and y_center <= y_range[1] and x_center >= x_range[0] and x_center <= x_range[1]:
                        instance_masks.append(mask)
                instance_masks = selected_masks
                if len(instance_masks) > 1:
                    if detected_keywords['left']:
                        instance_masks = [min(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[1])]
                    elif detected_keywords['right']:
                        instance_masks = [max(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[1])]
                    elif detected_keywords['bottom']:
                        instance_masks = [max(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[0])]
                    elif detected_keywords['top']:
                        instance_masks = [min(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[0])]
                    
                    
            
            else :
                instance_masks = selected_masks
                if detected_keywords['left']:
                    instance_masks = [min(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[1])]
                elif detected_keywords['right']:
                    instance_masks = [max(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[1])]
                elif detected_keywords['bottom']:
                    instance_masks = [max(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[0])]
                elif detected_keywords['top']:
                    instance_masks = [min(instance_masks, key=lambda mask: mask.nonzero(as_tuple=False).float().mean(dim=0)[0])]
                
            save_image(instance_masks[0],outpath+str(i+1)+".png")

