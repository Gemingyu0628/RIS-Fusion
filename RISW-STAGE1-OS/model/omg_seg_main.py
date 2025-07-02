from PIL import Image

import torch
import torch.nn.functional as F
from PIL import Image

# mm libs
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config, print_log
from mmengine.structures import InstanceData

from mmdet.datasets.coco_panoptic import CocoPanopticDataset
import numpy as np
from PIL import ImageDraw
import os
from time import time

model_cfg = Config.fromfile('app/configs/m2_convl.py')

model = MODELS.build(model_cfg.model)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
model = model.to(device=device)
model = model.eval()
model.init_weights()

mean = torch.tensor([123.675, 116.28, 103.53], device=device)[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375], device=device)[:, None, None]


mean = torch.tensor([123.675, 116.28, 103.53], device=device)[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375], device=device)[:, None, None]
IMG_SIZE = 1024

image_dir ="/home/wzy/code/wzyz/TRIS-main/model/imgs"
image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
images= []
for filename in image_filenames:
    img = Image.open(os.path.join(image_dir, filename)).convert('RGB')
    images.append(img)

# imagename = "twopeople"
# img = Image.open("/home/fk/code/GMY/OMG_Seg/imgs/"+ imagename +".png")
number = 0
for img in images:
    
    w, h = img.size
    scale = IMG_SIZE / max(w, h)
    new_w = int(w* scale ) # * scale
    new_h = int(h* scale ) # * scale
    img = img.resize((new_w, new_h),resample=Image.Resampling.BILINEAR)
    image_numpy = np.array(img)


    output_img= image_numpy
    h, w = output_img.shape[:2]

    img_tensor = torch.tensor(output_img, device=device, dtype=torch.float32).permute((2, 0, 1))[None]
    img_tensor = (img_tensor - mean) / std

    im_w = w if w % 32 == 0 else w // 32 * 32 + 32
    im_h = h if h % 32 == 0 else h // 32 * 32 + 32
    img_tensor = F.pad(img_tensor, (0, im_w - w, 0, im_h - h), 'constant', 0)

    a = time()
    with torch.no_grad():
        batch_data_samples = [DetDataSample()]
        batch_data_samples[0].data_tag = 'coco'
        batch_data_samples[0].set_metainfo(dict(batch_input_shape=(im_h, im_w)))
        batch_data_samples[0].set_metainfo(dict(img_shape=(h, w)))
        results = model.predict(img_tensor, batch_data_samples, rescale=False)
    b = time()
    print(b-a)
    masks = results[0]

    masks['ins_results'] = masks['ins_results'][masks['ins_results'].scores > .2]

    # output_img = visualizer._draw_instances(
    #             output_img,
    #             masks['ins_results'].to('cpu').numpy(),
    #             classes=CocoPanopticDataset.METAINFO['classes'],
    #             palette=CocoPanopticDataset.METAINFO['palette']
    #         )

    # 获取类别名称列表
    labels = masks['ins_results'].labels.to('cpu').numpy()
    classes = CocoPanopticDataset.METAINFO['classes']
    category_names = [classes[label] for label in labels]

    for i in range(masks['ins_results'].scores.shape[0]):
        output_img = masks['ins_results'].masks[i].cpu().numpy()*255
        output_image = Image.fromarray(output_img.astype(np.uint8))
        output_image.save("/home/fk/code/GMY/OMG_Seg/seg_masks/"+ "mask" + "_" + str(number) + "_" + str(category_names[i]) + str(i) +".png")
    
    number = number + 1
