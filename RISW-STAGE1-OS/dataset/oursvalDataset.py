import os
from torch.utils.data import Dataset
from PIL import Image
import clip as clip
import numpy as np 
import logging
import torch
logging.getLogger("PIL").setLevel(logging.WARNING)

import re
def natural_sort_key(s):
    """用于按自然数排序的key函数"""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

class CustomDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, gt_dir, text_dir, image_transforms=None, max_tokens=20, bert_tokenizer='clip',):
        """
        ir_dir: 红外图像的文件夹路径
        vis_dir: 可见光图像的文件夹路径
        text_dir: 文本描述文件夹路径
        transform: 图像预处理变换
        """
        self.clip = ('clip' in bert_tokenizer)
        self.tokenizer = clip.tokenize
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.gt_dir = gt_dir
        self.text_dir = text_dir
        self.image_transforms = image_transforms
        self.max_tokens = max_tokens

        # 获取ir和vis文件夹中所有图片的文件名（假设文件名是按数字排序的）
        self.ir_images = sorted(os.listdir(ir_dir))#, key=natural_sort_key
        self.vis_images = sorted(os.listdir(vis_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.total_images = len(self.ir_images)  # 假设ir和vis图像数量相同

    def __len__(self):
        return self.total_images

    def _load_image(self, folder, image_name):
        """加载图像"""
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path).convert("RGB")  # 转为RGB格式
        image_source = image
        if self.image_transforms:
            image, _ = self.image_transforms(image,image)
        return image,image_path

    def _load_text(self, text_id, idx):
        """加载对应的文本描述"""
        text_files = [f"{idx+1}.txt"]
        positive_text = os.path.join(self.text_dir, text_files[0])

        with open(positive_text, "r") as f:
            positive_desc = f.read()
        
        negative_descs = []

        return positive_desc
    def __getitem__(self, idx):
        """
        返回 ir_image, vis_image, positive_text, negative_texts
        """
        # 加载图像
        ir_image,image_path_1 = self._load_image(self.ir_dir, self.ir_images[idx])
        vis_image,image_path_2 = self._load_image(self.vis_dir, self.vis_images[idx])
        target_mask, mask_path = self._load_image(self.gt_dir, self.gt_images[idx])
        
        # 加载对应的文本描述
        positive_text = self._load_text(self.text_dir, idx)

        word_id_pos = self.tokenizer(positive_text).squeeze(0)[:self.max_tokens] #从1,77 到 20
        word_id_pos = np.array(word_id_pos)


        samples = {
            "img_ir": ir_image,
            "img_vis": vis_image,
            "word_ids": word_id_pos,
        }
        
        targets = {
            "target": target_mask,
            "sentences": positive_text,
        }

        return samples, targets
    
if __name__ == '__main__':
    from transform import get_transform
    import numpy as np 
    import json 
    from torch.utils.data import DataLoader

    ir_dir = "data/IVT_train/ir"
    vis_dir = "data/IVT_train/vis"
    gt_dir = "data/IVT_train/gt"
    text_dir = "data/IVT_train/text"

    refcoco_train = CustomDataset(ir_dir=ir_dir, vis_dir=vis_dir, text_dir=text_dir, image_transforms=get_transform(320, train=False),max_tokens=20, bert_tokenizer='clip') 
    train_loader=DataLoader(refcoco_train,
                            batch_size=12,
                            num_workers=2,
                            pin_memory=True,
                            sampler=None)
    for idx,(img, target) in enumerate(train_loader):
        print(idx, img["img_ir"].shape)

        # if idx > 10: break 