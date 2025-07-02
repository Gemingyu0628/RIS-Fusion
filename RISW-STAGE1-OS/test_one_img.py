import re
from torchvision.utils import save_image
from mmengine import Config
from mmdet.structures import DetDataSample
from mmdet.registry import MODELS
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import CLIP.clip as clip
import numpy as np
import cv2
from model.model_stage1 import TRIS
from args import get_parser
import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


keywords = {
    "left": ["left", "leftmost", "left-hand", "west"],
    "right": ["right", "rightmost", "right-hand", "east"],
    "top": ["top", "upper", "topmost", "north"],
    "bottom": ["bottom", "lower", "bottommost", "south"],
    "all": ["all of"],
    "whole": ["whole", "entire"],
}


def detect_keywords(text):
    detected_keywords = {
        "left": False,
        "right": False,
        "top": False,
        "bottom": False,
        "all": False,
        "whole": False,
    }

    for key, synonyms in keywords.items():
        for synonym in synonyms:
            if re.search(r"\b" + synonym + r"\b", text, re.IGNORECASE):
                detected_keywords[key] = True
                break

    return detected_keywords


def determine_image_range(detected_keywords, x_range, y_range):
    if detected_keywords["left"]:
        x_range[1] = width // 2
    if detected_keywords["right"]:
        x_range[0] = width // 2
    if detected_keywords["top"]:
        y_range[1] = height // 2
    if detected_keywords["bottom"]:
        y_range[0] = height // 2
    return x_range, y_range


def get_transform(size=None):
    if size is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transform


def visualize_cam(normalized_heatmap, original=None, root=None):
    map_img = np.uint8(normalized_heatmap * 255)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
    if original is not None:
        original_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(heatmap_img, 0.6, original_img, 0.4, 0)
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


def prepare_data(img_path_ir, img_path_vi, text, max_length=20):
    img1 = cv2.imread(img_path_ir)
    img2 = cv2.imread(img_path_vi)

    word_ids = []
    split_text = text.split(",")
    tokenizer = clip.tokenize

    for text in split_text:
        word_id = tokenizer(text).squeeze(0)[:max_length]
        word_ids.append(word_id.unsqueeze(0))
    word_ids = torch.cat(word_ids, dim=-1)

    h, w = img1.shape[0], img1.shape[1]

    transform = get_transform()
    transform2 = get_transform(size=(320, 320))
    img1 = Image.fromarray(img1)
    img1_1 = transform(img1)
    img1_2 = transform2(img1)

    img2 = Image.fromarray(img2)
    img2_1 = transform(img2)
    img2_2 = transform2(img2)

    return img1_1, img2_1, img1_2, img2_2, word_ids, h, w


def compute_overlap_response(score_map, masks):
    response_scores = []
    for mask in masks:

        overlap_nonzero = torch.sum((score_map > 0) & (mask > 0))

        mask_nonzero = torch.sum(mask > 0)

        if mask_nonzero > 0:
            overlap_ratio = overlap_nonzero / mask_nonzero
        else:
            overlap_ratio = 0
        response_scores.append(overlap_ratio)
    return response_scores


def select_nth_instance(masks, n: int, direction: str):

    if n < 1:
        raise ValueError("n 必须 >= 1")

    
    if direction in ("left", "right"):
        axis = 1  
        reverse = direction == "right"
    elif direction in ("top", "bottom"):
        axis = 0  
        reverse = direction == "bottom"
    else:
        raise ValueError("direction 只能是 left/right/top/bottom")

    
    sorted_masks = sorted(
        masks,
        key=lambda m: m.nonzero(as_tuple=False).float().mean(dim=0)[axis],
        reverse=reverse,
    )

    
    idx = n - 1
    if idx < len(sorted_masks):
        return [sorted_masks[idx]]
    return [] 



ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}


DIRECTIONS = ("left", "right", "top", "bottom")


def parse_keywords(text: str):

    text_lc = text.lower()

   
    detected = {kw: False for kw in DIRECTIONS}  
    detected.update({ord_word: False for ord_word in ORDINAL_WORDS})  
    detected["whole"] = False
    detected["direction"] = None
    detected["ordinal"] = None

    
    if re.search(r"\b(whole|entire|整个|整幅)\b", text_lc):
        detected["whole"] = True

    
    dir_match = re.search(r"\b(left|right|top|bottom)\b", text_lc)
    if dir_match:
        direction = dir_match.group(1)
        detected[direction] = True
        detected["direction"] = direction

    
    
    for word, idx in ORDINAL_WORDS.items():
        if re.search(r"\b" + word + r"\b", text_lc):
            detected[word] = True
            detected["ordinal"] = idx
            break

    
    if detected["ordinal"] is None:
        num_match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", text_lc)
        if num_match:
            n = int(num_match.group(1))
            if 1 <= n <= 20:
                
                detected["ordinal"] = n
                
                inv_map = {v: k for k, v in ORDINAL_WORDS.items()}
                if n in inv_map:
                    detected[inv_map[n]] = True

    return detected


if __name__ == "__main__":

    img_path_ir = "data/llvip_test/ir/1.png"
    image_path_vi = "data/llvip_test/vi/1.png"
    text_path = "data/llvip_test/text/1.txt"

    #models path
    response_model_path = "weights/stage1/refcoco_train/no10/ckpt_320_epoch_13_best.pth"
    seg_model_cfg = Config.fromfile("configs/m2_convl.py")

    parse = get_parser()
    args = parse.parse_args()
    img_size = 320
    max_length = 20

    
    response_model = TRIS(args)
    response_model.cuda()
    checkpoint = torch.load(response_model_path)
    response_model.load_state_dict(checkpoint["model"], strict=True)
    response_model.eval()

    # OMG_seg
    segmodel = MODELS.build(seg_model_cfg.model)
    segmodel = segmodel.eval().cuda()
    segmodel.init_weights()
    for p in segmodel.parameters():
        p.requires_grad = False

    with open(text_path, "r") as f:
        text = f.read()

    print("Input Text is:", text)

    width = 1280
    height = 1024

    x_range = [0, width]
    y_range = [0, height]

    img_ir, img_vi, img_ir_resize, img_vi_resize, word_id, h, w = prepare_data(
        img_path_ir, image_path_vi, text, max_length
    )
    img_ir = img_ir.unsqueeze(0)
    img_vi = img_vi.unsqueeze(0)
    print(img_ir.shape)
    img_ir_resize = img_ir_resize.unsqueeze(0)
    img_vi_resize = img_vi_resize.unsqueeze(0)
    word_id = word_id  # .view(-1) #20
    B, _, H, W = img_ir.size()  # b 3 h w     b 3 1 h w   b 3 num h w
    im_w = W if W % 32 == 0 else W // 32 * 32 + 32
    im_h = H if H % 32 == 0 else H // 32 * 32 + 32
    instance_masks = []
    with torch.no_grad():
        # for i in range(B):
        img_tensor = F.pad(
            img_ir[0:1], (0, im_w - W, 0, im_h - H), "constant", 0
        ).cuda()
        batch_data_samples = [DetDataSample()]
        batch_data_samples[0].data_tag = "coco"
        batch_data_samples[0].set_metainfo(
            dict(batch_input_shape=(im_h, im_w)))
        batch_data_samples[0].set_metainfo(dict(img_shape=(H, W)))

        results = segmodel.predict(
            img_tensor, batch_data_samples, rescale=False
        )  # omg_seg
        masks_temp = results[0]
        masks_temp["ins_results"] = masks_temp["ins_results"][
            masks_temp["ins_results"].scores > 0.2
        ]  # .2
        masks = masks_temp["ins_results"].masks.unsqueeze(0).to(torch.float)

        cmap = plt.get_cmap("jet", masks.shape[1])  # jet project

        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

        # RGB
        for c in range(masks.shape[1]):
            mask = (masks[0, c] == 1).cpu()
            rgb_image[mask] = np.array(cmap(c)[:3]) * 255  # project to 0-255

        # savethe instrance segmentation result
        plt.imsave("Result_instance_seg/instances.png", rgb_image)

        masks = masks.permute(1, 0, 2, 3)  # 1 320 320 1
        
        rgb_instance = Image.open("Result_instance_seg/instances.png").convert("RGB")
        response_map = response_model(
            img_ir_resize.cuda(), img_vi_resize.cuda(), word_id.cuda()
        )  # outputs  1 1 320 320

        response_map = (response_map > 0.03).to(torch.float)
        response_map = F.interpolate(
            response_map, (h, w), align_corners=True, mode="bilinear"
        )
        response_scores = compute_overlap_response(response_map, masks)

        threshold = 0  # you can adjust this threshold
        selected_masks = [
            mask for score, mask in zip(response_scores, masks) if score > threshold
        ]
      
        selected_masks = torch.cat(selected_masks, dim=0)
    
        detected_keywords = detect_keywords(text)
        if detected_keywords["whole"]:
            x_range, y_range = determine_image_range(
                detected_keywords, x_range, y_range
            )
            for mask in selected_masks:

                coords = mask.nonzero(as_tuple=False)
                y_center, x_center = coords.float().mean(dim=0)
                if (
                    y_center >= y_range[0]
                    and y_center <= y_range[1]
                    and x_center >= x_range[0]
                    and x_center <= x_range[1]
                ):
                    instance_masks.append(mask)
        
            if len(instance_masks) > 1:
                if detected_keywords["left"]:
                    instance_masks = [
                        min(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[1],
                        )
                    ]
                elif detected_keywords["right"]:
                    instance_masks = [
                        max(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[1],
                        )
                    ]
                elif detected_keywords["bottom"]:
                    instance_masks = [
                        max(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[0],
                        )
                    ]
                elif detected_keywords["top"]:
                    instance_masks = [
                        min(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[0],
                        )
                    ]

        else:
            instance_masks = selected_masks
            if len(instance_masks) == 2:
                if detected_keywords["left"]:
                    instance_masks = [
                        min(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[1],
                        )
                    ]
                elif detected_keywords["right"]:
                    instance_masks = [
                        max(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[1],
                        )
                    ]
                elif detected_keywords["bottom"]:
                    instance_masks = [
                        max(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[0],
                        )
                    ]
                elif detected_keywords["top"]:
                    instance_masks = [
                        min(
                            instance_masks,
                            key=lambda mask: mask.nonzero(as_tuple=False)
                            .float()
                            .mean(dim=0)[0],
                        )
                    ]

            elif len(instance_masks) > 2 | detected_keywords["all"]:
                kw = parse_keywords(text)

                if kw["direction"] and kw["ordinal"]:
                    instance_masks = select_nth_instance(
                        selected_masks, n=kw["ordinal"], direction=kw["direction"]
                    )

        save_image(instance_masks[0], "Result_single_mask/final_mask.png")
