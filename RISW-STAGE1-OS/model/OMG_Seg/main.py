import gradio as gr

import numpy as np

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

from PIL import ImageDraw

import spaces

IMG_SIZE = 1024

TITLE = "<center><strong><font size='8'>OMG-Seg: Is One Model Good Enough For All Segmentation?<font></strong></center>"
CSS = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

model_cfg = Config.fromfile('app/configs/m2_convl.py')

model = MODELS.build(model_cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
model = model.eval()
model.init_weights()

mean = torch.tensor([123.675, 116.28, 103.53], device=device)[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375], device=device)[:, None, None]

visualizer = DetLocalVisualizer()

examples = [
    ["assets/000000000139.jpg"],
    ["assets/000000000285.jpg"],
    ["assets/000000000632.jpg"],
    ["assets/000000000724.jpg"],
]


class IMGState:
    def __init__(self):
        self.img = None
        self.selected_points = []
        self.available_to_set = True

    def set_img(self, img):
        self.img = img
        self.available_to_set = False

    def clear(self):
        self.img = None
        self.selected_points = []
        self.available_to_set = True

    def clean(self):
        self.selected_points = []

    @property
    def available(self):
        return self.available_to_set

    @classmethod
    def cls_clean(cls, state):
        state.clean()
        return Image.fromarray(state.img), None

    @classmethod
    def cls_clear(cls, state):
        state.clear()
        return None, None


def store_img(img, img_state):
    w, h = img.size
    scale = IMG_SIZE / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    img_numpy = np.array(img)
    img_state.set_img(img_numpy)
    print_log(f"Successfully loaded an image with size {new_w} x {new_h}", logger='current')

    return img, None


def get_points_with_draw(image, img_state, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    print_log(f"Point: {x}_{y}", logger='current')
    point_radius, point_color = 10, (97, 217, 54)

    img_state.selected_points.append([x, y])
    if len(img_state.selected_points) > 0:
        img_state.selected_points = img_state.selected_points[-1:]
        image = Image.fromarray(img_state.img)

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image

@spaces.GPU()
def segment_point(image, img_state, mode):
    output_img = img_state.img
    h, w = output_img.shape[:2]

    img_tensor = torch.tensor(output_img, device=device, dtype=torch.float32).permute((2, 0, 1))[None]
    img_tensor = (img_tensor - mean) / std

    im_w = w if w % 32 == 0 else w // 32 * 32 + 32
    im_h = h if h % 32 == 0 else h // 32 * 32 + 32
    img_tensor = F.pad(img_tensor, (0, im_w - w, 0, im_h - h), 'constant', 0)

    if len(img_state.selected_points) > 0:
        input_points = torch.tensor(img_state.selected_points, dtype=torch.float32, device=device)
        batch_data_samples = [DetDataSample()]
        selected_point = torch.cat([input_points - 3, input_points + 3], 1)
        gt_instances = InstanceData(
            point_coords=selected_point,
        )
        pb_labels = torch.zeros(len(gt_instances), dtype=torch.long, device=device)
        gt_instances.bp = pb_labels
        batch_data_samples[0].gt_instances = gt_instances
        batch_data_samples[0].data_tag = 'sam'
        batch_data_samples[0].set_metainfo(dict(batch_input_shape=(im_h, im_w)))
        batch_data_samples[0].set_metainfo(dict(img_shape=(h, w)))
        is_prompt = True
    else:
        batch_data_samples = [DetDataSample()]
        batch_data_samples[0].data_tag = 'coco'
        batch_data_samples[0].set_metainfo(dict(batch_input_shape=(im_h, im_w)))
        batch_data_samples[0].set_metainfo(dict(img_shape=(h, w)))
        is_prompt = False
    with torch.no_grad():
        results = model.predict(img_tensor, batch_data_samples, rescale=False)

    masks = results[0]
    if is_prompt:
        masks = masks[0, :h, :w]
        masks = masks > 0.  # no sigmoid
        rgb_shape = tuple(list(masks.shape) + [3])
        color = np.zeros(rgb_shape, dtype=np.uint8)
        color[masks] = np.array([97, 217, 54])
        output_img = (output_img * 0.7 + color * 0.3).astype(np.uint8)
        output_img = Image.fromarray(output_img)
    else:
        if mode == 'Panoptic Segmentation':
            output_img = visualizer._draw_panoptic_seg(
                output_img,
                masks['pan_results'].to('cpu').numpy(),
                classes=CocoPanopticDataset.METAINFO['classes'],
                palette=CocoPanopticDataset.METAINFO['palette']
            )
        elif mode == 'Instance Segmentation':
            masks['ins_results'] = masks['ins_results'][masks['ins_results'].scores > .2]
            output_img = visualizer._draw_instances(
                output_img,
                masks['ins_results'].to('cpu').numpy(),
                classes=CocoPanopticDataset.METAINFO['classes'],
                palette=CocoPanopticDataset.METAINFO['palette']
            )
    return image, output_img


def register_title():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(TITLE)


def register_point_mode():
    with gr.Tab("Point mode"):
        img_state = gr.State(IMGState())
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_p = gr.Image(label="Input Image", type="pil")

            with gr.Column(scale=1):
                segm_p = gr.Image(label="Segment", interactive=False, type="pil")

        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    ["Panoptic Segmentation", "Instance Segmentation"],
                    label="Mode",
                    value="Panoptic Segmentation",
                    info="Please select the segmentation mode. (Ignored if provided with prompt.)"
                )
                with gr.Row():
                    with gr.Column():
                        segment_btn = gr.Button("Segment", variant="primary")
                    with gr.Column():
                        clean_btn = gr.Button("Clean Prompts", variant="secondary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[img_p, img_state],
                    outputs=[img_p, segm_p],
                    examples_per_page=4,
                    fn=store_img,
                    run_on_click=True
                )

        img_p.upload(
            store_img,
            [img_p, img_state],
            [img_p, segm_p]
        )

        img_p.select(
            get_points_with_draw,
            [img_p, img_state],
            img_p
        )

        segment_btn.click(
            segment_point,
            [img_p, img_state, mode],
            [img_p, segm_p]
        )

        clean_btn.click(
            IMGState.cls_clean,
            img_state,
            [img_p, segm_p]
        )

        img_p.clear(
            IMGState.cls_clear,
            img_state,
            [img_p, segm_p]
        )


def build_demo():
    with gr.Blocks(css=CSS, title="RAP-SAM") as _demo:
        register_title()
        register_point_mode()
    return _demo


if __name__ == '__main__':
    demo = build_demo()

    demo.queue(api_open=False)
    demo.launch(server_name='0.0.0.0')
