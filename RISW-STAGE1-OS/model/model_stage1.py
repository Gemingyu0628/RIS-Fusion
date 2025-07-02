import os

# os.environ['CUDA_ENABLE_DEVICES'] = '0,1'
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
grand_parent_dir = os.path.dirname(parent_dir)

# 将上级目录添加到sys.path
sys.path.append(grand_parent_dir)
import numpy as np
import torch
import torch.nn as nn

# import transformers
import torch.nn.functional as F
from model.attn import  mainfold_attention

import CLIP.clip as clip
from model.utils import Upsample
import time


class TRIS(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.bert_model = args.bert_tokenizer

        if args.backbone == "clip-RN50":
            last_vis_channel = 2048
            self.textdim = 1024
        elif args.backbone == "clip-RN101":
            last_vis_channel = 2048
            self.textdim = 512

        device = "cuda" if torch.cuda.is_available() else "cpu"
        type = args.backbone.split("-")[-1]
        clip_model, _ = clip.load(
            type, device=device, jit=False, txt_length=args.max_query_len
        )
        # clip_model, _ = clip.load('../clip_weights/RN50.pt', device=device, jit=False, txt_length=args.max_query_len)
        clip_model = clip_model.float()

        if "clip-RN" in args.backbone:
            self.backbone = clip_model

        self.vis_project = nn.Conv2d(
            in_channels=last_vis_channel,
            out_channels=args.hidden_dim,
            kernel_size=1,
            bias=True,
        )
        self.ir_project = nn.Conv2d(
            in_channels=last_vis_channel,
            out_channels=args.hidden_dim,
            kernel_size=1,
            bias=True,
        )

        self.lan_project_new = nn.Linear(512, args.hidden_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.args.attn_multi > 0:
            # self.attn_fusion = bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim)
            self.attn_fusion = mainfold_attention(
                args.hidden_dim, args.hidden_dim, lan_chans=args.hidden_dim
            )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        newly_add_params = [self.vis_project, self.ir_project, self.lan_project_new]
        try:
            newly_add_params.append(self.attn_fusion)
        except:
            print("no attn fusion")

        backbone = self.backbone
        return (
            list(backbone.parameters()),
            list(nn.ModuleList(newly_add_params).parameters()),
        )

    def forward(self, x_ir, x_vis, word_id):  # ir vis
        aa = time.time()
        img_size = x_ir.shape[2:]
        B, _, H, _ = x_ir.size()

        _, hidden = self.backbone.encode_text(word_id)

        hidden = _  # 特别添加的   20 c

        c1, c2, c3, c4, _ = self.backbone.encode_image(x_ir)
        c1_vis, c2_vis, c3_vis, c4_vis, _ = self.backbone.encode_image(x_vis)

        lan = self.lan_project_new(hidden)

        ir = self.ir_project(c4.float())
        vis = self.vis_project(c4_vis.float())

        h_, w_ = vis.shape[2:]
        ir_trans = ir.flatten(2).transpose(1, 2)
        vis_trans = vis.flatten(2).transpose(1, 2)

        lan = lan.unsqueeze(0).repeat(B, 1, 1, 1)
        norm_ir = ir_trans / ir_trans.norm(dim=-1, keepdim=True)
        norm_vis = vis_trans / vis_trans.norm(dim=-1, keepdim=True)

        norm_lan = lan / lan.norm(dim=-1, keepdim=True)

        # if self.args.attn_multi > 0:
        #     new_vis, new_lan = self.attn_fusion(norm_vis.permute(0, 2, 1).reshape(B, -1, h_, w_), norm_lan.transpose(1, 2))
        #     norm_vis = new_vis.flatten(2).transpose(1, 2) * 0.1 + norm_vis
        #     norm_lan = new_lan * 0.1 + norm_lan
        # score = torch.bmm(norm_vis, norm_lan.transpose(1,2))

        score = []
        if (
            self.args.attn_multi > 0
        ):  
            regulation_loss_final = torch.tensor(
                0.0, requires_grad=True, device="cuda:0"
            )
            for i in range(B):
                new_image, new_lan, regulation_loss_temp = self.attn_fusion(
                    norm_ir.permute(0, 2, 1).reshape(B, -1, h_, w_),
                    norm_vis.permute(0, 2, 1).reshape(B, -1, h_, w_),
                    norm_lan[:, i, :, :].transpose(1, 2),
                )
                # print(new_vis.mean(),new_lan.mean(), norm_lan[:,i,:].mean(), norm_vis.mean())
                norm_image_temp = (
                    new_image.flatten(2).transpose(1, 2) * 0.1
                    + 0.5 * norm_ir
                    + 0.5 * norm_vis
                )  # b 10*10 1024
                norm_lan_temp = new_lan * 0.1 + norm_lan[:, i, :, :]  # # b 1 1024
                score_temp = torch.bmm(
                    norm_image_temp, norm_lan_temp.transpose(1, 2)
                ).mean(
                    dim=-1, keepdim=True
                )  # 像素级响应分数计算  b 100 b 1个像素对b段文本 100,1
                score.append(score_temp)
                regulation_loss_final = regulation_loss_final + regulation_loss_temp
            regulation_loss_final /= B
            score = torch.cat(score, dim=-1)  # b 100 b

        logit_scale = self.logit_scale.exp()
        score = logit_scale * score

        if self.training:
            score_t = score.transpose(1, 2).reshape(B, -1, h_, w_)
            bg = torch.ones_like(score_t[:, :1])
            score_t = torch.cat([bg, score_t], 1)

            bs, c, h, w = score_t.size()

            masks = F.softmax(score_t, dim=1)  # default
            # masks = F.sigmoid(score_t)

            features = score_t.view(bs, c, -1)
            masks_ = masks.view(bs, c, -1)

            # classification loss
            cls_10 = features.mean(-1)
            cls_11 = torch.max(features, dim=-1).values
            cls_1 = cls_10 + cls_11

            # # focal penalty loss
            cls_2 = focal_loss(
                masks_.mean(-1), p=self.args.FOCAL_P, c=self.args.FOCAL_LAMBDA
            )

            # # adding the losses together
            cls_out = cls_1[:, 1:] + cls_2[:, 1:]
            # print(cls_out)

            # foreground stats
            masks_ = masks_[:, 1:]
            labels = torch.eye(bs).to(x_ir.device)
            cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        masks_out = []
        for i in range(B):
            masks_out.append(score[i, :, i].view(1, h_, w_))
        masks_out = torch.stack(masks_out, dim=0)
        seg_final_out = Upsample(masks_out, img_size)
        bb = time.time()
        # print('modeltime', bb - aa,flush=True)
        if self.training:
            return (
                cls_out,
                cls_fg,
                F.relu(seg_final_out),
                torch.sigmoid(seg_final_out),
                logit_scale,
                regulation_loss_final,
            )
        else:
            return F.relu(seg_final_out)


def focal_loss(x, p=1, c=0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from args import get_parser

    parse = get_parser()
    args = parse.parse_args()

    m = TRIS(args).cuda()
    x = torch.randn(4, 3, 320, 320)
    x = torch.tensor(x, dtype=torch.float32)
    word_id = torch.ones(4, 20)
    word_id = torch.tensor(word_id, dtype=torch.int64)
    att_mask = torch.ones(4, 20)

    output = m(x.cuda(), x.cuda(), word_id.cuda())
    print("success !!!")
