import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from CRIS.model.clip import build_model, ModifiedResNet

from .layers import FPN, Projector, TransformerDecoder



class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        self.sigmoid = nn.Sigmoid()
        self.resent = ModifiedResNet(layers=(3, 4, 6, 3),
                                         output_dim=1024,
                                         heads=32,
                                         input_resolution=224,
                                         width=64)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        #print(img.shape)
        vis = self.resent(img)
        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)
        
        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask) 

            return self.sigmoid(pred.detach()), mask, loss #
        
        else:
            return self.sigmoid(pred.detach())
            # return pred # .detach()
