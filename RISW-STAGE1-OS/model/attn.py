from turtle import forward
import torch
import torch.nn as nn
from torch import Tensor, cartesian_prod, normal
from torch.nn import functional as F
import math
from sklearn.manifold import Isomap


class PixelAttention(nn.Module):
    #  https://github.com/yz93/LAVT-RIS
    def __init__(
        self,
        visual_channel,  # input visual features' channel
        language_channel,  # input language features,
    ) -> None:

        super().__init__()
        self.Ci = visual_channel
        self.Ct = language_channel

        # convolution op
        # Ct  = > Ci
        self.Wk = nn.Conv1d(self.Ct, self.Ci, 1)
        self.Wv = nn.Conv1d(self.Ct, self.Ci, 1)
        # Ci  = > Ci
        self.Wq = nn.Conv2d(self.Ci, self.Ci, 1)
        self.Wm = nn.Conv2d(self.Ci, self.Ci, 1)
        self.Ww = nn.Conv2d(self.Ci, self.Ci, 1)
        self.Wo = nn.Conv2d(self.Ci, self.Ci, 1)

        # instance normalization
        self.ins_q = nn.InstanceNorm2d(self.Ci, affine=True)
        self.ins_w = nn.InstanceNorm2d(self.Ci, affine=True)
        # self.ins_q = nn.BatchNorm2d(self.Ci)
        # self.ins_w = nn.BatchNorm2d(self.Ci)

    def forward(self, vis_feat: Tensor, lan_feat: Tensor):
    
        N, Ci, H, W = vis_feat.size()
        N, Ct, T = lan_feat.size()
        Lk, Lv = self.Wk(lan_feat), self.Wv(lan_feat)  # [N,Ci,T]
        Vq = self.ins_q(self.Wq(vis_feat))  # [N,Ci,H,W]

        Vq = Vq.view(N, Ci, H * W).permute(0, 2, 1)  # [N,H*W,Ci]
        # get attention map
        attn = F.softmax(Vq.matmul(Lk) / math.sqrt(Ci), dim=2)  # [N,H*W,T]

        Lv = Lv.permute(0, 2, 1)  # [N,T,Ci]
        G = attn.matmul(Lv)  # [N,H*W,Ci]

        G = G.permute(0, 2, 1).view(N, Ci, H, W)  # [N,Ci,H,W]
        Gi = self.ins_w(self.Ww(G))  # [N,Ci,H,W]

        Vo = F.relu(self.Wm(vis_feat))  # [N,Ci,H,W]
        out_feat = F.relu(self.Wo(Vo * Gi))  # [N,Ci,H,W]

        return out_feat


class bilateral_prompt(nn.Module):
    def __init__(self, vis_chans, lan_chans, m_chans=None) -> None:
        super().__init__()
        if m_chans is None:
            m_chans = vis_chans
        self.v_proj1 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )
        self.v_proj2 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )
        self.v_proj3 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )

        self.t_proj1 = nn.Sequential(
            nn.Linear(lan_chans, m_chans), nn.ReLU(inplace=True)
        )
        self.t_proj2 = nn.Sequential(
            nn.Linear(lan_chans, m_chans), nn.ReLU(inplace=True)
        )
        self.t_proj3 = nn.Sequential(
            nn.Linear(lan_chans, m_chans), nn.ReLU(inplace=True)
        )

        self.v_output = nn.Sequential(
            nn.Conv2d(m_chans, vis_chans, 1), nn.InstanceNorm2d(vis_chans, affine=True)
        )

        self.t_output = nn.Sequential(nn.Linear(m_chans, lan_chans))

    def forward(self, vis, lan):
        B, C, H, W = vis.shape
        lan = lan.transpose(1, 2)
        B, N, C = lan.shape

        Ci = lan.shape[-1]

        Qv, Kv, Vv = self.v_proj1(vis), self.v_proj2(vis), self.v_proj3(vis)

        Qt, Kt, Vt = self.t_proj1(lan), self.t_proj2(lan), self.t_proj3(lan)
        Qv = Qv.reshape(B, C, -1).transpose(1, 2)
        Av = F.softmax(Qv.matmul(Kt.transpose(1, 2)) / math.sqrt(Ci), dim=2)

        Kv = Kv.reshape(B, C, -1)
        At = F.softmax(Qt.matmul(Kv) / math.sqrt(Ci), dim=2)

        new_vis = Av.matmul(Vt)  # 这边有问题 应该是 batchsize个  100*1024  才不互相干扰

        Vv = Vv.reshape(B, C, -1).transpose(1, 2)
        new_lan = At.matmul(Vv)

        new_vis = new_vis.permute(0, 2, 1).reshape(B, C, H, W)

        new_vis = self.v_output(new_vis)
        new_lan = self.t_output(new_lan)
        return new_vis, new_lan


def build_joint_features(image_features, text_features):
    B, N, C = image_features.shape
    _, M, _ = text_features.shape
    joint_features = torch.cat((image_features, text_features), dim=1)  # .cuda()
    return joint_features, N, M  # lv  lt


def compute_manifold_distances(joint_feature, n_img, n_txt):
    # 计算最短路径
    isomap_batch = Isomap(n_neighbors=8, n_components=2)

    isomap_batch.fit(joint_feature)
    distances = isomap_batch.dist_matrix_

    img_to_txt_distances = distances[:n_img, n_img:]

    # 从文本到图像的流形距离矩阵
    txt_to_img_distances = distances[n_img:, :n_img]

    return img_to_txt_distances, txt_to_img_distances


def MinMaxNormalize(tensor):
    min_val = tensor.min()  # 全局最小值
    max_val = tensor.max()  # 全局最大值

    # 归一化到 [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor


def compute_M_image(vis_features, ir_features):
    B = vis_features.shape[0]
    joint_features, N, M = build_joint_features(vis_features, ir_features)
    joint_features_np = joint_features.detach().cpu().numpy()
    vis_to__ir_distances_batch = []
    ir_to_vis_distances_batch = []

    for batch_idx in range(B):  # 200 1024

        vis_to_ir_distances, ir_to_vis_distances = compute_manifold_distances(
            joint_features_np[batch_idx], N, M
        )
        vis_to__ir_distances_batch.append(
            MinMaxNormalize(torch.tensor(vis_to_ir_distances))
        )
        ir_to_vis_distances_batch.append(
            MinMaxNormalize(torch.tensor(ir_to_vis_distances))
        )

    vis_to__ir_distances_batch = torch.stack(
        vis_to__ir_distances_batch, axis=0
    ).cuda()  # 从图像 到文本的 距离 n m lv lt

    return vis_to__ir_distances_batch


def compute_M_text(image_features, text_features):
    B = image_features.shape[0]
    joint_features, N, M = build_joint_features(image_features, text_features)
    joint_features_np = joint_features.detach().cpu().numpy()
    image_to_text_distances_batch = []
    text_to_image_distances_batch = []

    for batch_idx in range(B):  # 200 1024

        img_to_txt_distances, txt_to_img_distances = compute_manifold_distances(
            joint_features_np[batch_idx], N, M
        )
        image_to_text_distances_batch.append(
            MinMaxNormalize(torch.tensor(img_to_txt_distances))
        )
        text_to_image_distances_batch.append(
            MinMaxNormalize(torch.tensor(txt_to_img_distances))
        )
        # print(img_to_txt_distances[0][0]==txt_to_img_distances[0][0])
    image_to_text_distances_batch = torch.stack(
        image_to_text_distances_batch, axis=0
    ).cuda()  # 从图像 到文本的 距离 n m lv lt
    text_to_image_distances_batch = torch.stack(
        text_to_image_distances_batch, axis=0
    ).cuda()  # 从文本 到图像的 距离 m n lt lv

    return text_to_image_distances_batch, image_to_text_distances_batch  # lt lv  lv lt


class FusionWeightGeneratorWithGAP(nn.Module):
    def __init__(
        self,
    ):
        super(FusionWeightGeneratorWithGAP, self).__init__()
        # 生成 α 和 β 的线性层，输入为 GAP 的输出
        self.alpha_proj = nn.Linear(1, 1)  # 生成 α
        self.beta_proj = nn.Linear(1, 1)  # 生成 β

    def forward(self, M):
        # M 的形状是 (b, m, n)，对最后两个维度进行全局平均池化 (GAP)
        M_gap = M.mean(dim=(1, 2), keepdim=True)  # (b, input_dim) -> 维度为 (b, m)
        # 通过线性层生成 α 和 β
        alpha = torch.sigmoid(self.alpha_proj(M_gap))  # (b, 1)
        beta = torch.sigmoid(self.beta_proj(M_gap))  # (b, 1)

        return alpha, beta


class mainfold_attention(nn.Module):
    def __init__(self, vis_chans, ir_chans, lan_chans, m_chans=None) -> None:
        super().__init__()
        if m_chans is None:
            m_chans = vis_chans
       
        self.vis_proj1 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )
        self.vis_proj2 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )
        self.vis_proj3 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )

        # self.ir_reshape = nn.Sequential(
        #     nn.Conv2d(ir_chans, m_chans, 1),
        #     nn.InstanceNorm2d(m_chans, affine=True),
        # )
        self.ir_proj1 = nn.Sequential(
            nn.Conv2d(ir_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )
        self.ir_proj2 = nn.Sequential(
            nn.Conv2d(ir_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )
        self.ir_proj3 = nn.Sequential(
            nn.Conv2d(ir_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True),
        )

        self.t_proj1 = nn.Sequential(
            nn.Linear(lan_chans, m_chans), nn.ReLU(inplace=True)
        )
        self.t_proj2 = nn.Sequential(
            nn.Linear(lan_chans, m_chans), nn.ReLU(inplace=True)
        )
        self.t_proj3 = nn.Sequential(
            nn.Linear(lan_chans, m_chans), nn.ReLU(inplace=True)
        )

        self.v_output = nn.Sequential(
            nn.Conv2d(m_chans, vis_chans, 1),
            # nn.InstanceNorm2d(vis_chans, affine=False, eps=1e-2)
        )

        self.t_output = nn.Sequential(nn.Linear(m_chans, lan_chans))

        self.W1 = nn.Parameter(torch.Tensor(100, 100))  # W 的形状与 M 一致
        self.b1 = nn.Parameter(torch.Tensor(100, 100))  # b 的形状与 M 一致
        self.lambda_param1 = nn.Parameter(
            torch.tensor(0.5, requires_grad=True)
        )  # 初始化为 0.5

        self.W2 = nn.Parameter(torch.Tensor(100, 20))  # W 的形状与 M 一致
        self.b2 = nn.Parameter(torch.Tensor(100, 20))  # b 的形状与 M 一致
        self.lambda_param2 = nn.Parameter(
            torch.tensor(0.5, requires_grad=True)
        )  # 初始化为 0.5
        self.weight_gen = FusionWeightGeneratorWithGAP()

        self.W4 = nn.Parameter(torch.Tensor(20, 100))
        self.alpha_1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.beta_1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.gamma_1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        self.W5 = nn.Parameter(torch.Tensor(100, 20))
        self.alpha_2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.beta_2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.gamma_2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        self.m_linear = nn.Linear(m_chans, lan_chans)

        self.m_linear2 = nn.Linear(m_chans, lan_chans)

        nn.init.xavier_uniform_(self.W1)
        nn.init.zeros_(self.b1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b2)

        nn.init.xavier_uniform_(self.W4)
        nn.init.xavier_uniform_(self.W5)

    def forward(self, ir, vis, lan):
        B, C, H, W = vis.shape
        vis_reshape = vis.reshape(B, C, -1).transpose(
            1, 2
        )  # 把张量经过卷积后展平，再转置
        ir_reshape = ir.reshape(B, C, -1).transpose(1, 2)
        lan = lan.transpose(1, 2)
        B, N, C = lan.shape

        M_vis_to_ir_fix_distance = compute_M_image(
            vis_reshape, ir_reshape
        )  # 计算两个视觉特征之间的流形距离矩阵 b lv lv
        sigma_initial = M_vis_to_ir_fix_distance.median().item()
        M_vis_to_ir_fix = torch.exp(
            -(M_vis_to_ir_fix_distance**2) / (2 * sigma_initial**2)
        ).cuda()

        M_vis_to_ir_learn_temp = self.W1 * M_vis_to_ir_fix + self.b1
        lambda_value = torch.sigmoid(self.lambda_param1)  # 限制 lambda 在 [0, 1] 之间

        M_vis_to_ir_learn = (
            lambda_value * M_vis_to_ir_fix + (1 - lambda_value) * M_vis_to_ir_learn_temp
        )

        Ci = lan.shape[-1]  # v->t   lt lv * lv d'

        Q_vis, K_vis, V_vis = (
            self.vis_proj1(vis),
            self.vis_proj2(vis),
            self.vis_proj3(vis),
        )
        Q_vis = Q_vis.reshape(B, C, -1).transpose(1, 2)
        K_vis = K_vis.reshape(B, C, -1).transpose(1, 2)
        V_vis = V_vis.reshape(B, C, -1).transpose(1, 2)

        Q_ir, K_ir, V_ir = self.ir_proj1(ir), self.ir_proj2(ir), self.ir_proj3(ir)
        Q_ir = Q_ir.reshape(B, C, -1).transpose(1, 2)
        K_ir = K_ir.reshape(B, C, -1).transpose(1, 2)
        V_ir = V_ir.reshape(B, C, -1).transpose(1, 2)

        Q_t, K_t, V_t = (
            self.t_proj1(lan),
            self.t_proj2(lan),
            self.t_proj3(lan),
        )  # 这里lan b 1 d    b不会干扰

        alpha, beta = self.weight_gen(M_vis_to_ir_learn)
        Q_f = alpha * Q_vis + beta * Q_ir  # 3 数值正常
        K_f = alpha * K_vis + beta * K_ir
        V_f = (
            alpha * V_vis + beta * V_ir
        )  # (Q_f.matmul(K_f.transpose(1, 2))).matmul(M_vis_to_ir_learn).matmul((V_vis + V_ir)/2)
        feature_image_reshape = alpha * (
            vis.reshape(B, C, -1).transpose(1, 2)
        ) + beta * (
            ir.reshape(B, C, -1).transpose(1, 2)
        )  # b 100 c
        # alpha vis  beta ir

        M_text_to_image_fixdistance, M_image_to_text_fixdistance = compute_M_text(
            feature_image_reshape, lan
        )  # 计算视觉特征与文本特征之间的流形距离矩阵 b lv lv
        #          lt lv                lv lt
        sigma_initial = M_text_to_image_fixdistance.median().item()
        M_text_to_image_fix = torch.exp(
            -(M_text_to_image_fixdistance**2) / (2 * sigma_initial**2)
        ).cuda()  # similarity lt lv

        M_image_to_text_fix = M_text_to_image_fix.transpose(1, 2)  # lv lt

        M_image_to_text_learn_temp = self.W2 * M_image_to_text_fix + self.b2
        lambda_value2 = torch.sigmoid(self.lambda_param2)  # 限制 lambda 在 [0, 1] 之间
        M_image_to_text_learn_origin = (
            lambda_value2 * M_image_to_text_fix
            + (1 - lambda_value2) * M_image_to_text_learn_temp
        )  # lv lt

        M_text_to_image_learn_origin = M_image_to_text_learn_origin.transpose(
            1, 2
        )  # lt lv

        M_text_to_image_learn_origin = F.softmax(
            M_text_to_image_learn_origin / math.sqrt(Ci), dim=2
        )  # 归一化 lt lv
        M_image_to_text_learn_origin = F.softmax(
            M_image_to_text_learn_origin / math.sqrt(Ci), dim=2
        )  # 归一化 lv lt

        M_text_to_image_learn = self.m_linear(
            M_image_to_text_learn_origin.matmul(lan)
        )  # lv lt    lt d' ->   lv d'  对文本向量进行加权
        M_image_to_text_learn = self.m_linear2(
            M_text_to_image_learn_origin.matmul(feature_image_reshape)
        )  #   lt d'   对视觉特征加权

        # lv d'
        # text 查询 V
        A0 = Q_t.matmul(K_f.transpose(1, 2))  # lt lv
        # print(A0.shape)
        A1 = Q_t.matmul(
            M_text_to_image_learn.transpose(1, 2)
        )  #    lt d'       (lt lv    lv d')
        # print(A1.shape)
        A2 = self.W4.matmul(
            M_text_to_image_learn.matmul(K_f.transpose(1, 2))
        )  # lt lv lv d'
        # print(A2.shape)
        A = A0 + self.alpha_1 * A1 + self.beta_1 * A2
        Av = F.softmax(A / math.sqrt(Ci), dim=2)  # lt lv

        new_lan = Av.matmul(V_f) + self.gamma_1 * (
            M_text_to_image_learn_origin.matmul(V_f)
        )  # lt d' 本质上是v

        B0 = Q_f.matmul(K_t.transpose(1, 2))  # lv lt
        B1 = Q_f.matmul(M_image_to_text_learn.transpose(1, 2))  # lv lt
        B2 = self.W5.matmul(M_image_to_text_learn.matmul(K_t.transpose(1, 2)))  # lv lt
        B_ = B0 + self.alpha_2 * B1 + self.beta_2 * B2
        Bv = F.softmax(B_ / math.sqrt(Ci), dim=2)  # lv lt

        new_vis = Bv.matmul(V_t) + self.gamma_2 * (
            M_image_to_text_learn_origin.matmul(V_t)
        )  # lv d' # 本质上是t

        new_vis = new_vis.permute(0, 2, 1).reshape(B, C, H, W)

        new_vis = self.v_output(new_vis)
        new_lan = self.t_output(new_lan)

        f_M_text_to_image_fix_softmax = F.softmax(
            M_text_to_image_fix / math.sqrt(Ci), dim=2
        )  # 归一化 lt lv
        f_M_image_to_text_fix_softmax = F.softmax(
            M_image_to_text_fix / math.sqrt(Ci), dim=2
        )  # 归一化 lv lt

        regulation_loss_1 = F.l1_loss(Av, f_M_text_to_image_fix_softmax) + F.l1_loss(
            Bv, f_M_image_to_text_fix_softmax
        )
        regulation_loss_2 = -F.l1_loss(
            Av.var(), f_M_text_to_image_fix_softmax.var()
        ) - F.l1_loss(Bv.var(), f_M_image_to_text_fix_softmax.var())
        regulation_loss = (regulation_loss_1 + 1e2 * regulation_loss_2) * 1e3

        return new_vis, new_lan, regulation_loss


if __name__ == "__main__":
    model = mainfold_attention(1024, 1024, 1024).cuda()
    a = torch.randn(1, 1024, 10, 10).cuda()
    b = torch.randn(1, 1024, 10, 10).cuda()
    c = torch.randn(1, 1024, 20).cuda()
    d = model(a, b, c)
    print("SUCCESS")
