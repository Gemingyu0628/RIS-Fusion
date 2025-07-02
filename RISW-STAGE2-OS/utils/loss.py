
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
# import kornia
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class BCE_loss():
    def __init__(self):
        super().__init__()
    def cal(self,predictlabel, truelabel):
        validindex = torch.where(torch.sum(truelabel, axis=2) == 1)  # 人工拾取了的道
        criteria = nn.BCELoss()
        loss = criteria(predictlabel[validindex[0], validindex[1], :, validindex[2]],
                        truelabel[validindex[0], validindex[1], :, validindex[2]])
        return loss



class MSE_loss():
    def __init__(self):
        super().__init__()
    def cal(self,predictlabel, truelabel):
        validindex = torch.where(torch.sum(truelabel, axis=2) == 1)  # 人工拾取了的道
        valid_predictlabel=predictlabel[validindex[0], validindex[1], :, validindex[2]]
        valid_truelabel = truelabel[validindex[0], validindex[1], :, validindex[2]]
        label_index=torch.argmax(valid_truelabel,dim=1)/valid_truelabel.shape[1]
        predict_index=torch.argmax(valid_predictlabel,dim=1)/valid_truelabel.shape[1]
        criteria = nn.MSELoss()
        loss = criteria(label_index,predict_index)
        return loss

class BCE_MSE_loss():
    def __init__(self,balan_para):
        super().__init__()
        self.bp=balan_para
        self.BCEloss=BCE_loss()
        self.MSEloss=MSE_loss()
    def cal(self, predictlabel, truelabel):
        loss =self.BCEloss.cal(predictlabel,truelabel)+ self.bp*self.MSEloss.cal(predictlabel,truelabel)
        return loss

def CE_Loss(inputs, target, num_classes=0):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss( ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

class Fusionloss(nn.Module):
    def __init__(self,coeff_int=1,coeff_grad=10,in_max=True, device='cuda'):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device=device)
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.in_max=in_max
    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        if self.in_max:
            x_in_max=torch.max(image_y,image_ir)
        else:
            x_in_max=(image_y+image_ir)/2.0
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=self.coeff_int*loss_in+self.coeff_grad*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self,device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2,
                         groups=C).view(C, H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0,
                          patch_index.view(1, -1)).view(C, H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) /
                       denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q



class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


class LpLssimLossweight(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)   # [window_size, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, image_in, image_out, weight):

        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device())
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel

        # Lp
        Lp = torch.sqrt(torch.sum(torch.pow((image_in - image_out), 2)))  # 二范数
        # Lp = torch.sum(torch.abs(image_in - image_out))  # 一范数
        # Lssim
        Lssim = 1 - self._ssim(image_in, image_out, self.window, self.window_size, self.channel, self.size_average)
        return Lp + Lssim * weight, Lp, Lssim * weight

class OrthoLoss(nn.Module):

    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2 = input1
        input2_l2 = input2
        
        ortho_loss = 0
        dim = input1.shape[1]
        for i in range(input1.shape[0]):
            ortho_loss += torch.mean(((input1_l2[i:i+1,:].mm(input2_l2[i:i+1,:].t())).pow(2))/dim)

        ortho_loss = ortho_loss / input1.shape[0]

        return ortho_loss

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image
    
def compute_iou_loss(pred_mask, gt_mask):
    # 将形状从 (b, 1, h, w) 扁平化为 (b, h * w)
    pred_mask_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_mask_flat = gt_mask.view(gt_mask.size(0), -1)
    
    intersection = (pred_mask_flat * gt_mask_flat).sum(dim=1)
    union = pred_mask_flat.sum(dim=1) + gt_mask_flat.sum(dim=1) - intersection + 1e-6
    iou = intersection / union
    return 1 - iou.mean()  # 返回平均 IoU 的损失

def compute_centroid_loss(pred_mask, gt_mask):

    pred = pred_mask.squeeze(1)
    gt = gt_mask.squeeze(1)
    loss_temp = torch.tensor(0.0).to(pred_mask.device)
    for i in range(pred.shape[0]):
        
        pred_coords = pred[i].nonzero(as_tuple=False) 
        gt_coords = gt[i].nonzero(as_tuple=False)

        y_center, x_center = pred_coords.float().mean(dim=0) 
        y_center_gt, x_center_gt = gt_coords.float().mean(dim=0)

        # print(y_center)
        loss_temp = loss_temp + F.mse_loss(y_center/255, y_center_gt/255) + F.mse_loss(x_center/255, x_center_gt/255)
    
    # for i in range(pred_mask.shape[0]):    
    #     pred_indices = torch.nonzero(pred_mask[i].view(-1)).float()  # 扁平化后查找非零元素
    #     gt_indices = torch.nonzero(gt_mask[i].view(-1)).float()
        
    #     if pred_indices.numel() == 0 or gt_indices.numel() == 0:
    #         return torch.tensor(0.0).to(pred_mask.device)

    #     pred_centroid = pred_indices.mean(dim=0) / len(pred_mask[i].view(-1))
    #     gt_centroid = gt_indices.mean(dim=0) / len(pred_mask[i].view(-1))
    #     print(pred_centroid)
    #     print(gt_centroid)
    #     loss_temp = loss_temp + F.mse_loss(pred_centroid, gt_centroid)
    loss = loss_temp / pred_mask.shape[0]
    return loss

def compute_bce_loss(pred_mask, gt_mask):
    pred_probs = pred_mask # 确保概率值在0-1之间
    return F.binary_cross_entropy(pred_probs, gt_mask)

def combined_loss(pred_mask, gt_mask, lambda_iou=1.0, lambda_centroid=1.0, lambda_bce=1.0):
    iou_loss = compute_iou_loss(pred_mask, gt_mask)
    # centroid_loss = compute_centroid_loss(pred_mask, gt_mask)
    bce_loss = compute_bce_loss(pred_mask, gt_mask)
    
    total_loss = (lambda_iou * iou_loss +
                    # lambda_centroid * centroid_loss +
                  lambda_bce * bce_loss)
    
    return total_loss


def compute_spatial_frequency(image):
    """
    计算图像的空间频率（Spatial Frequency, SF）
    
    参数:
    image: 输入图像，形状为 (B, C, H, W)，其中 B 是批量大小，C 是通道数，H 是高度，W 是宽度
    
    返回:
    sf: 空间频率，形状为 (B,)
    """
    # 计算水平和垂直方向的梯度
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]  # 水平梯度
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]  # 垂直梯度
    
    # 计算梯度的幅值
    dx_norm = torch.norm(dx, p=2, dim=1)
    dy_norm = torch.norm(dy, p=2, dim=1)
    
    # 计算空间频率
    sf = torch.mean(dx_norm, dim=(1, 2)) + torch.mean(dy_norm, dim=(1, 2))
    
    return sf

def spatial_frequency_loss(output):
    """
    计算空间频率损失
    
    参数:
    output: 模型输出的图像，形状为 (B, C, H, W)
    
    返回:
    loss: 空间频率损失
    """
    sf_output = compute_spatial_frequency(output)
    
    # 使用对数变换计算空间频率损失
    loss = -torch.mean(torch.log(sf_output + 1e-8))  # 添加一个小常数以避免对数运算中的数值问题
    
    return loss

def compute_average_gradient(image):
    """
    计算图像的平均梯度（Average Gradient, AG）
    
    参数:
    image: 输入图像，形状为 (B, C, H, W)，其中 B 是批量大小，C 是通道数，H 是高度，W 是宽度
    
    返回:
    ag: 平均梯度，形状为 (B,)
    """
    # 计算水平和垂直方向的梯度
    Gx = torch.zeros_like(image)
    Gy = torch.zeros_like(image)

    Gx[:, :, :, 0] = image[:, :, :, 1] - image[:, :, :, 0]
    Gx[:, :, :, -1] = image[:, :, :, -1] - image[:, :, :, -2]
    Gx[:, :, :, 1:-1] = (image[:, :, :, 2:] - image[:, :, :, :-2]) / 2

    Gy[:, :, 0, :] = image[:, :, 1, :] - image[:, :, 0, :]
    Gy[:, :, -1, :] = image[:, :, -1, :] - image[:, :, -2, :]
    Gy[:, :, 1:-1, :] = (image[:, :, 2:, :] - image[:, :, :-2, :]) / 2

    # 计算梯度的幅值
    gradient_magnitude = torch.sqrt((Gx ** 2 + Gy ** 2) / 2)
    
    # 计算平均梯度
    ag = torch.mean(gradient_magnitude, dim=(1, 2, 3))
    
    return ag

def average_gradient_loss(output):
    """
    计算平均梯度损失
    
    参数:
    output: 模型输出的图像，形状为 (B, C, H, W)
    
    返回:
    loss: 平均梯度损失
    """
    ag_output = compute_average_gradient(output)
    
    # 使用对数变换计算平均梯度损失
    loss = -torch.mean(torch.log(ag_output + 1e-8))  # 添加一个小常数以避免对数运算中的数值问题
    
    return loss