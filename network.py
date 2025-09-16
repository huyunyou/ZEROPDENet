import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils import *
from loss import *


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        if not self.is_last: 
            self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)


class INF(nn.Module):
    def __init__(self, patch_dim, hidden_dim, weight_decay=None):
        super().__init__()

        patch_layers = [SirenLayer(patch_dim, hidden_dim, is_first=True)]
        spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]
        output_layers = []

        patch_layers.append(SirenLayer(hidden_dim, hidden_dim//2))         #new layer
        patch_layers.append(SirenLayer(hidden_dim//2, hidden_dim//4))
        spatial_layers.append(SirenLayer(hidden_dim, hidden_dim//2))       #new layer
        spatial_layers.append(SirenLayer(hidden_dim//2, hidden_dim//4))

        output_layers.append(SirenLayer(hidden_dim//2, hidden_dim//2))
        output_layers.append(SirenLayer(hidden_dim//2, 1, is_last=True))

        self.patch_net = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params = []
        self.params += [{'params':self.spatial_net.parameters(),'weight_decay':weight_decay[0]}]
        self.params += [{'params':self.patch_net.parameters(),'weight_decay':weight_decay[1]}]
        self.params += [{'params':self.output_net.parameters(),'weight_decay':weight_decay[2]}]

    def forward(self, patch, spatial):
        return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))


class DenoiseNet(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(DenoiseNet, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.norm = nn.BatchNorm2d(chan_embed)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(2*chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(2*chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        #self.dropout = nn.Dropout(0.20)
        self._initialize_weights()

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        #x2 = self.dropout(x2)
        x4 = self.act(self.conv4(x2))
        #x4 = self.dropout(x4)
        x5 = self.act(self.conv5(torch.cat([x2,x4],1)))
        x6 = self.act(self.conv6(torch.cat([x1,x5],1)))
        x = self.conv3(x6)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                #m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.denoiseNet = DenoiseNet(n_chan=1)
        self.siren = INF(patch_dim=1*(7**2), hidden_dim=256)
        self.loss_f2 = nn.MSELoss()
        self.l_TV = L_TV()

    def forward(self, input, img1, img2):
        eps = 1e-5
        # denoise learning
        pred1 = self.denoiseNet(img1)
        pred2 = self.denoiseNet(img2)
        loss1 = 1.0*self.loss_f2(img2,pred1) + 1.0*self.loss_f2(img1,pred2) +  1.5*self.loss_f2(pred1,pred2)
        pred = self.denoiseNet(input)
        pred = torch.clamp(pred, eps, 1)

        # reshape to 256×256
        DOLP_L = interpolate_image(pred.detach(), 256, 256)       # I

        coords = get_coords(256, 256)
        patches = get_patches(DOLP_L, 7)

        psi_theta = self.siren(patches, coords)
        psi_theta = psi_theta.view(1, 1, 256, 256)
        func_K = psi_theta + DOLP_L        # s

        DOLP_H = (DOLP_L) / (func_K + 1e-4)  #I/S

        '***********start**********'
        DOLP_L_mean = torch.mean(DOLP_L.detach(), dim=(1, 2))
        radiance_coefficient = 0.2/ (DOLP_L_mean + eps)          # 0.2（最终的），0.3，0.4，0.5
        radiance_coefficient = radiance_coefficient.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        radiance_coefficient = torch.clamp(radiance_coefficient, 1, 25)
        scaling_factor = torch.pow(0.7, -radiance_coefficient) / radiance_coefficient     # 0.7
        normalized_layer  = DOLP_L.detach() / func_K
        normalized_layer = torch.clamp(normalized_layer, eps, 1)
        enhanced_gray=torch.pow(DOLP_L.detach()*radiance_coefficient, radiance_coefficient)
        clamped_enhanced_gray = torch.clamp(enhanced_gray * scaling_factor, eps, 1)
        clamped_adjusted_light  = torch.clamp(DOLP_L.detach() *  radiance_coefficient,eps,1)

        loss_exp = 0.25*self.loss_f2(func_K, clamped_enhanced_gray) + 1.0*self.loss_f2(normalized_layer, clamped_adjusted_light)
        '***********end************'

        loss_spa = torch.mean(torch.abs(torch.pow(func_K - DOLP_L, 2)))
        loss_tv = self.l_TV(func_K)
        loss_sparsity = torch.mean(torch.abs(DOLP_H))

        DOLP_H_reshape = filter_up(DOLP_L, DOLP_H, pred)

        loss =  loss1 * 10.0 + loss_spa * 1.0 + loss_tv * 100.0 + loss_exp * 5.0 + loss_sparsity * 5.0

        return pred,DOLP_H_reshape,loss
