'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import json
import time

import torch
import torch.nn as nn
import torch.nn.init as init

from pathlib import Path
from random import uniform

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# noise = 0.35

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        defense_config_path = Path('config-jsons/defense_config.json')
        with open(defense_config_path) as f:
            self.defense_config = json.load(f)
        
        self.defense_type = self.defense_config['defense']
        

    def forward(self, x):
        # iRND:
        if self.defense_type == 'irnd':
            sigma = self.defense_config[self.defense_type]['sigma_noise']
            x = torch.clip(x + sigma * torch.randn_like(x), 0, 1)

        if self.defense_type == 'rfd':
            rfd_config = self.defense_config[self.defense_type]
            noise_sigma = rfd_config['sigma_noise']
            target_layers = rfd_config['target_layers'] # [-1] means all layers
            
        for i, l in enumerate(self.features.module):
            # RFD:
            if self.defense_type == 'rfd':
                if 'conv' in l._get_name().lower() and (i in target_layers or i == [-1]): # [-1] means all layers
                    l.cuda()
                    x = l(x)
                    x = x + noise_sigma * torch.randn_like(x)
                else:
                    l.cuda()
                    x = l(x)
            else:
                l.cuda()
                x = l(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # RLS:
        if self.defense_type == 'rls':
            rls_config = self.defense_config[self.defense_type]
            if rls_config['distribution'] == 'uniform':
                m_low = rls_config['uniform']['low']
                m_high = rls_config['uniform']['high']
                m = uniform(m_low, m_high)
            elif rls_config['distribution'] == 'gaussian':
                min_temp = rls_config['gaussian']['low']
                mean = rls_config['gaussian']['mean']
                std = rls_config['gaussian']['std']
                m = torch.normal(mean=mean, std=torch.tensor(float(std)))
                
                if m < 0:
                    m *= -1
                if m < min_temp:
                    m = min_temp
            
            x = m * x
        # oRND
        elif self.defense_type == 'ornd':
            noise_sigma = self.defense_config[self.defense_type]['sigma_noise']
            x = x + noise_sigma * torch.randn_like(x)

        # AAA
        elif self.defense_type == "aaa":
            self.temperature = self.defense_config['aaa']['temperature']
            self.dev = self.defense_config['aaa']['dev']
            self.attractor_interval = self.defense_config['aaa']['attractor_interval']
            self.calibration_loss_weight = self.defense_config['aaa']['calibration_loss_weight']
            self.optimizer_lr = self.defense_config['aaa']['optimizer_lr']
            self.num_iter = self.defense_config['aaa']['num_iter']

            logits = x
            logits_ori = logits.detach()    
            prob_ori = nn.functional.softmax(logits_ori / self.temperature, dim=1) # prob_ori: int[n_samples, n_classes]
            prob_max_ori = prob_ori.max(1)[0] ### largest probability for each sample
            value, index_ori = torch.topk(logits_ori, k=2, dim=1)
            #"""
            mask_first = torch.zeros(logits.shape, device=torch.device('cuda'))
            mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1 # Masks all the elements except the largest one
            mask_second = torch.zeros(logits.shape, device=torch.device('cuda'))
            mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1 # Masks all the elements except the second largest one
            #"""
            
            margin_ori = value[:, 0] - value[:, 1]
            # sine attractor:
            attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
            # linear attractor:
            #target = attractor - self.reverse_step * (margin_ori - attractor)

            target = margin_ori - 0.7 * self.attractor_interval * torch.sin(
                (1 - 2 / self.attractor_interval * (margin_ori - attractor)) * torch.pi)
            diff_ori = (margin_ori - target)
            real_diff_ori = margin_ori - attractor

            with torch.enable_grad():
                logits.requires_grad = True
                optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
                los_reverse_rate = 0
                prd_maintain_rate = 0
                for i in range(self.num_iter):
                #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
                    prob = nn.functional.softmax(logits, dim=1)
                    #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
                    loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                    #loss_calibration = (prob - prob_ori).abs().mean()

                    value, index = torch.topk(logits, k=2, dim=1)
                    margin = value[:, 0] - value[:, 1]
                    #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

                    diff = (margin - target)
                    real_diff = margin - attractor
                    loss_defense = diff.abs().mean()
                    
                    loss = loss_defense + loss_calibration * self.calibration_loss_weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
                    prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
                    #print('%d, %.2f, %.2f' % (i, los_reverse_rate * 100, prd_maintain_rate * 100), end='\r')
                    #print('%d, %.4f, %.4f, %.4f' % (itre, loss_calibration, loss_defense, loss))
            return logits

        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
