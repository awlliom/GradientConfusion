'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import time

from pathlib import Path
from random import uniform

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        defense_config_path = Path('config-jsons/defense_config.json')
        with open(defense_config_path) as f:
            self.defense_config = json.load(f)
        
        self.defense_type = self.defense_config['defense']
        print(self.defense_type, self.defense_config)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # iRND:
        if self.defense_type == 'irnd':
            sigma = self.defense_config[self.defense_type]['sigma_noise']
            out = F.relu(self.bn1(self.conv1(torch.clip(x + sigma * torch.randn_like(x), 0, 1))))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        # RFD:
        if self.defense_type == 'rfd':
            rfd_config = self.defense_config[self.defense_type]
            noise_sigma = rfd_config['sigma_noise']
            target_layers = rfd_config['target_layers'] # [-1] means all layers

        out = self.layer1(out)
        if self.defense_type == 'rfd' and (1 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)

        out = self.layer2(out)
        if self.defense_type == 'rfd' and (2 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)

        out = self.layer3(out)
        if self.defense_type == 'rfd' and (3 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)

        out = self.layer4(out)
        if self.defense_type == 'rfd' and (4 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

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
            
            out = m * out
        # oRND
        elif self.defense_type == 'ornd':
            noise_sigma = self.defense_config[self.defense_type]['sigma_noise']
            out = out + noise_sigma * torch.randn_like(out)

        # AAA
        elif self.defense_type == "aaa":
            self.temperature = self.defense_config['aaa']['temperature']
            self.dev = self.defense_config['aaa']['dev']
            self.attractor_interval = self.defense_config['aaa']['attractor_interval']
            self.calibration_loss_weight = self.defense_config['aaa']['calibration_loss_weight']
            self.optimizer_lr = self.defense_config['aaa']['optimizer_lr']
            self.num_iter = self.defense_config['aaa']['num_iter']

            logits = out
            logits_ori = logits.detach()    
            
            prob_ori = F.softmax(logits_ori / self.temperature, dim=1) # prob_ori: int[n_samples, n_classes]
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
                    prob = F.softmax(logits, dim=1)
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
        
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
