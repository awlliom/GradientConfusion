import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import time

from pathlib import Path
from random import uniform

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None


    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
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

        out = self.conv1(x)
        out = self.block1(out)
        if self.defense_type == 'rfd' and (1 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)

        out = self.block2(out)
        if self.defense_type == 'rfd' and (2 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)

        out = self.block3(out)
        if self.defense_type == 'rfd' and (3 in target_layers or target_layers == [-1]): # [-1] means all layers
            out = out + noise_sigma * torch.randn_like(out)
        
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)

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
