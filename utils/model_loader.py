
from __future__ import division
from __future__ import absolute_import

import torch
import os
import json
import torch.nn as nn
from models import resnet
from utils.misc import data_path_join
from models import resnet, vgg, wideresnet

import torchvision
import numpy as np
from torchvision import models, datasets
from pathlib import Path
from random import uniform
########################################################################
import gdown
import copy
import time
import torchvision.transforms as transforms

defense_config_path = Path('config-jsons/defense_config.json')

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std).cuda()
########################################################################


def load_torch_models(model_name):
    pretrained_path = Path(__file__).parent.parent / Path("data/pretrained_models")
    pretrained_path.mkdir(parents=True, exist_ok=True)  # make sure dir exists

    if model_name == 'resnet':
        file_id = '1yvfAGyOiUkN0hLlEFKogkor6AJSRcZ74'
        url = f'https://drive.google.com/uc?id={file_id}'
        output = pretrained_path / "Cifar10ResNet18WithoutNormalization_95.53.pth"

        # Only download if missing
        if not output.exists():
            gdown.download(url, str(output), quiet=False)

        pretrained_model = resnet.ResNet18()
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        checkpoint = torch.load(output)
        pretrained_model.load_state_dict(checkpoint['net'])

    elif model_name == 'wrn':
        file_id = '1AVI-k5foOxz78XWKYcIGL8ToXqUsHsPT'
        url = f'https://drive.google.com/uc?id={file_id}'
        output = pretrained_path / "WRN28-10_100Epochs_UNormalized_96.11Acc.tar"

        if not output.exists():
            gdown.download(url, str(output), quiet=False)

        pretrained_model = wideresnet.WideResNet(28, 10, 10, dropRate=0)
        checkpoint = torch.load(output)
        pretrained_model.load_state_dict(checkpoint['state_dict'])

    elif model_name == 'vgg':
        file_id = '1QfbiTT0pweJ303BqudBMYa0lsVJzKCNZ'
        url = f'https://drive.google.com/uc?id={file_id}'
        output = pretrained_path / "VGG16_300Epochs_UNormalized_91.80Acc.tar"

        if not output.exists():
            gdown.download(url, str(output), quiet=False)

        pretrained_model = vgg.__dict__["vgg16"]()
        checkpoint = torch.load(output)
        pretrained_model.features = torch.nn.DataParallel(pretrained_model.features)
        pretrained_model.load_state_dict(checkpoint['state_dict'])

    if 'wrn' in model_name:
        net = nn.Sequential(
            pretrained_model
        )

    elif "vgg" in model_name:
        net = pretrained_model
    else:
        net = nn.Sequential(
            pretrained_model
        )

    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    return net

class Permute(nn.Module):

    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):

        return input[:, self.permutation]

def load_torch_models_imagesub(model_name):
    if model_name == 'Resnet50':
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        output_layer = pretrained_model.fc
        blocks = [pretrained_model.layer1, pretrained_model.layer2, pretrained_model.layer3, pretrained_model.layer4]

        defense_config_path = Path('config-jsons/defense_config.json')
        with open(defense_config_path) as f:
            defense_config = json.load(f)
        defense_type = defense_config['defense']

        if defense_type == 'ornd':
            noise_sigma = defense_config[defense_type]['sigma_noise']
            def ornd_defense(sigma):
                def ornd_defense_hook(module, inp, out):
                    if isinstance(out, tuple):
                        activation = out[0]
                    else:
                        activation = out
                    
                    activation = activation + sigma * torch.randn_like(activation)

                    if isinstance(out, tuple):
                        return (activation, *out[1:])
                    else:
                        return activation
                    
                return ornd_defense_hook
            
            hook = ornd_defense(sigma=noise_sigma)
            output_layer.register_forward_hook(hook)

        elif defense_type == 'rls':
            rls_config = defense_config[defense_type]
            if rls_config['distribution'] == 'uniform':
                m_low = rls_config['uniform']['low']
                m_high = rls_config['uniform']['high']
                def m_sampled():
                    m = uniform(m_low, m_high)
                    return m
            elif rls_config['distribution'] == 'gaussian':
                min_temp = rls_config['gaussian']['low']
                mean = rls_config['gaussian']['mean']
                std = rls_config['gaussian']['std']
                def m_sampled():
                    m = torch.normal(mean=mean, std=torch.tensor(float(std)))
                    
                    if m < 0:
                        m *= -1
                    if m < min_temp:
                        m = min_temp
                    
                    return m

            def rls_defense(m_sampled):
                def rls_hook_defense(module, inp, out):
                    if isinstance(out, tuple):
                        activation = out[0]
                    else:
                        activation = out

                    m = m_sampled()
                    activation = m * activation

                    if isinstance(out, tuple):
                        return (activation, *out[1:])
                    else:
                        return activation
                return rls_hook_defense
            
            hook = rls_defense(m_sampled)
            output_layer.register_forward_hook(hook)

        elif defense_type == 'rfd':
            rfd_config = defense_config[defense_type]
            noise_sigma = rfd_config['sigma_noise']
            target_layers = rfd_config['target_layers'] # [-1] means all layers
        
            def rfd_defense(sigma):
                def rfd_defense_hook(module, inp, out):
                    if isinstance(out, tuple):
                        activation = out[0]
                    else:
                        activation = out

                    activation = activation + sigma * torch.randn_like(activation)

                    if isinstance(out, tuple):
                        return (activation, *out[1:])
                    else:
                        return activation
                return rfd_defense_hook
            
            for i, l in enumerate(blocks):
                if (i+1) in target_layers or target_layers == [-1]: # [-1] means all layers
                    hook = rfd_defense(sigma=noise_sigma)
                    l.register_forward_hook(hook)
        
        elif defense_type == 'aaa':
            temperature = defense_config['aaa']['temperature']
            dev = defense_config['aaa']['dev']
            attractor_interval = defense_config['aaa']['attractor_interval']
            calibration_loss_weight = defense_config['aaa']['calibration_loss_weight']
            optimizer_lr = defense_config['aaa']['optimizer_lr']
            num_iter = defense_config['aaa']['num_iter']

            def aaa_defense(temperature, dev, attractor_interval, calibration_loss_weight, optimizer_lr, num_iter, times):
                def aaa_defense_hook(module, inp, out):
                    if isinstance(out, tuple):
                        activation = out[0]
                    else:
                        activation = out

                    #logits = activation
                    logits_ori = activation.detach()    
                    prob_ori = nn.functional.softmax(logits_ori / temperature, dim=1) # prob_ori: int[n_samples, n_classes]
                    prob_max_ori = prob_ori.max(1)[0] ### largest probability for each sample
                    value, index_ori = torch.topk(logits_ori, k=2, dim=1)
                    #"""
                    mask_first = torch.zeros(activation.shape, device=torch.device('cuda'))
                    mask_first[torch.arange(activation.shape[0]), index_ori[:, 0]] = 1 # Masks all the elements except the largest one
                    mask_second = torch.zeros(activation.shape, device=torch.device('cuda'))
                    mask_second[torch.arange(activation.shape[0]), index_ori[:, 1]] = 1 # Masks all the elements except the second largest one
                    #"""
                    
                    margin_ori = value[:, 0] - value[:, 1]
                    # sine attractor:
                    attractor = ((margin_ori / attractor_interval + dev).round() - dev) * attractor_interval
                    # linear attractor:
                    #target = attractor - self.reverse_step * (margin_ori - attractor)

                    target = margin_ori - 0.7 * attractor_interval * torch.sin(
                        (1 - 2 / attractor_interval * (margin_ori - attractor)) * torch.pi)
                    diff_ori = (margin_ori - target)
                    real_diff_ori = margin_ori - attractor

                    with torch.enable_grad():
                        activation.requires_grad = True
                        optimizer = torch.optim.Adam([activation], lr=optimizer_lr)
                        los_reverse_rate = 0
                        prd_maintain_rate = 0
                        for i in range(num_iter):
                        #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
                            prob = nn.functional.softmax(activation, dim=1)
                            #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
                            loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                            #loss_calibration = (prob - prob_ori).abs().mean()

                            value, index = torch.topk(activation, k=2, dim=1)
                            margin = value[:, 0] - value[:, 1]
                            #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

                            diff = (margin - target)
                            real_diff = margin - attractor
                            loss_defense = diff.abs().mean()
                            
                            loss = loss_defense + loss_calibration * calibration_loss_weight
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
                            prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
                            #print('%d, %.2f, %.2f' % (i, los_reverse_rate * 100, prd_maintain_rate * 100), end='\r')
                            #print('%d, %.4f, %.4f, %.4f' % (itre, loss_calibration, loss_defense, loss))

                    if isinstance(out, tuple):
                        return (activation, *out[1:])
                    else:
                        return activation
                
                return aaa_defense_hook
            
            hook = aaa_defense(temperature, dev, attractor_interval, calibration_loss_weight, optimizer_lr, num_iter, 5)
            output_layer.register_forward_hook(hook)
        
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = NormalizeByChannelMeanStd(
        mean=mean.tolist(), std=std.tolist())

    net = nn.Sequential(
        normalize,
        pretrained_model
    )

    if defense_type == 'irnd':
        noise_sigma = defense_config[defense_type]['sigma_noise']
        def irnd_defense(sigma):
            def irnd_defense_hook(module, inp):
                if isinstance(inp, tuple):
                    activation = inp[0]
                else:
                    activation = inp

                activation = torch.clip(activation + sigma * torch.randn_like(activation), 0, 1)

                if isinstance(inp, tuple):
                    return (activation, *inp[1:])
                else:
                    return activation
            return irnd_defense_hook
        
        hook = irnd_defense(sigma=noise_sigma)
        net.register_forward_pre_hook(hook)

    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    net = torch.nn.DataParallel(net)
    return net
