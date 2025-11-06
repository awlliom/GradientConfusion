from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor as t
import cv2
import pandas as pd

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack


class ModelWrapper:
    """Wrapper to handle normalization like the original PretrainedModel"""
    def __init__(self, model, dataset='cifar10'):
        self.model = model
        self.dataset = dataset
        
        # Set normalization parameters based on dataset
        if self.dataset == 'cifar10':
            self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(1, 3, 1, 1)
            self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(1, 3, 1, 1)
        elif self.dataset == 'imagenet':
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1)
        else:
            # Default: no normalization
            self.mu = torch.zeros(1, 3, 1, 1)
            self.sigma = torch.ones(1, 3, 1, 1)
    
    def to(self, device):
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        return self
    
    def predict(self, x):
        """x in (N, C, H, W) format, range [0, 1]"""
        img = (x - self.mu) / self.sigma
        return self.model(img)
    
    def predict_label(self, x):
        out = self.predict(x)
        return torch.argmax(out, dim=1)
    
    def __call__(self, x):
        return self.predict(x)


class BruSLiAttack(ScoreBlackBoxAttack):
    """
    BruSLi Attack (Boundary-based Random Sparse Linear Attack)
    
    This is a sparse L0 attack based on the paper:
    "BruSLeAttack: A Query-Efficient Score-Based Black-Box Sparse Adversarial Attack"
    
    Optimized implementation compatible with the framework's iterative structure.
    """

    def __init__(self, max_loss_queries, epsilon, p, lb, ub, batch_size, name,
                 n_pix=4, pop_size=10, lamda=0.01, m1=0.24, m2=0.997,
                 seed=None, flag=False, ftype='ce', init_mode='uni', init_scale=1,
                 dataset='cifar10'):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation (not used for L0 attack, kept for compatibility)
        :param p: specifies lp-norm of perturbation (should be '0' or 'inf' for this attack)
        :param lb: data lower bound
        :param ub: data upper bound
        :param batch_size: batch size for attack
        :param n_pix: number of pixels to perturb initially
        :param pop_size: population size for evolutionary algorithm
        :param lamda: initial step decay parameter
        :param m1: power decay exponent
        :param m2: exponential decay factor
        :param seed: random seed
        :param flag: True for targeted attack, False for untargeted
        :param ftype: fitness type ('ce' for cross-entropy, 'margin' for margin)
        :param init_mode: initialization mode for target image ('uni', 'custom', 'inverted', etc.)
        :param init_scale: scale factor for target image initialization
        :param dataset: dataset name for proper normalization ('cifar10' or 'imagenet')
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size=batch_size,
                         name="BruSLi")
        
        self.n_pix = n_pix
        self.pop_size = pop_size
        self.lamda = lamda
        self.m1 = m1
        self.m2 = m2
        self.seed = seed
        self.flag = flag  # True: targeted, False: untargeted
        self.ftype = ftype
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.dataset = dataset
        
        # Attack state - will be initialized per batch
        self.model_wrapper = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Per-sample state for iterative attacks
        self.batch_state = {}

    def set_model(self, model):
        """Set the model for BruSLi attack with proper normalization wrapper"""
        self.model_wrapper = ModelWrapper(model, dataset=self.dataset).to(self.device)

    def search_space_init(self, img, seed=None, scale=1, mode='uni'):
        """
        Generate starting image for attack (target image for sparse perturbation)
        Optimized version of BruSLiAttack-main/utils.py
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # img shape: (1, C, H, W) in [0, 1]
        c, h, w = img.shape[1], img.shape[2], img.shape[3]
        
        if scale > 1:
            h_scaled, w_scaled = h // scale, w // scale
        else:
            h_scaled, w_scaled = h, w

        # Simplified generation for speed
        x = np.zeros((c, h_scaled, w_scaled))
        val = [1, 0]
        
        if mode == 'uni':
            for i in range(c):
                x[i] = np.random.choice(val, size=(h_scaled, w_scaled))
        elif mode == 'inverted':
            xo = img[0].cpu().numpy()
            m = 255
            xt = np.floor(((1 - xo) * m)).astype(int)
            val = np.arange(m + 1)
            distri = np.zeros((c, len(val)))
            for i in range(c):
                unique, counts = np.unique(xt[i], return_counts=True)
                distri[i, unique] = counts
            distri /= distri.sum(axis=1, keepdims=True)
            for i in range(c):
                x[i] = np.random.choice(val, p=distri[i], size=(h_scaled, w_scaled))
            x /= m
        else:
            # Default: uniform random
            for i in range(c):
                x[i] = np.random.choice(val, size=(h_scaled, w_scaled))
        
        # Upscale if needed
        if scale > 1:
            x = np.transpose(x, (1, 2, 0))
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)
            x = np.transpose(x, (2, 0, 1))
        
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def l0b(self, img1, img2):
        """Calculate L0 distance between two images"""
        diff = torch.abs(img1 - img2)
        diff_per_pixel = diff.sum(dim=1) > 0.0
        return diff_per_pixel.sum().item()

    def convert1D_to_2D(self, idx, wi):
        """Convert 1D index to 2D coordinates"""
        return idx // wi, idx % wi

    def modify(self, pop, oimg, timg):
        """Apply population (pixel mask) to create perturbed image"""
        wi = oimg.shape[2]
        img = oimg.clone()
        p_idx = np.where(pop == 1)[0]
        if len(p_idx) > 0:
            c1, c2 = self.convert1D_to_2D(p_idx, wi)
            img[:, :, c1, c2] = timg[:, :, c1, c2]
        return img

    def feval_score(self, oimg, timg, olabel, tlabel, pop):
        """Evaluate fitness score for a population member"""
        xp = self.modify(pop, oimg, timg)
        
        with torch.no_grad():
            pred_score = self.model_wrapper(xp)
            
            rank = torch.argsort(pred_score[0], descending=True)
            top1_id = rank[0].item()
            top2_id = rank[1].item()
            
            if self.flag:  # Targeted
                tscore = pred_score[0, tlabel].item()
                top_score = pred_score[0, top1_id].item()
                outp_margin = top_score - tscore
                if self.ftype == 'margin':
                    outp = outp_margin
                else:  # ce
                    y = torch.tensor([tlabel], device=self.device)
                    outp = F.cross_entropy(pred_score, y, reduction='none').item()
            else:  # Untargeted
                pred_score_soft = F.softmax(pred_score, dim=1)
                oscore = pred_score_soft[0, olabel].item()
                top_score = pred_score_soft[0, top2_id].item() if top2_id != olabel else pred_score_soft[0, top1_id].item()
                outp_margin = oscore - top_score
                outp = outp_margin
        
        return outp, outp_margin

    def selection(self, x1, f1, x2, f2):
        """Selection operator for evolutionary algorithm"""
        if f2 < f1:
            return x2.copy(), f2, 0  # non_update
        return x1.copy(), f1, 1  # update

    def power_stepdecay_scheduler(self, q):
        """Power step decay scheduler for lambda parameter"""
        return self.lamda * (pow(q + 1.0, -self.m1) + self.m2**(q + 1)) / 2

    def sampling(self, fail_map, visit_map, bias_map, p, m):
        """Sample new population member using failure and visit maps"""
        ep = 1e-2
        num = fail_map + ep
        den = visit_map + ep
        
        # 1. Select remaining bits
        mask = num / den * p
        idxs = np.where(mask > 0)[0]
        if len(idxs) == 0:
            return p.copy(), np.array([])
            
        prob = mask[idxs] / mask[idxs].sum()
        outp = np.zeros_like(p, dtype=int)
        
        n_p = max(1, int(p.sum() * m))
        n_keep = int(p.sum()) - n_p
        if n_keep > 0 and n_keep <= len(idxs):
            idx = np.random.choice(idxs, n_keep, p=prob, replace=False)
            outp[idx] = 1
        
        old = np.where(np.logical_xor(outp, p))[0]
        
        # 2. Select new bits to add
        mask = (num / den * bias_map) * (1 - p)
        idxs = np.where(mask > 0)[0]
        if len(idxs) > 0 and n_p > 0:
            prob = mask[idxs] / mask[idxs].sum()
            n_add = min(n_p, len(idxs))
            idx = np.random.choice(idxs, n_add, p=prob, replace=False)
            outp[idx] = 1
        
        return outp, old

    def rand_init(self, oimg, timg, olabel, tlabel):
        """Random initialization of population"""
        wi, he = oimg.shape[2], oimg.shape[3]
        total_pixels = wi * he
        feval = []
        pop = []
        
        for i in range(self.pop_size):
            p = np.zeros(total_pixels, dtype=int)
            idx = np.random.choice(total_pixels, min(self.n_pix, total_pixels), replace=False)
            p[idx] = 1
            fitness, _ = self.feval_score(oimg, timg, olabel, tlabel, p)
            pop.append(p)
            feval.append(fitness)
        
        return pop, torch.tensor(feval, device=self.device)

    def _perturb(self, xs_t, loss_fct):
        """
        Optimized perturbation method that works iteratively
        Process entire batch at once instead of one sample at a time
        """
        n_queries = torch.zeros(xs_t.shape[0])
        
        if self.is_new_batch:
            # Initialize state for new batch
            # xs_t is (batch, H, W, C) in [0, 255] range
            x_batch = xs_t.permute(0, 3, 1, 2) / 255.0  # (B, C, H, W) in [0, 1]
            x_batch = x_batch.to(self.device)
            
            # Get labels and initialize target images
            with torch.no_grad():
                pred = self.model_wrapper(x_batch)
                src_labels = torch.argmax(pred, dim=1).cpu().numpy()
            
            # Initialize per-sample state
            for idx in range(xs_t.shape[0]):
                seed_val = self.seed + idx if self.seed is not None else None
                timg = self.search_space_init(x_batch[idx:idx+1], seed=seed_val,
                                             scale=self.init_scale, mode=self.init_mode)
                
                # Initialize evolutionary state
                pop, feval = self.rand_init(x_batch[idx:idx+1], timg, 
                                           src_labels[idx], src_labels[idx])
                
                wi, he = x_batch.shape[2], x_batch.shape[3]
                visit_map = np.zeros(wi * he)
                for p in pop:
                    visit_map += p
                    
                fail_map = np.zeros(wi * he)
                bias_map = torch.abs(x_batch[idx:idx+1] - timg)[0].sum(dim=0).reshape(-1).cpu().numpy() / 3
                
                best_idx = torch.argmin(feval).item()
                
                self.batch_state[idx] = {
                    'oimg': x_batch[idx:idx+1],
                    'timg': timg,
                    'olabel': src_labels[idx],
                    'tlabel': src_labels[idx],
                    'pop': pop,
                    'feval': feval,
                    'visit_map': visit_map,
                    'fail_map': fail_map,
                    'bias_map': bias_map,
                    'best_p': pop[best_idx].copy(),
                    'best_f': feval[best_idx].item(),
                    'nqry': len(pop),
                    'done': False
                }
                n_queries[idx] = len(pop)
            
            # Return current best adversarials
            result_batch = []
            for idx in range(xs_t.shape[0]):
                state = self.batch_state[idx]
                adv = self.modify(state['best_p'], state['oimg'], state['timg'])
                adv_output = (adv * 255.0).permute(0, 2, 3, 1)
                result_batch.append(adv_output.cpu())
            
            return torch.cat(result_batch, dim=0), n_queries
        
        else:
            # Continue evolution for non-finished samples
            result_batch = []
            
            for idx in range(xs_t.shape[0]):
                state = self.batch_state[idx]
                
                if state['done'] or state['nqry'] >= self.max_loss_queries:
                    # Keep current best
                    adv = self.modify(state['best_p'], state['oimg'], state['timg'])
                    adv_output = (adv * 255.0).permute(0, 2, 3, 1)
                    result_batch.append(adv_output.cpu())
                    continue
                
                # Run one evolution step
                lamda = self.power_stepdecay_scheduler(state['nqry'])
                offspring, old = self.sampling(state['fail_map'], state['visit_map'],
                                              state['bias_map'], state['best_p'], lamda)
                
                ftemp, f_mg = self.feval_score(state['oimg'], state['timg'],
                                               state['olabel'], state['tlabel'], offspring)
                state['nqry'] += 1
                n_queries[idx] = 1
                
                # Selection
                state['best_p'], state['best_f'], fh_update = self.selection(
                    state['best_p'], state['best_f'], offspring, ftemp)
                
                # Update maps
                if fh_update:
                    state['fail_map'][old] += 1
                if len(old) > 0:
                    state['visit_map'][old] += 1
                state['visit_map'] += offspring
                
                # Check termination
                if f_mg <= 0:
                    state['done'] = True
                
                # Return current best
                adv = self.modify(state['best_p'], state['oimg'], state['timg'])
                adv_output = (adv * 255.0).permute(0, 2, 3, 1)
                result_batch.append(adv_output.cpu())
            
            return torch.cat(result_batch, dim=0), n_queries

    def _config(self):
        """Return attack configuration"""
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "n_pix": self.n_pix,
            "pop_size": self.pop_size,
            "lamda": self.lamda,
            "m1": self.m1,
            "m2": self.m2,
            "seed": self.seed,
            "flag": self.flag,
            "ftype": self.ftype,
            "init_mode": self.init_mode,
            "init_scale": self.init_scale,
            "dataset": self.dataset,
            "attack_name": self.__class__.__name__
        }

    def run(self, xs, loss_fct, early_stop_extra_fct):
        """
        Override run method to setup model reference
        """
        # Ensure device is set
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Reset batch state for new run
        self.batch_state = {}
        
        # Call parent run method
        return super().run(xs, loss_fct, early_stop_extra_fct)