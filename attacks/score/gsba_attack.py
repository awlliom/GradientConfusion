from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor as t
from torch.autograd import Variable
from scipy.fftpack import dct, idct
import copy
import math

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack


class GSBAAttack(ScoreBlackBoxAttack):
    """
    GSBA (Gradient Estimation on Decision Boundary for Score-Based Black-box Attack)
    
    Based on the paper and implementation from GSBA-K-main folder.
    This is a wrapper to make GSBA compatible with the existing attack framework.
    """

    def __init__(self, max_loss_queries, epsilon, p, lb, ub, batch_size, name,
                 dim_reduc_factor=4, iteration=150, top_k=1, base_query=30,
                 tol=0.0001, sigma=0.0002, step_size=6, query_type='nonuniform',
                 b_grad_approach_name='Proposed', init_method='grad_Proposed'):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm of perturbation
        :param lb: data lower bound
        :param ub: data upper bound
        :param batch_size: batch size for attack
        :param dim_reduc_factor: dimension reduction factor for DCT
        :param iteration: maximum number of iterations
        :param top_k: top-k for untargeted attack
        :param base_query: base number of queries for gradient estimation
        :param tol: tolerance for binary search
        :param sigma: noise level for gradient estimation
        :param step_size: step size for initial boundary finding
        :param query_type: 'uniform' or 'nonuniform' query distribution
        :param b_grad_approach_name: gradient estimation approach on boundary
        :param init_method: initial boundary finding method
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size=batch_size,
                         name="GSBA")
        
        self.dim_reduc_factor = dim_reduc_factor
        self.iteration = iteration
        self.top_k = top_k
        self.I0 = base_query
        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = 40
        self.step_size = step_size
        self.query_type = query_type
        self.approach_name = b_grad_approach_name
        self.init_method = init_method
        
        # Attack state variables
        self.model = None
        self.src_img = None
        self.src_lbl = None
        self.tar_lbls = [None]
        self.attack_type = 'untargeted'
        self.mean = None
        self.std = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0

    def clip_image_values(self, x, minv, maxv):
        """Clip image values to valid range"""
        return torch.clamp(x, minv, maxv)

    def inv_tf(self, x, mean, std):
        """Inverse transformation for visualization"""
        for i in range(len(mean)):
            x[i] = np.multiply(x[i], std[i], dtype=np.float32)
            x[i] = np.add(x[i], mean[i], dtype=np.float32)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 0, 1)
        return x

    def is_adversarial(self, x):
        """Query success indicator - Does NOT count queries (counted in main loop)"""
        pred_score = self.model(x).data
        val, ind = torch.sort(pred_score, descending=True)
        ind = ind.reshape(-1).cpu().numpy()
        
        # NOTE: Original GSBA doesn't count queries here - counted in main Attack loop
        
        if self.attack_type == 'untargeted':
            is_adv = self.src_lbl not in ind[:self.top_k]
        else:
            is_adv = set(ind[:self.top_k]) <= set(self.tar_lbls)
        return is_adv

    def get_socres(self, image):
        """Get softmax scores - Returns on CPU like original"""
        pred_score = self.model.forward(Variable(image, requires_grad=True)).data
        val = F.softmax(pred_score, dim=-1).data
        # NOTE: Original doesn't count here - query counted in main loop
        return val.cpu()

    def get_sorted_scores_nd_indices(self, image):
        """Get sorted scores and indices"""
        pred_score = self.model.forward(Variable(image, requires_grad=True)).data
        val, ind = torch.sort(pred_score, descending=True)
        val = F.softmax(val, dim=-1).data
        # NOTE: Original doesn't count here
        return val.reshape(-1).cpu().numpy(), ind.reshape(-1).cpu().numpy()

    def find_random_adversarial(self, image):
        """Find random adversarial example - OPTIMIZED"""
        num_calls = 1
        step = 0.03
        perturbed = image
        while self.is_adversarial(perturbed) == 0:
            # Generate perturbation directly on device
            pert = torch.randn(image.shape, device=self.device)
            perturbed = image + num_calls * step * pert
            perturbed = self.clip_image_values(perturbed, self.lb, self.ub)
            num_calls += 1
        return perturbed, num_calls

    def bin_search(self, x_0, x_random):
        """Binary search for boundary"""
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1
            if self.is_adversarial(mid):
                adv = mid
            else:
                cln = mid
            # Keep norm computation on GPU
            if torch.norm(adv - cln).item() < self.tol or num_calls >= 100:
                break
        return adv, num_calls

    def find_random(self, x, n):
        """Generate random perturbations with DCT reduction - FIXED to match original"""
        image_size = x.shape
        # Original creates (n, 3, H, W) output
        out = torch.zeros(n, 3, int(image_size[-2]), int(image_size[-1]))
        
        for i in range(n):
            # Create temp tensor with batch dimension like original
            temp = torch.zeros(image_size[0], 3, int(image_size[-2]), int(image_size[-1]))
            fill_size = int(image_size[-1] / self.dim_reduc_factor)
            temp[:, :, :fill_size, :fill_size] = torch.randn(image_size[0], temp.size(1), fill_size, fill_size)
            
            if self.dim_reduc_factor > 1.0:
                # Apply DCT on CPU like original (scipy DCT is CPU-only)
                temp = torch.from_numpy(idct(idct(temp.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'))
            out[i] = temp
        
        return out

    def grad_estimation_untargeted(self, x, noises):
        """Gradient estimation for untargeted attack - FIXED to match original"""
        Xs = x + self.sigma * noises
        batch_size = self.grad_estimator_batch_size
        batch_num = int(np.ceil(Xs.shape[0] / batch_size))
        logits = []
        for k in range(batch_num):
            batch_Xs = Xs[k * batch_size:(k + 1) * batch_size]
            # Keep logits on CPU like original for efficiency
            logits_Xs = self.model.forward(Variable(batch_Xs, requires_grad=True)).data.cpu()
            logits.append(logits_Xs)
        logits = torch.cat(logits, 0)
        # NOTE: Original doesn't count queries here - counted in main loop

        pred_scores = F.softmax(logits, dim=-1)
        Pcs = pred_scores[:, self.src_lbl].clone()

        pred_scores[:, self.src_lbl] = -torch.inf
        sorted_scores, _ = torch.sort(pred_scores, 1, descending=True)
        Pvk = sorted_scores[:, self.top_k - 1]

        Fs = Pvk - Pcs
        Fs = Fs[:, None, None, None].to(self.device)
        weighted_noises = Fs * 10000 * noises
        grad = sum(weighted_noises)  # Use sum() instead of .sum(0) like original
        
        if torch.sum(Fs == 0) == Xs.shape[0]:
            val, ind = torch.sort(logits, descending=True)
            # Match original's list comprehension approach
            is_advs = [1 if self.src_lbl not in ind[k][:self.top_k].numpy() else -1 for k in range(ind.shape[0])]
            is_advs = torch.tensor(is_advs)
            is_advs = is_advs[:, None, None, None].to(self.device)
            weighted_noises = is_advs * noises
            grad = sum(weighted_noises)
        return grad / torch.norm(grad)

    def grad_estimation_init_untargeted(self, x, noises):
        """Gradient estimation for initialization (untargeted) - FIXED to match original"""
        p_cur = self.get_socres(x)
        scr_pred_cur = p_cur[:, self.src_lbl].clone()
        p_cur[:, self.src_lbl] = -torch.inf
        sorted_scores_cur, _ = torch.sort(p_cur, 1, descending=True)
        val_cur = sorted_scores_cur[:, self.top_k - 1]
        F_cur = val_cur - scr_pred_cur

        Xs = x + self.sigma * noises
        batch_size = self.grad_estimator_batch_size
        batch_num = int(np.ceil(Xs.shape[0] / batch_size))
        logits = []
        for k in range(batch_num):
            batch_Xs = Xs[k * batch_size:(k + 1) * batch_size]
            # Keep on CPU like original
            logits_Xs = self.model(batch_Xs).data.cpu()
            logits.append(logits_Xs)
        logits = torch.cat(logits, 0)
        # NOTE: Original doesn't count queries here

        pred_scores = F.softmax(logits, dim=-1)
        scr_pred = pred_scores[:, self.src_lbl].clone()

        pred_scores[:, self.src_lbl] = -torch.inf
        sorted_scores, _ = torch.sort(pred_scores, 1, descending=True)
        val = sorted_scores[:, self.top_k - 1]

        F_n = val - scr_pred
        diffs = F_n - F_cur
        diffs = diffs[:, None, None, None].to(self.device)
        weighted_noises = diffs * noises
        grad = sum(weighted_noises)  # Use sum() like original
        
        if torch.sum(diffs == 0) == Xs.shape[0]:
            val, ind = torch.sort(logits, descending=True)
            # Match original's implementation
            is_advs = [1 if self.src_lbl not in ind[k][:self.top_k].numpy() else -1 for k in range(ind.shape[0])]
            is_advs = torch.tensor(is_advs)
            is_advs = is_advs[:, None, None, None].to(self.device)
            weighted_noises = is_advs * noises
            grad = sum(weighted_noises)
        return grad / torch.norm(grad)

    def find_next_boundary_untargeted(self, x_s, g, x_b):
        """Find next boundary point for untargeted attack"""
        num_calls = 1
        g_hat = g / torch.norm(g)
        psi_hat = (x_b - x_s) / torch.norm(x_b - x_s)
        phi = torch.acos(torch.dot(psi_hat.reshape(-1), g_hat.reshape(-1)))
        
        # Pre-compute constant for efficiency
        half_pi = math.pi / 2
        
        while True:
            alpha = half_pi * (1 - 1 / pow(2, num_calls))
            gamma = (torch.sin(phi) * math.cos(alpha) / math.sin(alpha) - torch.cos(phi)).item()
            theta = (g_hat + gamma * psi_hat) / torch.norm(g_hat + gamma * psi_hat)
            xq = x_s + theta * torch.norm(x_b - x_s) * torch.dot(psi_hat.reshape(-1), theta.reshape(-1))
            xq = self.clip_image_values(xq, self.lb, self.ub)
            if not self.is_adversarial(xq):
                break
            num_calls += 1
            if num_calls > 40:
                return x_b, num_calls
        
        perturbed, n_calls = self.SemiCircular_boundary_search(x_s, x_b, xq)
        return perturbed, num_calls + n_calls

    def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
        """Semi-circular boundary search"""
        num_calls = 0
        norm_dis = torch.norm(x_b - x_0)
        boundary_dir = (x_b - x_0) / torch.norm(x_b - x_0)
        clean_dir = (p_near_boundary - x_0) / torch.norm(p_near_boundary - x_0)
        adv_dir = boundary_dir
        adv = x_b
        clean = x_0
        
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir / torch.norm(mid_dir)
            theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) /
                             (torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir)))
            d = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) * norm_dis
            x_mid = x_0 + mid_dir * d
            num_calls += 1
            
            if self.is_adversarial(x_mid):
                adv_dir = mid_dir
                adv = x_mid
            else:
                clean_dir = mid_dir
                clean = x_mid
            
            # Keep computation on GPU
            if torch.norm(adv - clean).item() < self.tol:
                break
            if num_calls > 100:
                break
        return adv, num_calls

    def initial_boundary_with_grad(self, x):
        """Find initial boundary point using gradient estimation - FIXED query counting"""
        while self.is_adversarial(x) == 0:
            noises = self.find_random(x, self.I0)
            grad = self.grad_estimation_init_untargeted(x, noises.to(self.device))
            # Count queries: 1 for get_socres in grad_estimation + I0 for gradient estimation
            self.all_queries += (1 + self.I0)
            x = x + self.step_size * grad
            x = self.clip_image_values(x, self.lb, self.ub)
            # Count query for is_adversarial check in while condition
            self.all_queries += 1
            if self.all_queries >= self.max_loss_queries:
                return x, 'failed'
        return x, 'success'

    def _perturb(self, xs_t, loss_fct):
        """
        Main perturbation method for compatibility with the framework
        GSBA runs all iterations internally, so we do everything on first call
        """
        if self.is_new_batch:
            # Initialize for new batch - run complete GSBA attack
            batch_results = []
            n_queries_batch = torch.zeros(xs_t.shape[0])
            
            for idx in range(xs_t.shape[0]):
                # Extract single image and convert to model input format
                # xs_t is (batch, H, W, C) in [0, 255] range
                x_single = xs_t[idx:idx+1]
                x_input = x_single.permute(0, 3, 1, 2) / 255.0  # Convert to (1, C, H, W) in [0, 1]
                x_input = x_input.to(self.device)
                
                # Store original image and get source label
                self.src_img = x_input
                self.all_queries = 0
                self.src_lbl = torch.argmax(self.model(x_input).data).item()
                self.all_queries += 1
                
                # Find initial adversarial boundary point
                if 'random' in self.init_method:
                    x_in_adv, Q_to_boundary = self.find_random_adversarial(self.src_img)
                    self.all_queries += Q_to_boundary  # Count random search queries
                    x_b, Q_bin = self.bin_search(self.src_img, x_in_adv)
                    self.all_queries += Q_bin  # Count binary search queries
                else:
                    x_in_adv, note = self.initial_boundary_with_grad(self.src_img)
                    if note == 'failed':
                        # Failed to find initial boundary
                        batch_results.append(x_single)
                        n_queries_batch[idx] = self.all_queries
                        continue
                    x_b, Q_bin = self.bin_search(self.src_img, x_in_adv)
                    self.all_queries += Q_bin  # Count binary search queries
                
                # Run GSBA iterations
                size = self.src_img.shape
                for i in range(self.iteration):
                    # Determine number of queries for this iteration
                    if self.query_type == 'uniform':
                        I_n = int(self.I0)
                    else:
                        I_n = int(self.I0 * np.sqrt(i + 1))
                    
                    # Generate noises
                    noises = self.find_random(self.src_img, I_n)
                    
                    # Estimate gradient on boundary
                    grad_oi = self.grad_estimation_untargeted(x_b, noises.to(self.device))
                    self.all_queries += I_n  # Count gradient estimation queries
                    
                    # Find next boundary point
                    x_adv, qs = self.find_next_boundary_untargeted(self.src_img, grad_oi, x_b)
                    self.all_queries += qs  # Count boundary search queries
                    
                    # Update boundary point
                    x_b = x_adv
                    
                    # Check if query budget exceeded
                    if self.all_queries >= self.max_loss_queries:
                        break
                
                # Convert back to output format: (1, C, H, W) [0,1] -> (1, H, W, C) [0, 255]
                x_adv_output = x_b * 255.0
                x_adv_output = x_adv_output.permute(0, 2, 3, 1)
                batch_results.append(x_adv_output.cpu())
                n_queries_batch[idx] = self.all_queries
            
            result = torch.cat(batch_results, dim=0)
            # Return results - framework won't call _perturb again for this batch
            return result, n_queries_batch
        else:
            # Should not be called again for same batch since GSBA runs all iterations internally
            # Just return unchanged
            return xs_t, torch.zeros(xs_t.shape[0])

    def _config(self):
        """Return attack configuration"""
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "dim_reduc_factor": self.dim_reduc_factor,
            "iteration": self.iteration,
            "top_k": self.top_k,
            "base_query": self.I0,
            "tol": self.tol,
            "sigma": self.sigma,
            "step_size": self.step_size,
            "query_type": self.query_type,
            "b_grad_approach": self.approach_name,
            "init_method": self.init_method,
            "attack_name": self.__class__.__name__
        }

    def set_model(self, model):
        """
        Set the model for GSBA attack
        
        Args:
            model: The PyTorch model to attack
        """
        self.model = model
    
    def run(self, xs, loss_fct, early_stop_extra_fct):
        """
        Override run method to setup model reference
        """
        # Ensure device is set
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Call parent run method
        return super().run(xs, loss_fct, early_stop_extra_fct)