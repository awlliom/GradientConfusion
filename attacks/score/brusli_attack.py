
import torch
import numpy as np
from utils import *
import cv2
import pandas as pd

from utils.defense import *

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack

def search_space_init(img,seed=0,scale=1,mode='uni',scale_mode='INTER_LINEAR'):
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    img_f = img.float()
    scale_factor = 255.0 if img_f.max().item() > 1.0 else 1.0
    img_norm = img_f / scale_factor
    # 1. find the distribution of reveresed image (opposite)
    # img = img.permute(2,0,1).unsqueeze(0)
    c = img.shape[1]
    w = img.shape[2]//scale
    h = img.shape[3]//scale
    # print('img shape',img.shape)
    xo = img_norm[0].cpu().numpy().reshape(c,-1)
    x_pd=pd.DataFrame(np.round(1-xo.transpose()), columns=['r', 'g','b'])

    p = (x_pd.sum()/x_pd.shape[0]).to_numpy()
    distri = np.stack((p,1-p),axis = 1)

    # 2. generate customized distribution image in lower scale if scale > 1
    val = [1,0]
    x = np.zeros((c,w,h))

    # =================== generate ===================
    if mode == 'uni': #final selection used for the paper
        for i in(range(c)):
            x[i] = np.random.choice(val, size=(w, h))
    elif mode == 'custom':
        for i in(range(c)):
            x[i] = np.random.choice(val, p=distri[i], size=(w, h))
    elif mode == 'sp':
        x[0] = np.random.choice(val, size=(w, h))
        x[1] = x[0]
        x[2] = x[0]
    elif mode == 'sp_custom':
        x[0] = np.random.choice(val, p=(distri[0]),size=(w, h))
        x[1] = x[0]
        x[2] = x[0]
    elif mode == 'inverted':
        m = 2**8-1 # scale level
        val = np.arange(m+1)
        xt = np.floor(((1-xo)*m)).astype(int)
        distri = np.zeros((c,len(val)))
        for i in range(c):
            for j in (val):
                distri[i,j] = np.count_nonzero(xt[i] == j)
            
        distri /= (w*h)
        for i in(range(c)):
            x[i] = np.random.choice(val, p=distri[i], size=(w, h))
        x /=m
    
    # 3. =============== Upscale ===================
    x = np.transpose(x,(1,2,0))

    if scale_mode == 'INTER_LINEAR':
        x = cv2.resize(x, (0, 0), fx=scale, fy=scale)
    elif scale_mode == 'INTER_NEAREST':
        x = cv2.resize(x, (scale*w,scale*h), interpolation=cv2.INTER_NEAREST)    
    elif scale_mode == 'INTER_AREA':
        x = cv2.resize(x, (scale*w,scale*h), interpolation=cv2.INTER_AREA) 
    elif scale_mode == 'INTER_CUBIC':
        x = cv2.resize(x, (scale*w,scale*h), interpolation=cv2.INTER_CUBIC) 

    x = x * scale_factor
    xo = torch.tensor(
        np.transpose(x, (2, 0, 1)), dtype=torch.float, device=img.device
    ).unsqueeze(0)
            
    return xo

# ======================== Measurement ========================

def l0b(img1,img2):
    xo = torch.abs(img1-img2)
    d = torch.sum(xo,1)>0.0
    return d.sum().item()

# ================ l0_projection for HSJA adapted ==============
def project_l0(original_image, perturbed_images, k):
    '''
    1. Clone "https://github.com/Jianbo-Lab/HSJA"
    2. Replace projection step built for l_2 and l_inf in the original code
    '''
    
    x = np.abs(original_image[0] - perturbed_images[0])
    wi = original_image.shape[2]
    x2 = x**2
    x2 = np.sum(x2,axis=0)
    x2 = x2.reshape(1,-1)
    n_same_px = len(np.where(x2==0)[0])
    out_images = original_image.copy()
    
    if n_same_px+k<wi*wi:
        idxs = np.argsort(x2)[:,n_same_px :n_same_px +k]
        c1 = idxs //wi
        c2 = idxs - c1 * wi
        out_images[:,:,c1,c2] = perturbed_images[:,:,c1,c2]

    return out_images

# main attack
class BruSLeAttack(ScoreBlackBoxAttack):
    def __init__(self, max_loss_queries, epsilon, p, lb, ub, batch_size, name,
                n_pix = 4, pop_size = 10, lamda=0.01, m1=0.24, m2=0.997, seed = None, flag=True, ftype='margin', init_mode= "uni", init_scale=1):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                            max_loss_queries=max_loss_queries,
                            epsilon=epsilon,
                            p=p,
                            lb=lb,
                            ub=ub,
                            batch_size= batch_size,
                            name="BruSLe")

        self.n_pix = n_pix 
        self.pop_size = pop_size
        self.lamda = lamda
        self.m1 = m1
        self.m2 = m2
        self.seed = seed
        self.flag = flag
        self.ftype = ftype
        self.scale= init_scale
        self.mode = init_mode
        self.scale_mode = "INTER_NEAREST"
        self._p = None
        self._fp = None
        self._visit_map = None
        self._fail_map = None
        self._bias_map = None
        self._timg = None
        self._oimg = None
        self._nqry = 0

    def selection(self,x1,f1,x2,f2):
        if f1 is None:
            return x2.copy(), f2, np.zeros(x2.shape[0], dtype=bool)
        better = f2 < f1
        if torch.any(better):
            better_mask = better.detach().cpu().numpy()
            x1[better_mask] = x2[better_mask]
        fo = torch.where(better, f2, f1)
        fh_update = (~better).detach().cpu().numpy()
        return x1, fo, fh_update

    
    def power_stepdecay_scheduler(self,q):
        
        lamda = self.lamda * (pow(q + 1.0, -self.m1) + self.m2**(q+1))/2
        
        return lamda

    def convert1D_to_2D(self,idx,wi):
        c1 = idx //wi
        c2 = idx % wi # = idx - c1 * wi
        return c1, c2
        
    def modify(self,pop,oimg,timg):
        if isinstance(pop, torch.Tensor):
            pop = pop.detach().cpu().numpy()
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        bsz, _, h, w = oimg.shape
        if pop.shape[0] != bsz:
            raise ValueError("pop batch size does not match oimg batch size")
        if pop.shape[1] != h * w:
            raise ValueError("pop spatial size does not match oimg spatial size")
        mask = torch.from_numpy(pop.astype(np.bool_)).to(oimg.device)
        mask = mask.view(bsz, 1, h, w)
        return torch.where(mask, timg, oimg)

    def rand_init(self,oimg,timg, loss_fct):

        nqry = 0
        bsz, _, h, w = oimg.shape
        dim = h * w
        visit_map = np.zeros((bsz, dim), dtype=np.int32)
        best_p = None
        best_f = None

        for _ in range(self.pop_size):
            p = np.zeros((bsz, dim), dtype=np.int8)
            for b in range(bsz):
                idx = np.random.choice(dim, self.n_pix, replace=False)
                p[b, idx] = 1
            visit_map += p
            nqry += 1
            fitness,_ = self.feval_score(oimg,timg,p,loss_fct)
            if best_f is None:
                best_f = fitness.detach()
                best_p = p.copy()
            else:
                better = fitness < best_f
                if torch.any(better):
                    better_mask = better.detach().cpu().numpy()
                    best_p[better_mask] = p[better_mask]
                    best_f = torch.where(better, fitness, best_f)

        return best_p, best_f, visit_map, nqry

    def feval_score(self,oimg,timg,pop, loss_fct):

        xp = self.modify(pop,oimg,timg)
        xp = xp.permute(0, 2, 3, 1)
        score = loss_fct(xp)
        # loss_fct is higher-is-better in this codebase; BruSLe minimizes margin.
        outp_margin = -score
        return outp_margin, outp_margin



    def visited_pixel_map(self,visit_map,p): # record history of all pixels (search space).
        out_a = visit_map.copy()
        out_a += p # number of time of visit
        out_b = np.clip(out_a,0,1) # count visited or not
        n_px = out_b.sum()
        return out_a,out_b,n_px


# =================================== Method 4 =======================================
    def sampling(self,fail_map,visit_map,bias_map,p,m):

        ep = 1e-2
        num = fail_map + ep# s+a: number of succ + a
        den = visit_map + ep# s+a + N-s+b = N+a+b: number of Visit + a+b 
        
        # ep is a need to avoid den = 0 and num = 0
        # num/den < 1
        
        # 1. select remaing bits
        mask = num/den*p 
        #---------------------------------------------------
        idxs = np.where(mask>0)[0] # => pixel position
        prob = mask[idxs] # => value of 'fail_pix' matrix
        prob = prob/prob.sum()    
        outp = np.zeros(*p.shape).astype(int)
        n_p = int(p.sum()*m)
        if n_p<1:
            n_p=1
        idx = np.random.choice(idxs,p.sum()-n_p,p=prob,replace=False)
        outp[idx]=1
        #---------------------------------------------------

        tmp = np.logical_xor(outp,p)
        idx = np.where(tmp==1)[0]
        old = idx.copy()

        # 2. select new bits to add in
        mask = (num/den * (bias_map))*(1-p)
        idxs = np.where(mask>0)[0]
        prob = mask[idxs] # => value of 'fail_pix' matr
        prob = prob/prob.sum()    
        idx = np.random.choice(idxs,n_p,p=prob,replace=False)
        outp[idx]=1

        return outp,old

    def _perturb(self,oimg, loss_fct,max_query=10000):
        oimg = oimg.permute(0, 3, 1, 2)
        batch_size = oimg.shape[0]
        init_queries = 0

        if self.is_new_batch:
            self._oimg = oimg
            if self.seed is not None:
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
            timgs = []
            for i in range(batch_size):
                timgs.append(search_space_init(
                    oimg[i:i + 1],
                    seed=None,
                    scale=self.scale,
                    mode=self.mode,
                    scale_mode=self.scale_mode,
                ))
            self._timg = torch.cat(timgs, dim=0)
            self._p, self._fp, self._visit_map, self._nqry = self.rand_init(
                oimg, self._timg, loss_fct)
            self._fail_map = np.zeros_like(self._visit_map)
            self._bias_map = (
                torch.abs(self._oimg - self._timg).sum(dim=1) / self._oimg.shape[1]
            ).reshape(batch_size, -1).cpu().numpy()
            init_queries = self.pop_size
        else:
            oimg = self._oimg

        lamda = self.power_stepdecay_scheduler(self._nqry)
        offspring = np.zeros_like(self._p)
        old_idxs = []
        for i in range(batch_size):
            offspring_i, old_i = self.sampling(
                self._fail_map[i],
                self._visit_map[i],
                self._bias_map[i],
                self._p[i],
                lamda,
            )
            offspring[i] = offspring_i
            old_idxs.append(old_i)

        self._nqry += 1
        ftemp,_ = self.feval_score(oimg, self._timg, offspring, loss_fct)
        self._p, self._fp, fh_update = self.selection(
            self._p, self._fp, offspring, ftemp)

        for i in range(batch_size):
            if fh_update[i]:
                self._fail_map[i, old_idxs[i]] += 1
            self._visit_map[i, old_idxs[i]] += 1
            self._visit_map[i] += offspring[i]

        adv = self.modify(self._p, oimg, self._timg)
        num_queries = torch.ones(batch_size) * (1 + init_queries)
        return adv.permute(0, 2, 3, 1), num_queries
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
            "init_scale":self.scale,
            "init_mode":self.mode,
            "attack_name": self.__class__.__name__
        }
