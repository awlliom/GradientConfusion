

import numpy as np
import torch
from scipy.optimize import minimize
import torch.nn.functional as F
import multiprocessing as mp


def compute_batch_jv_chunked(model, batch_images, chunk_size=10):
    """
    Computes JVP for large batches by splitting into smaller chunks.
    """
    B = batch_images.shape[0]
    all_Jv = []
    all_probs = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        images_chunk = batch_images[start:end].clone().detach().requires_grad_(True)

        logits = model(images_chunk) 
        probs = F.softmax(logits, dim=1)

        _, sorted_indices = torch.sort(probs, dim=1, descending=True)
        i = sorted_indices[:, 0]
        j = sorted_indices[:, 1]

        batch_indices = torch.arange(end - start, device=batch_images.device)
        logits_i = logits[batch_indices, i]
        logits_j = logits[batch_indices, j]

        s = logits_i - logits_j

        v = torch.autograd.grad(torch.sum(s), images_chunk, retain_graph=True)[0]

        _, Jv_chunk = torch.autograd.functional.jvp(
            lambda x: model(x), images_chunk, v
        )

        all_Jv.append(-Jv_chunk.detach())
        all_probs.append(probs.detach())

        del logits, probs, i, j, logits_i, logits_j, s, v, Jv_chunk
        torch.cuda.empty_cache()

    return torch.cat(all_Jv, dim=0), torch.cat(all_probs, dim=0)

def softmax_gradient_torch_vectorized(y, i, j):
    """
    Vectorized computation of softmax gradient with respect to ith and jth class.
    """
    delta = y[j] - y[i]
    grad = delta * y
    grad[i] += y[i]
    grad[j] -= y[j]
    return grad

def best_effort_match_sparse(b_star, y_prime, epsilon, i, j, top_k=5):
    """
    Optimized version that only optimizes the top_k most significant elements.
    
    Args:
        b_star: Target gradient vector
        y_prime: Initial probability vector 
        epsilon: L1 constraint bound
        i, j: Indices of top two probabilities
        top_k: Number of main elements to optimize (default 5)
    """
    n_classes = len(y_prime)
    
    # Find the main indices based on values of b_star
    # Always include i and j in the optimization set
    top_indices = np.argsort(b_star)[:top_k]
    bottom_indices = np.argsort(b_star)[-top_k:]
    main_indices = np.concatenate([top_indices, bottom_indices])
    # print(len(top_indices))
    # Ensure i and j are included in the optimization set
    if i not in main_indices:
        main_indices = np.append(main_indices, i)
    if j not in main_indices:
        main_indices = np.append(main_indices, j)
    
    main_indices = np.unique(main_indices)
    top_k_actual = len(main_indices)
    
    # Create mask for indices to optimize
    mask = np.zeros(n_classes, dtype=bool)
    mask[main_indices] = True
    
    # Split into optimized and fixed components
    y_fixed = y_prime.copy()
    y_fixed[mask] = 0  # Zero out the elements we'll optimize
    
    # Initial values for optimization (only top_k elements)
    y_init_sparse = y_prime[mask]
    
    def objective(y_sparse):
        # Reconstruct full y vector
        y_full = y_fixed.copy()
        y_full[mask] = y_sparse
        
        # Compute gradient
        B = softmax_gradient_torch_vectorized(y_full, i, j)
        
        # Only consider the loss for the optimized elements
        # This focuses the optimization on matching the important gradients
        return -np.dot(B[mask], b_star[mask])
        # return -np.dot(B[mask], b_star[mask])/(np.linalg.norm(B[mask]) * np.linalg.norm(b_star[mask]))
    
    # The sum of all probabilities (fixed + optimized) should be 1, we normalize the optimized result hence we didn't include this in the optimizer
    def prob_sum_constraint(y_sparse):
        return np.sum(y_sparse) + np.sum(y_fixed) - 1
    
    # L1 constraint only on the optimized elements
    def l1_constraint(y_sparse):
        y_full = y_fixed.copy()
        y_full[mask] = y_sparse
        return epsilon - np.sum(np.abs(y_full - y_prime))
    
    # Bounds for the sparse vector
    bounds = [(0, 1)] * top_k_actual
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': l1_constraint}
    ]
    # Optimize
    result = minimize(
        objective,
        y_init_sparse,
        bounds=bounds,
        constraints=constraints,
        method='trust-constr',
        options={'disp': False, 'maxiter': 1000, 
        'xtol': 1e-4,
        'gtol': 1e-4,
        'barrier_tol': 1e-4} 
    )
    
    # Reconstruct full solution
    y_opt = y_fixed.copy()
    y_opt[mask] = result.x
    
    # Ensure it's a valid probability distribution
    y_opt = np.clip(y_opt, 0, 1)
    y_opt = y_opt / np.sum(y_opt)
    max_idx = np.argmax(y_opt)
    y_opt[i], y_opt[max_idx] = y_opt[max_idx], y_opt[i]
    # y_opt = randomize_probs_keep_top_class(y_prime)
    
    B_opt = softmax_gradient_torch_vectorized(y_opt, i, j)
    
    return y_opt, B_opt, True
    


def process_single_sample(args):
    """Process a single sample for multiprocessing."""
    b_star, y_prime, epsilon, i, j, top_k = args
    y_opt, B_opt, solved = best_effort_match_sparse(b_star, y_prime, epsilon, i, j, top_k=top_k)
    return y_opt

def parallel_optimization(b_star_batch, probs_batch, epsi=0.5, top_k=5, n_workers=None):
    """
    Parallelize the optimization across multiple CPU cores.
    
    Args:
        b_star_batch: Tensor of shape (batch_size, num_classes, 1)
        probs_batch: Tensor of shape (batch_size, num_classes, 1)
        epsi: Epsilon value for L1 constraint
        top_k: Number of main classes to optimize
        n_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        List of optimized probability distributions
    """
    batch_size = b_star_batch.size(0)
    
    # Prepare arguments for each sample
    args_list = []
    for b in range(batch_size):
        bstar = b_star_batch[b].detach().squeeze().cpu().numpy()
        y_prime = probs_batch[b].detach().squeeze().cpu().numpy()
        i = int(np.argmax(y_prime))
        j = int(np.argsort(y_prime)[-2])
        args_list.append((bstar, y_prime, epsi, i, j, top_k))
    
    # Use multiprocessing pool
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_single_sample, args_list)
    
    # Convert back to torch tensors
    out = [torch.tensor(pr_y) for pr_y in results]
    return torch.stack(out)