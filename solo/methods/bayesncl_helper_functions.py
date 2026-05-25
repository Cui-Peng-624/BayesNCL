import math
import torch
import torch.nn.functional as F


def gumbel_sigmoid_no_gumbel(logits: torch.Tensor, temperature: float, hard: bool = False, gate_threshold: float = 0.5) -> torch.Tensor:        
    y = torch.sigmoid(logits / temperature) 
                    
    if hard:
        y_hard = (y > gate_threshold).float()
        y = y_hard.detach() + y - y.detach()
                    
    return y


def gumbel_sigmoid_no_gumbel_no_sigmoid(logits: torch.Tensor, hard: bool = False, gate_threshold: float = 0.5) -> torch.Tensor:
    y = logits
                    
    if hard:
        y_hard = (y > gate_threshold).float()
        y = y_hard.detach() + y - y.detach()
                    
    return y


def gumbel_sigmoid(logits: torch.Tensor, temperature: float, hard: bool = False, gate_threshold: float = 0.5) -> torch.Tensor:
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-4) + 1e-4)
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
    
    if hard:
        y_hard = (y_soft > gate_threshold).float()
        y = y_hard.detach() + y_soft - y_soft.detach()
    else:
        y = y_soft
        
    return y


def compute_bernoulli_kl_divergence(posterior_prob: torch.Tensor, prior_prob: float) -> torch.Tensor:
    eps = 1e-2
        

    posterior_prob = torch.clamp(posterior_prob, eps, 1 - eps)

    prior_prob_tensor = torch.as_tensor(prior_prob, device=posterior_prob.device, dtype=posterior_prob.dtype)
    prior_prob_tensor = torch.clamp(prior_prob_tensor, eps, 1 - eps)

    log_ratio1 = torch.log(posterior_prob) - torch.log(prior_prob_tensor)
    log_ratio2 = torch.log(1 - posterior_prob) - torch.log(1 - prior_prob_tensor)
        
    term1 = posterior_prob * log_ratio1
    term2 = (1 - posterior_prob) * log_ratio2
        
    kl_div = term1 + term2

    return kl_div.sum(dim=1).mean()


def apply_topk_mask(z: torch.Tensor, topk_k: int) -> torch.Tensor:
    batch_size, feature_dim = z.shape
    
    topk_k = min(topk_k, feature_dim)
    
    _, topk_indices = torch.topk(z, k=topk_k, dim=1)

    mask = torch.zeros(batch_size, feature_dim, device=z.device, dtype=torch.float32)

    batch_indices = torch.arange(batch_size, device=z.device).unsqueeze(1).expand(-1, topk_k)
    mask[batch_indices, topk_indices] = 1.0
    
    return mask
