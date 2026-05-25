import torch 
import torch.nn.functional as F 

def non_neg(z): 
    return (z>=0).float().mean()

def act_dim(z): 
    return (z.abs().mean(dim=0)>0).float().sum() 

def sparsity(z):
    return 1 - (z.abs()>1e-5).float().mean()

def erank(z):
    z = z.float()
    s = torch.linalg.svdvals(z)
    s = s / s.sum()
    return -torch.sum(s * torch.log(s + 1e-6))

def orthogonality(features, eps=1e-5):
    active_dim_mask = features.abs().sum(0)>0
    features  = features[:,active_dim_mask] 
    n, d = features.shape
    features = F.normalize(features, dim=0)
    corr = features.T @ features 
    err = (corr - torch.eye(d, device=features.device)).abs() 
    # err = err.mean()
    err = err / (d*(d-1))
    return err

def semantic_consistency(features, labels, eps=1e-5, take_abs=False, topk=False):
    active_dim_mask = features.abs().sum(0)>0
    features  = features[:, active_dim_mask]
    features = F.normalize(features, dim=1) 

    acc_per_dim = []
    for i in range(features.shape[1]):
        active_sample_mask = features.abs()[:,i] > eps 
        labels_selected = labels[active_sample_mask]
        try:
            dist = labels_selected.bincount()
            dist = dist / dist.sum() 
            acc = dist.max().item() 
            acc_per_dim.append(acc)
        except:
            pass
    mean_acc =  torch.tensor(acc_per_dim).mean()
    return mean_acc


def semantic_entropy_sum_activation(features, labels, eps=1e-5):
    active_dim_mask = features.abs().sum(0) > eps
    features = features[:, active_dim_mask]
    unique_labels, label_indices = torch.unique(labels, return_inverse=True)
    
    entropy_per_dim = []
    n_dims = features.shape[1] 
    
    for i in range(n_dims):

        col_values = features[:, i].abs()

        class_activation_sums = torch.bincount(label_indices, weights=col_values)
        

        total_activation = class_activation_sums.sum()
        

        if total_activation < eps:
            continue
            

        p = class_activation_sums / total_activation
        

        valid_p = p[p > eps]
        if len(valid_p) > 0:
            entropy = -(valid_p * torch.log(valid_p)).sum()
            entropy_per_dim.append(entropy)
    
    # 6. 返回所有维度的平均熵
    if len(entropy_per_dim) == 0:
        return torch.tensor(0.0, device=features.device)
        
    return torch.tensor(entropy_per_dim, device=features.device).mean()



def semantic_entropy_mean_activation(features, labels, eps=1e-5):


    active_dim_mask = features.abs().sum(0) > eps
    features = features[:, active_dim_mask]

    unique_labels, label_indices = torch.unique(labels, return_inverse=True)
    

    class_counts = torch.bincount(label_indices).float()

    class_counts = class_counts + eps
    
    entropy_per_dim = []
    n_dims = features.shape[1] 
    

    for i in range(n_dims):
        col_values = features[:, i].abs()
        

        class_activation_sums = torch.bincount(label_indices, weights=col_values)
        
        class_activation_means = class_activation_sums / class_counts
        
        total_mean_activation = class_activation_means.sum()
        
        if total_mean_activation < eps:
            continue
            
        p = class_activation_means / total_mean_activation
        
        valid_p = p[p > eps]
        if len(valid_p) > 0:
            entropy = -(valid_p * torch.log(valid_p)).sum()
            entropy_per_dim.append(entropy)

    if len(entropy_per_dim) == 0:
        return torch.tensor(0.0, device=features.device)
        
    return torch.tensor(entropy_per_dim, device=features.device).mean()



def semantic_entropy_frequency(features, labels, eps=1e-5):

    active_dim_mask = features.abs().sum(0) > eps
    features = features[:, active_dim_mask]
    
    unique_labels, label_indices = torch.unique(labels, return_inverse=True)
    entropy_per_dim = []
    n_dims = features.shape[1]
    
    for i in range(n_dims):
        col_values = features[:, i].abs()
        
        is_active = (col_values > eps).float()
        
        class_active_counts = torch.bincount(label_indices, weights=is_active)
        
        total_active_count = class_active_counts.sum()
        
        if total_active_count < eps:
            continue
            
        p = class_active_counts / total_active_count
        
        valid_p = p[p > eps]
        if len(valid_p) > 0:
            entropy = -(valid_p * torch.log(valid_p)).sum()
            entropy_per_dim.append(entropy)
    
    if len(entropy_per_dim) == 0:
        return torch.tensor(0.0, device=features.device)
        
    return torch.tensor(entropy_per_dim, device=features.device).mean()
