# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import numpy as np
import omegaconf 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from solo.losses.simclr import *
from solo.methods.bayesncl_helper_functions import gumbel_sigmoid_no_gumbel, gumbel_sigmoid_no_gumbel_no_sigmoid, gumbel_sigmoid, compute_bernoulli_kl_divergence, apply_topk_mask
from solo.methods.bayesncl_eval_helper_functions import non_neg, act_dim, sparsity, erank, orthogonality, semantic_consistency, semantic_entropy_sum_activation, semantic_entropy_mean_activation, semantic_entropy_frequency
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim 
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim 

        self.non_neg = cfg.method_kwargs.non_neg

        self.use_bayes = cfg.method_kwargs.use_bayes
        if self.use_bayes:
            self.gating_head = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
            self.gating_head_lr: float = cfg.optimizer.gating_head_lr
            self.gate_threshold = cfg.method_kwargs.gate_threshold
            self.bayesian_prior_prob = cfg.method_kwargs.bayesian_prior_prob
            self.bayesian_kl_lambda = cfg.method_kwargs.bayesian_kl_lambda
            self.sigmoid_temperature = cfg.method_kwargs.sigmoid_temperature
        
        self.use_topk = cfg.method_kwargs.use_topk
        if self.use_topk:
            self.topk_k = cfg.method_kwargs.topk_k

        self.gate_type = cfg.method_kwargs.gate_type
        if self.gate_type not in ["STE", "gumbel_sigmoid"]:
            raise ValueError(f"gate_type {self.gate_type} is not supported")

        self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)
        cfg.method_kwargs.non_neg = omegaconf_select(cfg, "method_kwargs.non_neg", None)

        cfg.method_kwargs.use_bayes = omegaconf_select(cfg, "method_kwargs.use_bayes", False)
        cfg.optimizer.gating_head_lr = omegaconf_select(cfg, "optimizer.gating_head_lr", cfg.optimizer.lr)
        cfg.method_kwargs.gate_threshold = omegaconf_select(cfg, "method_kwargs.gate_threshold", 0.5)
        cfg.method_kwargs.bayesian_prior_prob = omegaconf_select(cfg, "method_kwargs.bayesian_prior_prob", 0.00)
        cfg.method_kwargs.bayesian_kl_lambda = omegaconf_select(cfg, "method_kwargs.bayesian_kl_lambda", 0.0)
        cfg.method_kwargs.sigmoid_temperature = omegaconf_select(cfg, "method_kwargs.sigmoid_temperature", 0.0)
        cfg.method_kwargs.use_topk = omegaconf_select(cfg, "method_kwargs.use_topk", False)
        cfg.method_kwargs.topk_k = omegaconf_select(cfg, "method_kwargs.topk_k", 0)
        cfg.method_kwargs.gate_type = omegaconf_select(cfg, "method_kwargs.gate_type", "STE")

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]

        if self.use_bayes:
            extra_learnable_params.append({
                "name": "gating_head",
                "params": self.gating_head.parameters(),
                "lr": self.gating_head_lr,
            })

        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats_list = out["feats"]
        h = torch.cat(feats_list)
        z = torch.cat(out["z"])

        # only for imagenet
        # z = F.layer_norm(z, (z.size(1),))

        supported_non_neg_list = [None, 'relu', 'rep_relu', 'gelu', 'sigmoid', 'softplus', 'exp', 'leakyrelu']
        assert self.non_neg in supported_non_neg_list, f"non_neg {self.non_neg} should be one of {supported_non_neg_list}"

        if self.non_neg is None:
            pass # z=z
        if self.non_neg == 'relu': 
            z = F.relu(z)
        if self.non_neg == 'rep_relu': 
            gelu_z = F.gelu(z)
            z = gelu_z - gelu_z.data + F.relu(z).data
        if self.non_neg == 'gelu':
            z = F.gelu(z)
        if self.non_neg == 'sigmoid':
            z = F.sigmoid(z)
        if self.non_neg == 'softplus':
            z = F.softplus(z)
        if self.non_neg == 'exp':
            z = torch.exp(z)
        if self.non_neg == 'leakyrelu':
            z = F.leaky_relu(z)

        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs) 

        if self.use_topk:
            topk_mask = apply_topk_mask(z, self.topk_k)
            z = z * topk_mask

        z = F.normalize(z, dim=-1) 

        nce_loss = torch.tensor(0.0, device=z.device)  

        if self.use_bayes:
            bayesian_kl_loss = torch.tensor(0.0, device=z.device)  

            gating_head_output = self.gating_head(h.detach())

            gating_head_output_soft = gumbel_sigmoid_no_gumbel(gating_head_output, self.sigmoid_temperature, hard=False, gate_threshold=self.gate_threshold)
                
            bayesian_kl_loss = compute_bernoulli_kl_divergence(
                posterior_prob=gating_head_output_soft,
                prior_prob=self.bayesian_prior_prob,
            )

            if self.gate_type == "STE":
                gating_head_output_hard = gumbel_sigmoid_no_gumbel_no_sigmoid(gating_head_output_soft, hard=True, gate_threshold=self.gate_threshold)
                gated_z = z * gating_head_output_hard
            elif self.gate_type == "gumbel_sigmoid":
                gating_head_output_hard = gumbel_sigmoid(gating_head_output, self.sigmoid_temperature, hard=True, gate_threshold=self.gate_threshold)
                gated_z = z * gating_head_output_hard

            normalized_gated_z = F.normalize(gated_z, dim=-1)

            nce_loss = simclr_loss_func(
                normalized_gated_z,
                indexes=indexes,
                temperature=self.temperature,
            )
            
            self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
            self.log("train_bayesian_kl_loss", bayesian_kl_loss, on_epoch=True, sync_dist=True)
        else:
            nce_loss = simclr_loss_func(
                z,
                indexes=indexes,
                temperature=self.temperature,
            )

            self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            _, X, targets = batch
            targets2 = targets.repeat(n_augs)
            stats = {
                # z
                'non_neg_ratio_z': non_neg(z),
                'num_active_dim_z': act_dim(z),
                'sparse_vals_ratio_z': sparsity(z),
                'effective_rank_z': erank(z),
                'orthogonality_z': orthogonality(z),
                'semantic_consistency_z': semantic_consistency(z, targets2),
                'semantic_entropy_sum_activation_z': semantic_entropy_sum_activation(z, targets2),
                'semantic_entropy_mean_activation_z': semantic_entropy_mean_activation(z, targets2),
                'semantic_entropy_frequency_z': semantic_entropy_frequency(z, targets2),
                # h
                'non_neg_ratio_h': non_neg(h),
                'num_active_dim_h': act_dim(h),
                'sparse_vals_ratio_h': sparsity(h),
                'effective_rank_h': erank(h),
                'orthogonality_h': orthogonality(h),
                'semantic_consistency_h': semantic_consistency(h, targets2),
                'semantic_entropy_sum_activation_h': semantic_entropy_sum_activation(h, targets2),
                'semantic_entropy_mean_activation_h': semantic_entropy_mean_activation(h, targets2),
                'semantic_entropy_frequency_h': semantic_entropy_frequency(h, targets2),
            }
            if self.use_bayes:
                stats.update(
                    {
                        'non_neg_ratio_gate': non_neg(gating_head_output),
                        'num_active_dim_gate': act_dim(gating_head_output), 
                        'sparse_vals_ratio_gate': sparsity(gating_head_output),
                        'effective_rank_gate': erank(gating_head_output),
                        'orthogonality_gate': orthogonality(gating_head_output),
                        'semantic_consistency_gate': semantic_consistency(gating_head_output, targets2),
                        'semantic_entropy_sum_activation_gate': semantic_entropy_sum_activation(gating_head_output, targets2),
                        'semantic_entropy_mean_activation_gate': semantic_entropy_mean_activation(gating_head_output, targets2),
                        'semantic_entropy_frequency_gate': semantic_entropy_frequency(gating_head_output, targets2),
                        "gating_head_output_hard_mean": gating_head_output_hard.float().mean(),
                        'non_neg_ratio_z*gate': non_neg(normalized_gated_z),
                        'num_active_dim_z*gate': act_dim(normalized_gated_z),
                        'sparse_vals_ratio_z*gate': sparsity(normalized_gated_z),
                        'effective_rank_z*gate': erank(normalized_gated_z),
                        'orthogonality_z*gate': orthogonality(normalized_gated_z),
                        'semantic_consistency_z*gate': semantic_consistency(normalized_gated_z, targets2),
                        'semantic_entropy_sum_activation_z*gate': semantic_entropy_sum_activation(normalized_gated_z, targets2),
                        'semantic_entropy_mean_activation_z*gate': semantic_entropy_mean_activation(normalized_gated_z, targets2),
                        'semantic_entropy_frequency_z*gate': semantic_entropy_frequency(normalized_gated_z, targets2),
                    }
                )

            for k, v in stats.items():
                self.log(k, v, on_epoch=True, on_step=False, sync_dist=True)

        total_loss = nce_loss + class_loss

        if self.use_bayes:
            total_loss = total_loss + self.bayesian_kl_lambda * bayesian_kl_loss

        return total_loss
