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

import inspect
import os

import torchvision.utils as tv
import hydra 
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf 
from pytorch_lightning import seed_everything 
# from pytorch_lightning.loggers import WandbLogger
import random
from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous, omegaconf_select
from solo.methods.bayesncl_helper_functions import gumbel_sigmoid_no_gumbel, gumbel_sigmoid_no_gumbel_no_sigmoid, apply_topk_mask
from solo.methods.bayesncl_eval_helper_functions import act_dim, sparsity, orthogonality, semantic_consistency, semantic_entropy_sum_activation, semantic_entropy_mean_activation, semantic_entropy_frequency

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


def inference(model, loader, device=torch.device('cuda'), use_soft_gating=False):
    feature_vector = []
    labels_vector = []
    
    has_gating_head = hasattr(model, 'gating_head')
    
    use_topk = hasattr(model, 'use_topk') and model.use_topk
    if use_topk:
        topk_k = model.topk_k if hasattr(model, 'topk_k') else 0
    
    for step, (x, y) in enumerate(loader):
        x = x.cuda()

        # get encoding
        with torch.no_grad():
            out = model(x)
            
            if has_gating_head:
                h = out['feats'] 
                z = out['z']     
            else:
                z = out['z']
            
            non_neg = getattr(model, 'non_neg', None)
            if non_neg == 'relu':
                z = F.relu(z)
            
            if use_topk and topk_k > 0:
                topk_mask = apply_topk_mask(z, topk_k)
                z = z * topk_mask
            
            if has_gating_head:
                z = F.normalize(z, dim=-1)
                gating_head_output = model.gating_head(h)
                gating_head_output_soft = gumbel_sigmoid_no_gumbel(
                    gating_head_output, 
                    temperature=1.0, 
                    hard=False, 
                    gate_threshold=0.5
                )
                
                if use_soft_gating:
                    gated_z = z * gating_head_output_soft
                else:
                    gating_head_output_hard = gumbel_sigmoid_no_gumbel_no_sigmoid(
                        gating_head_output_soft, 
                        hard=True, 
                        gate_threshold=0.5
                    )
                    gated_z = z * gating_head_output_hard
                
                normalized_gated_z = F.normalize(gated_z, dim=-1)
                
                features = normalized_gated_z
            else:
                features = F.normalize(z, dim=-1)

        feature_vector.append(features.data.to(device))
        labels_vector.append(y.to(device))

    feature_vector = torch.cat(feature_vector)
    labels_vector = torch.cat(labels_vector)
    return feature_vector, labels_vector


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    train_loader, val_loader = prepare_data_classification(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        )
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset


    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    
    # Set seed for reproducibility
    seed_everything(cfg.seed)
    
    ckpt = torch.load(cfg.resume_from_checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    def _infer_proj_output_dim_from_state_dict(state_dict: dict) -> int:
        if "projector.2.weight" in state_dict:
            return int(state_dict["projector.2.weight"].shape[0])

        candidates = []
        for k, v in state_dict.items():
            if not (isinstance(k, str) and k.startswith("projector.") and k.endswith(".weight")):
                continue
            if hasattr(v, "shape") and len(v.shape) == 2:
                candidates.append((k, int(v.shape[0])))
        if not candidates:
            raise KeyError("checkpoint 中未找到 projector 权重，无法推断 proj_output_dim")
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][1]

    try:
        inferred_proj_dim = _infer_proj_output_dim_from_state_dict(state_dict)
        cfg_dim = int(omegaconf_select(cfg, "method_kwargs.proj_output_dim", inferred_proj_dim))
        if cfg_dim != inferred_proj_dim:
            cfg.method_kwargs.proj_output_dim = inferred_proj_dim
    except Exception as e:
        print(f"[自适应维度] 推断 proj_output_dim 失败，继续使用 cfg 中的设置。原因: {e}")

    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    

    has_gating_head_in_ckpt = any(key.startswith("gating_head.") for key in state_dict.keys())
    has_gating_head_in_model = hasattr(model, 'gating_head')
    

    if has_gating_head_in_ckpt:

        need_rebuild = False
        if not has_gating_head_in_model:
            need_rebuild = True
        else:

            try:

                test_dict = {k: v for k, v in state_dict.items() if k.startswith("gating_head.")}
                model_state = model.state_dict()
                for k, v in test_dict.items():
                    if k in model_state:
                        if model_state[k].shape != v.shape:
                            need_rebuild = True
                            break
            except Exception:
                need_rebuild = True
        
        if need_rebuild:

            gating_head_keys = [key for key in state_dict.keys() if key.startswith("gating_head.") and key.endswith(".weight")]
            

            layer_indices = []
            for key in gating_head_keys:

                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        layer_indices.append(idx)
                    except ValueError:
                        continue
            
            layer_indices = sorted(set(layer_indices))
            

            if len(layer_indices) == 1:

                gating_head_0_weight = state_dict["gating_head.0.weight"]
                features_dim = gating_head_0_weight.shape[1]
                proj_output_dim = gating_head_0_weight.shape[0]
                
                model.gating_head = nn.Sequential(
                    nn.Linear(features_dim, proj_output_dim),
                )

                
            elif len(layer_indices) == 2:

                gating_head_0_weight = state_dict["gating_head.0.weight"]
                gating_head_2_weight = state_dict["gating_head.2.weight"]
                
                features_dim = gating_head_0_weight.shape[1]
                proj_hidden_dim = gating_head_0_weight.shape[0]
                proj_output_dim = gating_head_2_weight.shape[0]
                
                model.gating_head = nn.Sequential(
                    nn.Linear(features_dim, proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(proj_hidden_dim, proj_output_dim),
                )

                
            elif len(layer_indices) == 3:

                gating_head_0_weight = state_dict["gating_head.0.weight"]
                gating_head_2_weight = state_dict["gating_head.2.weight"]
                gating_head_4_weight = state_dict["gating_head.4.weight"]
                
                features_dim = gating_head_0_weight.shape[1]
                proj_hidden_dim = gating_head_0_weight.shape[0]
                proj_hidden_dim_half = gating_head_2_weight.shape[0]
                proj_output_dim = gating_head_4_weight.shape[0]
                
                model.gating_head = nn.Sequential(
                    nn.Linear(features_dim, proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(proj_hidden_dim, proj_hidden_dim_half),
                    nn.ReLU(),
                    nn.Linear(proj_hidden_dim_half, proj_output_dim),
                )

            else:
                raise ValueError(f"不支持的 gating_head 层数: {len(layer_indices)} 层 (层索引: {layer_indices})")
        
        if not hasattr(model, 'sigmoid_temperature'):
            model.sigmoid_temperature = 1.0
        if not hasattr(model, 'gate_threshold'):
            model.gate_threshold = 0.5
        if not hasattr(model, 'use_topk'):
            model.use_topk = omegaconf_select(cfg, "method_kwargs.use_topk", False)
        if not hasattr(model, 'topk_k'):
            model.topk_k = omegaconf_select(cfg, "method_kwargs.topk_k", 0)
        if not hasattr(model, 'non_neg'):
            model.non_neg = omegaconf_select(cfg, "method_kwargs.non_neg", None)

    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    

    use_soft_gating = omegaconf_select(cfg, "eval.use_soft_gating", False)
    print(f"use_soft_gating: {use_soft_gating}")

    val_features,val_labels = inference(model, val_loader, use_soft_gating=use_soft_gating)


    act_dim_val = act_dim(val_features)
    print('act_dim:', act_dim_val.item())


    sparsity_val = sparsity(val_features)
    print('sparsity:', sparsity_val.item())
    

    cluster_acc_val = semantic_consistency(val_features, val_labels)
    print('[cluster acc] mean {:.4f}'.format(cluster_acc_val.item() * 100))
    

    ent_sum_activation = semantic_entropy_sum_activation(val_features, val_labels)
    print('[semantic entropy sum activation] mean {:.4f}'.format(ent_sum_activation.item()))

    ent_mean_activation = semantic_entropy_mean_activation(val_features, val_labels)
    print('[semantic entropy mean activation] mean {:.4f}'.format(ent_mean_activation.item()))
    

    ent_frequency = semantic_entropy_frequency(val_features, val_labels)
    print('[semantic entropy frequency] mean {:.4f}'.format(ent_frequency.item()))
    

    ortho_err = orthogonality(val_features)
    ortho_err_mean = ortho_err.mean().item()
    ortho_err_median = ortho_err.median().item()
    print('disentanglement mean {:.3f} median {:.3f}'.format(ortho_err_mean, ortho_err_median))


    def retrieval(val_features, val_labels):

        num_samples = val_features.size(0)
        dims = val_features.size(1)

        f = F.normalize(val_features, p=2, dim=1)

        feature_sum = torch.sum(f, dim=0)
        _, index = torch.sort(feature_sum, descending=True)

        print(f"{'Dims':>5} | {'P@1':>8} | {'P@3':>8} | {'P@5':>8} | {'P@10':>8}")
        print("-" * 40)


        for k in range(0, 512 + 32, 32): 

            f_sub = f[:, index[0:k]]

            f_sub = F.normalize(f_sub, p=2, dim=1)


            sim = torch.mm(f_sub, f_sub.t())

            _, indices = torch.topk(sim, k=11, dim=1)
            

            neighbor_indices = indices[:, 1:]


            query_labels = val_labels.view(-1, 1) 
            neighbor_labels = val_labels[neighbor_indices] 
            

            correct_mask = (neighbor_labels == query_labels)


            p1 = correct_mask[:, :1].sum().float() / num_samples
            p3 = correct_mask[:, :3].sum().float() / (num_samples * 3)
            p5 = correct_mask[:, :5].sum().float() / (num_samples * 5)
            p10 = correct_mask[:, :10].sum().float() / (num_samples * 10)

            print(f"{k:5d} | {p1:8.4f} | {p3:8.4f} | {p5:8.4f} | {p10:8.4f}")

    retrieval(val_features,val_labels)

    # def plot_tsne(data, labels, n_classes, save_dir='figs', file_name='simclr', y_name='Class'):

    #     from sklearn.manifold import TSNE # type: ignore
    #     from matplotlib import ft2font
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns # type: ignore
    #     import pandas as pd # type: ignore
    #     """ Input:
    #             - model weights to fit into t-SNE
    #             - labels (no one hot encode)
    #             - num_classes
    #     """
    #     if isinstance(data, torch.Tensor):
    #         data = data.cpu().numpy()
    #     if isinstance(labels, torch.Tensor):
    #         labels = labels.cpu().numpy()
        
    #     n_components = 2
    #     if n_classes == 10:
    #         platte = sns.color_palette(n_colors=n_classes)
    #     else:
    #         platte = sns.color_palette("Set2", n_colors=n_classes)

    #     tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=0)
    #     tsne_res = tsne.fit_transform(data)

    #     v = pd.DataFrame(data,columns=[str(i) for i in range(data.shape[1])])
    #     v[y_name] = labels
    #     v['label'] = v[y_name].apply(lambda i: str(i))
    #     v["t1"] = tsne_res[:,0]
    #     v["t2"] = tsne_res[:,1]

    #     sns.scatterplot(
    #         x="t1", y="t2",
    #         hue=y_name,
    #         palette=platte,
    #         legend=True,
    #         data=v,
    #     )
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.xlabel('')
    #     plt.ylabel('')
    #     os.makedirs(save_dir, exist_ok=True)
    #     plt.savefig(os.path.join(save_dir, file_name+'_t-SNE.png'))
    
    # plot_tsne(val_features, val_labels, n_classes=len(list(val_labels.unique())))

if __name__ == "__main__":
    main()
