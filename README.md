# Bayesian Gated Non-Negative Contrastive Learning

This is the official code repository for the ICML 2026 paper "Bayesian Gated Non-Negative Contrastive Learning".

## Pretrain

```bash
python main_pretrain.py \
  --config-path scripts/pretrain/cifar \
  --config-name ncl.yaml \
  name=bayesncl-resnet18-cifar100 \
  data.dataset=cifar100 \
  method_kwargs.use_bayes=True \
  method_kwargs.bayesian_prior_prob=0.8 \
  method_kwargs.gate_threshold=0.5 \
  method_kwargs.bayesian_kl_lambda=0.00003 \
  method_kwargs.sigmoid_temperature=1.0 \
  optimizer.gating_head_lr=0.1
```

## Linear Probe

```bash
python main_linear.py \
  --config-path scripts/linear/cifar \
  --config-name simclr.yaml \
  name=linear-cifar100 \
  data.dataset=cifar100 \
  pretrained_feature_extractor=/path/to/pretrained.ckpt
```

## Eval

```bash
python main_eval.py \
  --config-path scripts/eval/cifar \
  --config-name ncl.yaml \
  name=eval-cifar100 \
  data.dataset=cifar100 \
  resume_from_checkpoint=/path/to/pretrained.ckpt
```
