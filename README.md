# Variational Diffusion Model (VDM) — CIFAR-10

This repository contains the implementation and training code for a **Variational Diffusion Model (VDM)** trained on **CIFAR-10**, including sampling and full reproducibility via Weights & Biases (W&B) artifacts.

---

## Training

To train the model from scratch:

```bash
python main.py --mode train
```

## Sampling
To generate samples with a model from `./checkpoints` folder, edit the set the "use_learned_scheduler" in CONFIG in main.py according to the model.
Then run:
```bash
python main.py --mode sample
```
## Artifacts
[Model - learned scheduler] (https://wandb.ai/DL_group99/VDM-CIFAR10/reports/Model-checkpoint-learned-scheduler--VmlldzoxNTI4MzE5NQ?accessToken=ez34za31ulodcicshn81ekigkw7ecda2fjhslmvpnfwh1284mnpf4qyjzqj82263)
[Model - fixed linear scheduler] (https://wandb.ai/DL_group99/VDM-CIFAR10/reports/Model-checkpoints-fixed-linear-scheduler--VmlldzoxNTI4MzE4NA?accessToken=dp3ozzmfhnv8u97nw7ibxiz5jf5mnq1fuz0gzbjo4ggpgkbqdfjofs3eipnvix7p)

