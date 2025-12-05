# Variational Diffusion Model (VDM) — CIFAR-10

This repository contains the implementation and training code for a **Variational Diffusion Model (VDM)** trained on **CIFAR-10**. It includes functionality for:

- Training from scratch
- Sampling from pretrained models
- Full reproducibility via **Weights & Biases (W&B) artifacts**

---

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download a pretrained model from the W&B reports (see Artifacts below) and place the checkpoint in the `./checkpoints` folder.
3. Train or sample using the commands below.

## Artifacts

Pretrained weights are provided via W&B reports:

* **Learned scheduler**: [Download here](https://wandb.ai/DL_group99/VDM-CIFAR10/reports/Model-checkpoint-learned-scheduler--VmlldzoxNTI4MzE5NQ?accessToken=ez34za31ulodcicshn81ekigkw7ecda2fjhslmvpnfwh1284mnpf4qyjzqj82263)
* **Fixed linear scheduler**: [Download here](https://wandb.ai/DL_group99/VDM-CIFAR10/reports/Model-checkpoints-fixed-linear-scheduler--VmlldzoxNTI4MzE4NA?accessToken=dp3ozzmfhnv8u97nw7ibxiz5jf5mnq1fuz0gzbjo4ggpgkbqdfjofs3eipnvix7p)

**Note**: Place the downloaded checkpoint in:

```bash
./checkpoints/
```

## Training

To train the model from scratch:

```bash
python main.py --mode train
```

This will:

* Train the VDM on CIFAR-10
* Log metrics to W&B
* Save checkpoints to `./checkpoints/`
* Generate samples periodically

## Sampling

To generate samples from a pretrained model:

1. Make sure the checkpoint is in `./checkpoints/`.
2. In `main.py`, adjust the following in `CONFIG` according to the model you are using:
   * `use_learned_scheduler`
   * `best_model_path`
3. Run:

```bash
python main.py --mode sample
```

Generated samples will be saved to:

```bash
./samples/generated_samples.png
```

## Configuration

All hyperparameters and settings are defined in the `CONFIG` dictionary in `main.py`. This includes:

* Learning rate, batch size, and number of epochs
* Diffusion steps and scheduler type (learned vs linear)
* Model architecture parameters
* W&B logging options

## Notes

* Ensure the scheduler type (`use_learned_scheduler`) matches the checkpoint you are using.
* If you encounter model loading errors, double-check the checkpoint filename and path.
* `requirements.txt` contains all necessary Python dependencies to run training and sampling.
