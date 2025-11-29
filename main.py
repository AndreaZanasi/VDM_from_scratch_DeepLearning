import torch
from unet import UNet
from vdm import VDM
from data_provider import DataProvider
from trainer import Trainer
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import argparse
import wandb

CONFIG = {
        'use_wandb': True,
        'wandb_project': 'VDM-CIFAR10',
        'wandb_run_name': 'vdm_experiment_2',
        'wandb_entity': 'DL_group99',
        'wandb_log_every': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 2e-4,
        'batch_size': 32,
        'epochs': 100,
        'save_dir': './checkpoints',
        'embedding_dim': 64,
        'n_blocks': 8,
        'n_attention_heads': 8,
        'dropout_prob': 0.1,
        'norm_groups': 32,
        'input_channels': 3,
        'gamma_min': -13.3,
        'gamma_max': 5.0,
        'vocab_size': 256,
        'T': 1000,
        'use_fourier_features': True,
        'attention_everywhere': True,
        'num_samples': 64,
        'sample_path': './samples',
        'learned_schedule': True,
        'best_model_path': 'learned_best_model.pt',
        'last_model_path': 'learned_last_model.pt',
    }

def init_models():
    unet = UNet(
        embedding_dim=CONFIG['embedding_dim'],
        n_blocks=CONFIG['n_blocks'],
        n_attention_heads=CONFIG['n_attention_heads'],
        dropout_prob=CONFIG['dropout_prob'],
        norm_groups=CONFIG['norm_groups'],
        input_channels=CONFIG['input_channels'],
        use_fourier_features=CONFIG['use_fourier_features'],
        attention_everywhere=CONFIG['attention_everywhere'],
        gamma_min=CONFIG['gamma_min'],
        gamma_max=CONFIG['gamma_max']
    ).to(CONFIG['device'])

    vdm = VDM(
        model=unet,
        gamma_min=CONFIG['gamma_min'],
        gamma_max=CONFIG['gamma_max'],
        vocab_size=CONFIG['vocab_size'],
        T=CONFIG['T'],
        device=CONFIG['device'],
        learned_schedule=CONFIG['learned_schedule']
    ).to(CONFIG['device'])

    return vdm, unet

def train(vdm, unet):
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    data_provider = DataProvider(batch_size=CONFIG['batch_size'], num_workers=4)
    
    # Initialize WandB and log config
    
    if CONFIG.get("use_wandb", False):
        wandb.init(
            project=CONFIG.get("wandb_project", "vdm"),
            entity=CONFIG.get("wandb_entity", None),
            name=CONFIG.get("wandb_run_name", None),
            config=CONFIG,
        )

        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params
        })
    
    trainer = Trainer(vdm, data_provider, CONFIG)

    print(f"\n{'='*70}")
    print("VDM Training - CIFAR-10")
    print(f"{'='*70}")
    print(f"Device: {CONFIG['device']}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("\nStarting training...")

    checkpoint_path = os.path.join(CONFIG['save_dir'], CONFIG['best_model_path'])
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)

    trainer.train(epochs=CONFIG['epochs'])
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Final Loss: {trainer.train_losses[-1]:.4f} BPD")
    print(f"Best Loss: {trainer.best_loss:.4f} BPD")
    print(f"{'='*70}\n")

    sample(vdm)
    
    if CONFIG.get("use_wandb", False):
        artifact = wandb.Artifact(
        name="vdm-best-model",
        type="model",
        description="Best VDM checkpoint based on training loss"
        )

        best_model_path = os.path.join(
            CONFIG["save_dir"], CONFIG["best_model_path"]
        )
        
        if os.path.exists(best_model_path):
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
        wandb.finish()


def sample(vdm):
    print("Sampling...")
    checkpoint = torch.load(f'{CONFIG["save_dir"]}/{CONFIG["best_model_path"]}')
    vdm.load_state_dict(checkpoint['model_state_dict'])
    vdm.eval()
    
    with torch.no_grad():
        samples = vdm.reverse_diffusion.sample(
            batch_size=CONFIG['num_samples'], 
            shape=(3, 32, 32), 
            T=CONFIG['T'], 
            device=CONFIG['device']
        )

    grid = make_grid(samples, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f"Generated Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['sample_path'], "generated_samples.png"), dpi=150, bbox_inches='tight')
    print(f"✓ Samples saved to {os.path.join(CONFIG['sample_path'], 'generated_samples.png')}")
    
    # Log samples to WandB
    if CONFIG.get("use_wandb", False):
        wandb.log({
            "samples": wandb.Image(
                os.path.join(CONFIG['sample_path'], "generated_samples.png"),
                caption="VDM Generated Samples"
            )
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], default='train',
                        help="Mode to run the script in: 'train' to train the model, 'sample' to generate samples.")
    args = parser.parse_args()
    vdm, unet = init_models()

    if args.mode == 'train':
        train(vdm, unet)
    else:
        sample(vdm)