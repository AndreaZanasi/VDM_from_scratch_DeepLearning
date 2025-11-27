import torch
from unet import UNet
from vdm import VDM
from DataProvider import DataProvider
from Trainer import Trainer
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import time

def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 2e-4,
        'batch_size': 64,
        'epochs': 500,
        'save_dir': './checkpoints',
        'embedding_dim': 64,
        'n_blocks': 4,
        'n_attention_heads': 4,
        'dropout_prob': 0.1,
        'norm_groups': 32,
        'input_channels': 3,
        'gamma_min': -13.3,
        'gamma_max': 5.0,
        'vocab_size': 256,
        'T': 1000,
        'use_fourier_features': True,
        'attention_everywhere': True
    }
    
    print(f"\n{'='*70}")
    print("VDM Training - CIFAR-10")
    print(f"{'='*70}")
    print(f"Device: {config['device']}")
    
    # Initialize Model
    unet = UNet(
        embedding_dim=config['embedding_dim'],
        n_blocks=config['n_blocks'],
        n_attention_heads=config['n_attention_heads'],
        dropout_prob=config['dropout_prob'],
        norm_groups=config['norm_groups'],
        input_channels=config['input_channels'],
        use_fourier_features=config['use_fourier_features'],
        attention_everywhere=config['attention_everywhere'],
        gamma_min=config['gamma_min'],
        gamma_max=config['gamma_max']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    vdm = VDM(
        model=unet,
        gamma_min=config['gamma_min'],
        gamma_max=config['gamma_max'],
        vocab_size=config['vocab_size'],
        T=config['T'],
        device=config['device'],
        learned_schedule=False  
    )
    
    # Initialize Data
    data_provider = DataProvider(batch_size=config['batch_size'], num_workers=4)
    
    # Initialize Trainer
    trainer = Trainer(vdm, data_provider, config)
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        trainer = Trainer(vdm, data_provider, config)
        checkpoint_path = os.path.join(config['save_dir'], "best_model.pt")
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
        trainer.train(epochs=config['epochs'])
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current state...")
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Total Time: {hours}h {minutes}m")
    print(f"Final Loss: {trainer.train_losses[-1]:.4f} BPD")
    print(f"Best Loss: {trainer.best_loss:.4f} BPD")
    print(f"Target (Paper): 2.67 BPD")
    print(f"{'='*70}\n")
    
    # Load best model for sampling
    print("Loading best model for sampling...")
    checkpoint = torch.load(f'{config["save_dir"]}/best_model.pt')
    vdm.load_state_dict(checkpoint['model_state_dict'])
    
    # Sample
    print("Generating samples...")
    vdm.eval()
    
    with torch.no_grad():
        # Generate samples with progress
        samples = vdm.reverse_diffusion.sample(
            batch_size=64, 
            image_shape=(3, 32, 32), 
            T=config['T'], 
            device=config['device']
        )
    
    # Plot and Save
    print("Saving samples...")
    save_path = 'generated_samples.png'
    
    # Create a grid of images
    grid = make_grid(samples, nrow=8, padding=2, normalize=False)
    
    # Convert to numpy for matplotlib (C, H, W) -> (H, W, C)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f"Generated Samples (Loss: {trainer.best_loss:.3f} BPD)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Samples saved to {save_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Training Loss', linewidth=2)
    plt.axhline(y=2.67, color='r', linestyle='--', label='Paper Baseline (T=1000)', alpha=0.7)
    plt.axhline(y=trainer.best_loss, color='g', linestyle='--', label=f'Best Loss ({trainer.best_loss:.3f})', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BPD)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training curve saved to training_curve.png")
    
    print(f"\n{'='*70}")
    print("All done! 🎉")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()