import torch
from UNet import UNet
from VDM import VDM
from DataProvider import DataProvider
from Trainer import Trainer
import shutil
import os

def test_smoke_training():
    print("Setting up smoke test...")
    
    config = {
        'device': 'cpu', 
        'lr': 1e-4,
        'save_dir': './test_checkpoints'
    }
    
    # Initialize small model
    unet = UNet(
        embedding_dim=32,
        n_blocks=1,
        n_attention_heads=1,
        norm_groups=4,
        input_channels=3,
        use_fourier_features=False
    )
    
    vdm = VDM(unet, device=config['device'])
    
    # Initialize data provider with small batch
    data = DataProvider(batch_size=4, num_workers=0)
    
    trainer = Trainer(vdm, data, config)
    
    # Limit training to few batches for smoke test
    # We consume the iterator to get a list of batches
    original_train_loader = trainer.data_provider.train
    trainer.data_provider.train = list(iter(original_train_loader))[:2]
    
    print("Starting training loop...")
    try:
        trainer.train(epochs=1)
        print("Smoke test passed successfully!")
    except Exception as e:
        print(f"Smoke test failed: {e}")
        raise e
    finally:
        if os.path.exists(config['save_dir']):
            shutil.rmtree(config['save_dir'])

if __name__ == "__main__":
    test_smoke_training()
