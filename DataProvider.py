from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class NormalizeTransform:
    """Custom transform to normalize to [-1, 1] (picklable for Windows multiprocessing)."""
    def __call__(self, tensor):
        return (tensor * 2) - 1

class DataProvider:
    def __init__(self, batch_size=32, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            NormalizeTransform()
        ])
        self.train, self.test = self.get_data()
    
    def get_data(self):
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )
        
        train = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
        
        test = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
        
        return train, test

    def show_images(self, num_images=16):  
        images, labels = next(iter(self.train))
        images = (images + 1) / 2
        images = images[:num_images]
        labels = labels[:num_images]
        grid_size = int(np.ceil(np.sqrt(num_images)))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']
        
        _, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for idx in range(num_images):
            img = images[idx].permute(1, 2, 0).numpy()
            axes[idx].imshow(img)
            axes[idx].set_title(class_names[labels[idx]])
            axes[idx].axis('off')
        
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()