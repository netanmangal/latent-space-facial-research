import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
import zipfile
from tqdm import tqdm

class CelebADataset(Dataset):
    """
    CelebA dataset loader for facial image experiments.
    
    The CelebA dataset contains over 200,000 celebrity face images with
    40 binary attribute annotations. Perfect for studying facial features
    in latent space.
    """
    
    def __init__(self, root_dir, split='train', transform=None, download=True):
        """
        Args:
            root_dir: Root directory to store the dataset
            split: 'train', 'val', or 'test'
            transform: Image transformations
            download: Whether to download the dataset if not present
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Dataset paths
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
        
        if download and not self._check_dataset_exists():
            print("CelebA dataset not found. Please download manually.")
            print("Instructions:")
            print("1. Go to: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            print("2. Download 'Align&Cropped Images' (img_align_celeba.zip)")
            print("3. Download 'Attribute Annotations' (list_attr_celeba.txt)")
            print(f"4. Extract to: {root_dir}")
            print("\nAlternatively, use the Kaggle dataset:")
            print("https://www.kaggle.com/jessicali9530/celeba-dataset")
            
        # Load image filenames
        self.image_files = self._load_image_files()
        
        # Load attributes if available
        self.attributes = self._load_attributes() if os.path.exists(self.attr_file) else None
        
    def _check_dataset_exists(self):
        """Check if dataset files exist"""
        return os.path.exists(self.img_dir) and len(os.listdir(self.img_dir)) > 0
    
    def _load_image_files(self):
        """Load list of image files based on split"""
        if not os.path.exists(self.img_dir):
            return []
            
        all_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
        # Simple split (you might want to use official CelebA splits)
        if self.split == 'train':
            return all_files[:160000]  # First 160k for training
        elif self.split == 'val':
            return all_files[160000:180000]  # Next 20k for validation
        else:  # test
            return all_files[180000:200000]  # Last 20k for testing
    
    def _load_attributes(self):
        """Load attribute annotations"""
        if not os.path.exists(self.attr_file):
            return None
            
        attributes = {}
        with open(self.attr_file, 'r') as f:
            lines = f.readlines()
            
            # Skip header lines
            num_images = int(lines[0].strip())
            attr_names = lines[1].strip().split()
            
            for i, line in enumerate(lines[2:]):
                parts = line.strip().split()
                filename = parts[0]
                attrs = [int(x) for x in parts[1:]]
                attributes[filename] = dict(zip(attr_names, attrs))
                
        return attributes
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        if idx >= len(self.image_files):
            raise IndexError("Index out of range")
            
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (64, 64), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get attributes if available
        attrs = None
        if self.attributes and img_name in self.attributes:
            attrs = torch.tensor(list(self.attributes[img_name].values()), dtype=torch.float32)
            
        return image, attrs if attrs is not None else torch.zeros(40)


def get_celeba_transforms(image_size=64):
    """
    Get standard transforms for CelebA dataset.
    Includes data augmentation for training.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # Normalize to [0, 1] range (good for sigmoid output)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    return train_transform, test_transform


def create_celeba_dataloaders(root_dir, batch_size=32, image_size=64, num_workers=4):
    """
    Create train, validation, and test dataloaders for CelebA.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, test_transform = get_celeba_transforms(image_size)
    
    # Create datasets
    train_dataset = CelebADataset(
        root_dir, split='train', transform=train_transform
    )
    val_dataset = CelebADataset(
        root_dir, split='val', transform=test_transform
    )
    test_dataset = CelebADataset(
        root_dir, split='test', transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_sample_images(dataloader, num_samples=8, device='cpu'):
    """Get a batch of sample images for testing"""
    for images, _ in dataloader:
        return images[:num_samples].to(device)
    return None


# Alternative: Use a smaller subset for quick testing
class MiniCelebADataset(Dataset):
    """
    A smaller version of CelebA for quick testing and demos.
    Uses only a subset of images.
    """
    
    def __init__(self, root_dir, num_samples=1000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        img_dir = os.path.join(root_dir, 'img_align_celeba')
        if os.path.exists(img_dir):
            all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            self.image_files = all_files[:num_samples]
        else:
            self.image_files = []
            print(f"Warning: Image directory not found at {img_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, 'img_align_celeba', img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (64, 64), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.zeros(1)  # Dummy label


def create_demo_dataloader(root_dir, batch_size=16, image_size=64):
    """Create a small dataloader for quick demos"""
    _, transform = get_celeba_transforms(image_size)
    
    dataset = MiniCelebADataset(root_dir, num_samples=500, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Demo dataset created with {len(dataset)} samples")
    return dataloader