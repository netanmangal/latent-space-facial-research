"""
Data package for latent space facial research.

This package contains:
- celeba_loader.py: CelebA dataset loading utilities
"""

from .celeba_loader import create_demo_dataloader, get_sample_images, CelebADataset

__all__ = ['create_demo_dataloader', 'get_sample_images', 'CelebADataset']