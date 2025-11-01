"""
Models package for latent space facial research.

This package contains:
- vae_model.py: VAE implementations (StandardVAE, BetaVAE)
- autoencoder_model.py: Standard autoencoder implementation
- utils.py: Utility functions for model operations
"""

# Import key classes and functions for easy access
from .vae_model import StandardVAE, BetaVAE, beta_vae_loss_function, vae_loss_function
from .autoencoder_model import StandardAutoencoder, autoencoder_loss_function
from .utils import get_device, set_seed, save_image_grid, normalize_images

__all__ = [
    'StandardVAE', 'BetaVAE', 'StandardAutoencoder',
    'beta_vae_loss_function', 'vae_loss_function', 'autoencoder_loss_function',
    'get_device', 'set_seed', 'save_image_grid', 'normalize_images'
]