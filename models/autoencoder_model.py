import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardAutoencoder(nn.Module):
    """
    Standard Autoencoder with discrete latent space.
    
    This implementation demonstrates the limitations of discrete latent spaces:
    - Random points in latent space don't generate meaningful images
    - No smooth interpolation between points
    - Gaps and discontinuities in the representation space
    """
    
    def __init__(self, latent_dim=128, image_channels=3, image_size=64):
        super(StandardAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        
        # Encoder - same structure as VAE but deterministic
        self.encoder = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(image_channels, 32, 4, 2, 1),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),              # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),             # (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),            # (256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                            # (256 * 4 * 4 = 4096)
            nn.Linear(4096, latent_dim)              # Direct mapping to latent space
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.Unflatten(1, (256, 4, 4)),           # (256, 4, 4)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1),  # (3, 64, 64)
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """Encode input to latent representation (deterministic)"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent vector to image"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
    def sample(self, num_samples, device):
        """
        Sample random images from the latent space.
        WARNING: In standard autoencoders, random points often generate
        meaningless or distorted images due to discrete latent space.
        """
        # Sample from the same distribution as training data latent codes
        # This often produces poor results due to gaps in latent space
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def interpolate(self, x1, x2, num_steps=10):
        """
        Interpolate between two images in latent space.
        This often produces unrealistic intermediate images due to
        discontinuities in the latent space.
        """
        with torch.no_grad():
            z1 = self.encode(x1)
            z2 = self.encode(x2)
            
            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, num_steps).to(x1.device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)
                
            return torch.cat(interpolations, dim=0)
    
    def traverse_latent_dimension(self, base_image, dim_idx, values, device):
        """
        Traverse a specific latent dimension while keeping others fixed.
        Results often show abrupt changes or artifacts due to discrete nature.
        """
        with torch.no_grad():
            z = self.encode(base_image)
            traversals = []
            
            for value in values:
                z_modified = z.clone()
                z_modified[0, dim_idx] = value
                x_traversal = self.decode(z_modified)
                traversals.append(x_traversal)
                
            return torch.cat(traversals, dim=0)


def autoencoder_loss_function(recon_x, x):
    """
    Standard autoencoder loss function (only reconstruction loss).
    
    Args:
        recon_x: Reconstructed images
        x: Original images
    """
    # Only reconstruction loss - no regularization of latent space
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return recon_loss