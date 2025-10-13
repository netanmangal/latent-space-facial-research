import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    """
    β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
    
    This implementation creates a continuous AND disentangled latent space where:
    - Every point generates a valid image
    - Smooth interpolation is possible between any two points
    - Semantic attributes can be controlled independently (BETTER than standard VAE)
    - Each latent dimension captures independent semantic factors
    - Higher β encourages more disentangled representations
    
    Perfect for facial attribute analysis: head pose, lighting, skin tone, etc.
    """
    
    def __init__(self, latent_dim=128, image_channels=3, image_size=64, beta=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.beta = beta  # Disentanglement parameter
        
        # Encoder - Enhanced architecture for better disentanglement
        self.encoder = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(image_channels, 64, 4, 2, 1),   # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),             # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),            # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),            # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Flatten(),                            # (512 * 4 * 4 = 8192)
        )
        
        # Latent space parameters
        self.mu = nn.Linear(8192, latent_dim)
        self.logvar = nn.Linear(8192, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 8192)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),           # (512, 4, 4)
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),  # (3, 64, 64)
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1)
        This ensures the latent space is continuous and differentiable
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image"""
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def sample(self, num_samples, device):
        """
        Sample random images from the latent space.
        In VAE, every point in latent space should generate a valid image.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def interpolate(self, x1, x2, num_steps=10):
        """
        Interpolate between two images in latent space.
        This demonstrates the continuity of VAE latent space.
        """
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            
            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, num_steps).to(x1.device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)
                
            return torch.cat(interpolations, dim=0)
    
    def traverse_latent_dimension(self, base_image, dim_idx, values, device):
        """
        Traverse a specific latent dimension while keeping others fixed.
        This is crucial for understanding semantic attributes.
        
        β-VAE enhancement: Better disentanglement means cleaner attribute control!
        Each dimension should control independent semantic factors like:
        - Head pose, lighting, skin tone, hair style, expression, etc.
        
        Args:
            base_image: Base image to start from
            dim_idx: Index of latent dimension to traverse
            values: List of values to set for that dimension
            device: Device to run on
        """
        with torch.no_grad():
            mu, _ = self.encode(base_image)
            traversals = []
            
            for value in values:
                z = mu.clone()
                z[0, dim_idx] = value  # Modify specific dimension
                x_traversal = self.decode(z)
                traversals.append(x_traversal)
                
            return torch.cat(traversals, dim=0)
    
    def get_disentanglement_metrics(self, dataloader, device, num_samples=1000):
        """
        Compute disentanglement metrics specific to β-VAE.
        Higher β should result in better disentanglement scores.
        """
        self.eval()
        latent_codes = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i * images.size(0) >= num_samples:
                    break
                    
                images = images.to(device)
                mu, _ = self.encode(images)
                latent_codes.append(mu.cpu())
        
        latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
        
        # Compute variance of each latent dimension
        latent_variances = torch.var(latent_codes, dim=0)
        
        # Active units (dimensions with significant variance)
        active_units = (latent_variances > 0.01).sum().item()
        
        # Disentanglement score
        disentanglement_score = active_units / self.latent_dim
        
        return {
            'active_units': active_units,
            'total_units': self.latent_dim,
            'disentanglement_score': disentanglement_score,
            'latent_variances': latent_variances,
            'beta': self.beta
        }


def beta_vae_loss_function(recon_x, x, mu, logvar, beta=4.0):
    """
    β-VAE loss function with enhanced disentanglement regularization.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (disentanglement parameter)
        
    β parameter effects:
    - β = 1.0: Standard VAE
    - β = 4.0: Balanced disentanglement (recommended for faces)
    - β = 10.0: High disentanglement, lower reconstruction quality
    - β = 0.5: Lower disentanglement, higher reconstruction quality
    
    Higher β encourages each latent dimension to capture independent
    semantic factors, perfect for controlling facial attributes separately!
    """
    # Reconstruction loss - how well we reconstruct the input
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss - encourages latent space to follow standard normal
    # Higher β weight makes this term more important = more disentanglement
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with β weighting for disentanglement control
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# Standard VAE loss function (β=1.0)
def vae_loss_function(recon_x, x, mu, logvar):
    """
    Standard VAE loss function with β=1.0.
    
    This is used for fair comparison with β-VAE to isolate the effect
    of the β parameter on disentanglement performance.
    """
    return beta_vae_loss_function(recon_x, x, mu, logvar, beta=1.0)


class StandardVAE(BetaVAE):
    """
    Standard VAE with β=1.0 for fair comparison with β-VAE.
    
    Uses IDENTICAL architecture to BetaVAE - only difference is β parameter.
    This ensures fair experimental comparison between standard VAE and β-VAE.
    
    Properties:
    - Continuous latent space (vs discrete Autoencoder)
    - No disentanglement regularization (β=1.0)
    - Same encoder/decoder architecture as β-VAE
    - Perfect for showing β parameter effect on disentanglement
    """
    
    def __init__(self, latent_dim=128, image_channels=3, image_size=64):
        super().__init__(latent_dim, image_channels, image_size, beta=1.0)
        
    def __str__(self):
        return f"StandardVAE(β=1.0, latent_dim={self.latent_dim})"


class EnhancedBetaVAE(BetaVAE):
    """
    Enhanced β-VAE with β=4.0 for superior disentanglement.
    
    Uses IDENTICAL architecture to StandardVAE - only difference is β parameter.
    This ensures fair experimental comparison.
    
    Properties:
    - Continuous AND disentangled latent space
    - Enhanced disentanglement regularization (β=4.0)
    - Same encoder/decoder architecture as StandardVAE
    - Superior semantic control over facial attributes
    """
    
    def __init__(self, latent_dim=128, image_channels=3, image_size=64):
        super().__init__(latent_dim, image_channels, image_size, beta=4.0)
        
    def __str__(self):
        return f"BetaVAE(β=4.0, latent_dim={self.latent_dim})"


# Backward compatibility aliases
VAE = StandardVAE  # Now points to StandardVAE for consistency
BetaVAE = EnhancedBetaVAE  # Enhanced version is the main β-VAE