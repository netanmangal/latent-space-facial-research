import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def save_image_grid(images, nrow=8, filename=None, title=None):
    """Save a grid of images"""
    import torchvision.utils as vutils
    
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=16)
    
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.show()

def normalize_images(images):
    """Normalize images to [0, 1] range"""
    return (images - images.min()) / (images.max() - images.min())

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_model_weights(model, filepath, device):
    """Load model weights from file"""
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model weights loaded from {filepath}")
        return True
    except FileNotFoundError:
        print(f"No weights found at {filepath}. Using randomly initialized model.")
        return False

def analyze_latent_space_statistics(model, dataloader, device, num_samples=1000):
    """
    Analyze statistical properties of the latent space.
    This helps understand the distribution and structure.
    """
    model.eval()
    latent_codes = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if len(latent_codes) * images.size(0) >= num_samples:
                break
                
            images = images.to(device)
            
            if hasattr(model, 'encode'):
                if 'VAE' in model.__class__.__name__:
                    mu, logvar = model.encode(images)
                    z = model.reparameterize(mu, logvar)
                else:
                    z = model.encode(images)
            else:
                # For models without explicit encode method
                _, z = model(images)
                
            latent_codes.append(z.cpu().numpy())
    
    latent_codes = np.concatenate(latent_codes, axis=0)[:num_samples]
    
    # Calculate statistics
    stats = {
        'mean': np.mean(latent_codes, axis=0),
        'std': np.std(latent_codes, axis=0),
        'min': np.min(latent_codes, axis=0),
        'max': np.max(latent_codes, axis=0),
        'shape': latent_codes.shape
    }
    
    return latent_codes, stats

def create_latent_traversal_video(model, base_image, dim_idx, device, 
                                num_frames=50, output_path='traversal.gif'):
    """
    Create an animated GIF showing latent space traversal.
    Perfect for presentations!
    """
    model.eval()
    
    # Define traversal range
    values = np.linspace(-3, 3, num_frames)
    
    frames = []
    with torch.no_grad():
        for value in values:
            if hasattr(model, 'traverse_latent_dimension'):
                traversed = model.traverse_latent_dimension(
                    base_image, dim_idx, [value], device
                )
            else:
                # Manual traversal for models without the method
                if 'VAE' in model.__class__.__name__:
                    mu, _ = model.encode(base_image)
                    z = mu.clone()
                    z[0, dim_idx] = value
                else:
                    z = model.encode(base_image)
                    z[0, dim_idx] = value
                traversed = model.decode(z)
            
            # Convert to numpy for saving
            frame = traversed[0].cpu().permute(1, 2, 0).numpy()
            frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
    
    # Save as GIF (requires Pillow)
    from PIL import Image
    frames_pil = [Image.fromarray(frame) for frame in frames]
    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=100,
        loop=0
    )
    
    print(f"Traversal animation saved to {output_path}")

def compare_model_outputs(vae_model, ae_model, test_images, device, save_path=None):
    """
    Compare outputs from VAE and standard autoencoder side by side.
    Demonstrates the difference in reconstruction quality and latent space behavior.
    """
    vae_model.eval()
    ae_model.eval()
    
    with torch.no_grad():
        # Get reconstructions
        vae_recon, _, _, _ = vae_model(test_images)
        ae_recon, _ = ae_model(test_images)
        
        # Create comparison grid
        comparison = torch.cat([
            test_images,
            vae_recon,
            ae_recon
        ], dim=0)
        
        if save_path:
            save_image_grid(
                comparison, 
                nrow=test_images.size(0),
                filename=save_path,
                title="Original | VAE Reconstruction | Autoencoder Reconstruction"
            )
        
        return comparison