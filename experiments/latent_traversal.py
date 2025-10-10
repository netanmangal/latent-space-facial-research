"""
Latent Space Traversal Experiments

This script performs systematic traversal of latent dimensions to identify
semantic attributes like head pose, lighting, skin tone, etc.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae_model import VAE
from models.autoencoder_model import StandardAutoencoder
from models.utils import get_device, save_image_grid, load_model_weights
from data.celeba_loader import create_demo_dataloader, get_sample_images


def systematic_latent_traversal(model, base_images, device, save_dir='outputs/traversals'):
    """
    Systematically traverse all latent dimensions to identify semantic attributes.
    This is crucial for understanding what the model has learned.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = 'VAE' if 'VAE' in model.__class__.__name__ else 'Autoencoder'
    print(f"Performing systematic traversal for {model_name}...")
    
    latent_dim = model.latent_dim
    num_base_images = base_images.size(0)
    
    # Define traversal range
    traversal_range = torch.linspace(-3, 3, 11).to(device)
    
    # Store results for analysis
    traversal_results = {}
    
    for img_idx in range(min(num_base_images, 3)):  # Analyze first 3 images
        base_image = base_images[img_idx:img_idx+1]
        
        print(f"Analyzing base image {img_idx + 1}...")
        
        # Traverse each latent dimension
        for dim_idx in tqdm(range(latent_dim), desc=f"Dimensions for image {img_idx+1}"):
            
            traversals = []
            
            with torch.no_grad():
                if hasattr(model, 'traverse_latent_dimension'):
                    # Use model's built-in method
                    for value in traversal_range:
                        traversed = model.traverse_latent_dimension(
                            base_image, dim_idx, [value], device
                        )
                        traversals.append(traversed)
                else:
                    # Manual traversal
                    if 'VAE' in model.__class__.__name__:
                        mu, _ = model.encode(base_image)
                        base_z = mu
                    else:
                        base_z = model.encode(base_image)
                    
                    for value in traversal_range:
                        z_modified = base_z.clone()
                        z_modified[0, dim_idx] = value
                        traversed = model.decode(z_modified)
                        traversals.append(traversed)
            
            # Concatenate all traversals
            traversal_grid = torch.cat(traversals, dim=0)
            
            # Save traversal for this dimension
            save_path = f'{save_dir}/{model_name.lower()}_img{img_idx}_dim{dim_idx:03d}.png'
            save_image_grid(
                traversal_grid,
                nrow=len(traversal_range),
                filename=save_path,
                title=f'{model_name} - Image {img_idx+1} - Dimension {dim_idx}'
            )
            
            # Store for analysis (optional - could analyze variance, etc.)
            traversal_results[f'img{img_idx}_dim{dim_idx}'] = traversal_grid.cpu()
    
    return traversal_results


def compare_interpolation_quality(vae_model, ae_model, base_images, device, 
                                 save_dir='outputs/comparisons'):
    """
    Compare interpolation quality between VAE and standard autoencoder.
    This demonstrates the importance of continuous latent space.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vae_model.eval()
    ae_model.eval()
    
    # Select pairs of images for interpolation
    num_pairs = min(3, base_images.size(0) - 1)
    
    for pair_idx in range(num_pairs):
        img1 = base_images[pair_idx:pair_idx+1]
        img2 = base_images[pair_idx+1:pair_idx+2]
        
        print(f"Comparing interpolation for image pair {pair_idx + 1}...")
        
        with torch.no_grad():
            # VAE interpolation
            vae_interpolation = vae_model.interpolate(img1, img2, num_steps=10)
            
            # Autoencoder interpolation
            ae_interpolation = ae_model.interpolate(img1, img2, num_steps=10)
        
        # Create comparison grid
        comparison = torch.cat([vae_interpolation, ae_interpolation], dim=0)
        
        save_image_grid(
            comparison,
            nrow=10,
            filename=f'{save_dir}/interpolation_comparison_pair_{pair_idx+1}.png',
            title=f'Interpolation Comparison - Pair {pair_idx+1}\nTop: VAE | Bottom: Autoencoder'
        )


def analyze_semantic_directions(model, base_images, device, save_dir='outputs/semantic'):
    """
    Identify potential semantic directions in latent space.
    This helps understand what facial attributes the model has learned.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model_name = 'VAE' if 'VAE' in model.__class__.__name__ else 'Autoencoder'
    
    print(f"Analyzing semantic directions for {model_name}...")
    
    # Get latent representations of base images
    with torch.no_grad():
        if 'VAE' in model.__class__.__name__:
            mu, _ = model.encode(base_images)
            latent_codes = mu
        else:
            latent_codes = model.encode(base_images)
    
    # Find directions with high variance (potentially semantic)
    latent_std = torch.std(latent_codes, dim=0)
    high_variance_dims = torch.argsort(latent_std, descending=True)[:20]  # Top 20 dimensions
    
    print(f"Top dimensions by variance: {high_variance_dims[:10].tolist()}")
    
    # Test these high-variance dimensions with extreme values
    for i, dim_idx in enumerate(high_variance_dims[:5]):  # Test top 5
        dim_idx = dim_idx.item()
        
        # Use first image as base
        base_image = base_images[0:1]
        
        # Test extreme values
        extreme_values = [-3, -2, -1, 0, 1, 2, 3]
        
        traversals = []
        with torch.no_grad():
            for value in extreme_values:
                if hasattr(model, 'traverse_latent_dimension'):
                    traversed = model.traverse_latent_dimension(
                        base_image, dim_idx, [value], device
                    )
                else:
                    if 'VAE' in model.__class__.__name__:
                        mu, _ = model.encode(base_image)
                        z = mu.clone()
                    else:
                        z = model.encode(base_image).clone()
                    
                    z[0, dim_idx] = value
                    traversed = model.decode(z)
                
                traversals.append(traversed)
        
        traversal_grid = torch.cat(traversals, dim=0)
        
        save_image_grid(
            traversal_grid,
            nrow=len(extreme_values),
            filename=f'{save_dir}/{model_name.lower()}_semantic_dim_{dim_idx}.png',
            title=f'{model_name} - High Variance Dimension {dim_idx}'
        )


def create_presentation_traversals(model, base_images, device, specific_dims=None,
                                 save_dir='outputs/presentation'):
    """
    Create clean traversals for presentation purposes.
    Focus on the most interesting/semantic dimensions.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model_name = 'VAE' if 'VAE' in model.__class__.__name__ else 'Autoencoder'
    
    # If specific dimensions not provided, use some common ones
    if specific_dims is None:
        # These are just examples - you'd identify these through analysis
        specific_dims = [5, 12, 23, 37, 45, 67, 89, 103]  # Random selection for demo
    
    # Use first image as base for clean presentation
    base_image = base_images[0:1]
    
    traversal_range = torch.linspace(-2.5, 2.5, 9).to(device)
    
    for dim_idx in specific_dims[:4]:  # Show top 4 for presentation
        print(f"Creating presentation traversal for dimension {dim_idx}...")
        
        traversals = []
        with torch.no_grad():
            for value in traversal_range:
                if hasattr(model, 'traverse_latent_dimension'):
                    traversed = model.traverse_latent_dimension(
                        base_image, dim_idx, [value], device
                    )
                else:
                    if 'VAE' in model.__class__.__name__:
                        mu, _ = model.encode(base_image)
                        z = mu.clone()
                    else:
                        z = model.encode(base_image).clone()
                    
                    z[0, dim_idx] = value
                    traversed = model.decode(z)
                
                traversals.append(traversed)
        
        traversal_grid = torch.cat(traversals, dim=0)
        
        # Create a cleaner visualization for presentation
        fig, axes = plt.subplots(1, len(traversal_range), figsize=(18, 2))
        for i, (ax, img) in enumerate(zip(axes, traversal_grid)):
            img_np = img.cpu().permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(f'{traversal_range[i]:.1f}', fontsize=10)
            ax.axis('off')
        
        plt.suptitle(f'{model_name} - Dimension {dim_idx} Traversal', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{model_name.lower()}_presentation_dim_{dim_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Latent space traversal experiments')
    parser.add_argument('--data_dir', type=str, default='data/celeba',
                       help='Path to CelebA dataset')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Path to trained models')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of base images to use')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    dataloader = create_demo_dataloader(args.data_dir, batch_size=16)
    base_images = get_sample_images(dataloader, args.num_images, device)
    
    if base_images is None:
        print("Error: Could not load sample images. Check your data directory.")
        return
    
    # Load models
    vae_model = VAE(latent_dim=args.latent_dim)
    ae_model = StandardAutoencoder(latent_dim=args.latent_dim)
    
    vae_weights_path = os.path.join(args.model_dir, 'vae_final.pth')
    ae_weights_path = os.path.join(args.model_dir, 'autoencoder_final.pth')
    
    vae_loaded = load_model_weights(vae_model, vae_weights_path, device)
    ae_loaded = load_model_weights(ae_model, ae_weights_path, device)
    
    if not (vae_loaded or ae_loaded):
        print("Warning: No trained models found. Using randomly initialized models.")
        print("Run experiments/train_models.py first for better results.")
    
    vae_model.to(device)
    ae_model.to(device)
    
    # Run experiments
    print("\n" + "="*60)
    print("LATENT SPACE TRAVERSAL EXPERIMENTS")
    print("="*60)
    
    # 1. Systematic traversal
    print("\n1. Systematic Latent Traversal...")
    vae_results = systematic_latent_traversal(
        vae_model, base_images, device, 
        save_dir=os.path.join(args.output_dir, 'traversals', 'vae')
    )
    ae_results = systematic_latent_traversal(
        ae_model, base_images, device,
        save_dir=os.path.join(args.output_dir, 'traversals', 'autoencoder')
    )
    
    # 2. Interpolation comparison
    print("\n2. Interpolation Quality Comparison...")
    compare_interpolation_quality(
        vae_model, ae_model, base_images, device,
        save_dir=os.path.join(args.output_dir, 'comparisons')
    )
    
    # 3. Semantic direction analysis
    print("\n3. Semantic Direction Analysis...")
    analyze_semantic_directions(
        vae_model, base_images, device,
        save_dir=os.path.join(args.output_dir, 'semantic', 'vae')
    )
    analyze_semantic_directions(
        ae_model, base_images, device,
        save_dir=os.path.join(args.output_dir, 'semantic', 'autoencoder')
    )
    
    # 4. Presentation-ready traversals
    print("\n4. Creating Presentation Traversals...")
    create_presentation_traversals(
        vae_model, base_images, device,
        save_dir=os.path.join(args.output_dir, 'presentation', 'vae')
    )
    create_presentation_traversals(
        ae_model, base_images, device,
        save_dir=os.path.join(args.output_dir, 'presentation', 'autoencoder')
    )
    
    print(f"\nAll experiments completed! Results saved in {args.output_dir}")
    print("\nRecommended next steps:")
    print("1. Examine the traversal results to identify semantic dimensions")
    print("2. Use the best dimensions in your presentation demo")
    print("3. Run visualization.py to create latent space plots")


if __name__ == '__main__':
    main()