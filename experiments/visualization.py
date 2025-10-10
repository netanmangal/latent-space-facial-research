"""
Visualization utilities for latent space analysis.

This script creates various visualizations to understand the structure
and properties of the latent space in VAEs vs standard autoencoders.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae_model import VAE
from models.autoencoder_model import StandardAutoencoder
from models.utils import get_device, load_model_weights, analyze_latent_space_statistics
from data.celeba_loader import create_demo_dataloader


def visualize_latent_distribution(vae_model, ae_model, dataloader, device, 
                                save_dir='outputs/visualizations'):
    """
    Visualize the distribution of latent representations.
    Shows the difference between VAE's regularized space and AE's arbitrary space.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get latent representations
    vae_codes, vae_stats = analyze_latent_space_statistics(vae_model, dataloader, device)
    ae_codes, ae_stats = analyze_latent_space_statistics(ae_model, dataloader, device)
    
    # 1. Distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # VAE latent distribution
    axes[0, 0].hist(vae_codes.flatten(), bins=50, alpha=0.7, density=True, color='blue')
    axes[0, 0].set_title('VAE Latent Distribution')
    axes[0, 0].set_xlabel('Latent Value')
    axes[0, 0].set_ylabel('Density')
    
    # Autoencoder latent distribution
    axes[1, 0].hist(ae_codes.flatten(), bins=50, alpha=0.7, density=True, color='red')
    axes[1, 0].set_title('Autoencoder Latent Distribution')
    axes[1, 0].set_xlabel('Latent Value')
    axes[1, 0].set_ylabel('Density')
    
    # Standard deviation per dimension
    axes[0, 1].plot(vae_stats['std'], color='blue')
    axes[0, 1].set_title('VAE - Std per Dimension')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Standard Deviation')
    
    axes[1, 1].plot(ae_stats['std'], color='red')
    axes[1, 1].set_title('Autoencoder - Std per Dimension')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Standard Deviation')
    
    # Mean per dimension
    axes[0, 2].plot(vae_stats['mean'], color='blue')
    axes[0, 2].set_title('VAE - Mean per Dimension')
    axes[0, 2].set_xlabel('Dimension')
    axes[0, 2].set_ylabel('Mean')
    
    axes[1, 2].plot(ae_stats['mean'], color='red')
    axes[1, 2].set_title('Autoencoder - Mean per Dimension')
    axes[1, 2].set_xlabel('Dimension')
    axes[1, 2].set_ylabel('Mean')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return vae_codes, ae_codes, vae_stats, ae_stats


def create_2d_latent_visualization(vae_codes, ae_codes, save_dir='outputs/visualizations'):
    """
    Create 2D visualizations of the latent space using PCA and t-SNE.
    This shows the overall structure and clustering in latent space.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Reduce dimensionality for visualization
    print("Computing PCA projections...")
    pca = PCA(n_components=2)
    vae_pca = pca.fit_transform(vae_codes)
    ae_pca = pca.transform(ae_codes)
    
    print("Computing t-SNE projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # Combine for consistent t-SNE
    combined = np.vstack([vae_codes, ae_codes])
    combined_tsne = tsne.fit_transform(combined)
    vae_tsne = combined_tsne[:len(vae_codes)]
    ae_tsne = combined_tsne[len(vae_codes):]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PCA plots
    axes[0, 0].scatter(vae_pca[:, 0], vae_pca[:, 1], alpha=0.6, s=20, color='blue')
    axes[0, 0].set_title('VAE Latent Space (PCA)')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    axes[0, 1].scatter(ae_pca[:, 0], ae_pca[:, 1], alpha=0.6, s=20, color='red')
    axes[0, 1].set_title('Autoencoder Latent Space (PCA)')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # t-SNE plots
    axes[1, 0].scatter(vae_tsne[:, 0], vae_tsne[:, 1], alpha=0.6, s=20, color='blue')
    axes[1, 0].set_title('VAE Latent Space (t-SNE)')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    
    axes[1, 1].scatter(ae_tsne[:, 0], ae_tsne[:, 1], alpha=0.6, s=20, color='red')
    axes[1, 1].set_title('Autoencoder Latent Space (t-SNE)')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_space_2d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return vae_pca, ae_pca, vae_tsne, ae_tsne


def create_interactive_3d_visualization(vae_codes, ae_codes, save_dir='outputs/visualizations'):
    """
    Create interactive 3D visualization of latent space.
    Great for presentations and detailed exploration.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Use PCA for 3D reduction
    pca_3d = PCA(n_components=3)
    vae_pca_3d = pca_3d.fit_transform(vae_codes)
    ae_pca_3d = pca_3d.transform(ae_codes)
    
    # Create interactive plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['VAE Latent Space', 'Autoencoder Latent Space'],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    # VAE plot
    fig.add_trace(
        go.Scatter3d(
            x=vae_pca_3d[:, 0],
            y=vae_pca_3d[:, 1],
            z=vae_pca_3d[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='blue',
                opacity=0.6
            ),
            name='VAE'
        ),
        row=1, col=1
    )
    
    # Autoencoder plot
    fig.add_trace(
        go.Scatter3d(
            x=ae_pca_3d[:, 0],
            y=ae_pca_3d[:, 1],
            z=ae_pca_3d[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='red',
                opacity=0.6
            ),
            name='Autoencoder'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='3D Latent Space Comparison',
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})'
        ),
        scene2=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})'
        )
    )
    
    # Save interactive plot
    fig.write_html(f'{save_dir}/interactive_3d_comparison.html')
    print(f"Interactive 3D plot saved to {save_dir}/interactive_3d_comparison.html")
    
    return vae_pca_3d, ae_pca_3d


def visualize_random_sampling_quality(vae_model, ae_model, device, num_samples=25,
                                    save_dir='outputs/visualizations'):
    """
    Compare the quality of random sampling from VAE vs Autoencoder latent space.
    This demonstrates the completeness of VAE's continuous latent space.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vae_model.eval()
    ae_model.eval()
    
    with torch.no_grad():
        # Sample from both models
        vae_samples = vae_model.sample(num_samples, device)
        ae_samples = ae_model.sample(num_samples, device)
    
    # Create comparison grid
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # VAE samples
    vae_grid = vae_samples.cpu()
    grid_size = int(np.sqrt(num_samples))
    vae_display = torch.zeros(3, grid_size * 64, grid_size * 64)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j < num_samples:
                vae_display[:, i*64:(i+1)*64, j*64:(j+1)*64] = vae_grid[i*grid_size + j]
    
    axes[0].imshow(vae_display.permute(1, 2, 0))
    axes[0].set_title('VAE Random Samples (Continuous Latent Space)', fontsize=14)
    axes[0].axis('off')
    
    # Autoencoder samples
    ae_grid = ae_samples.cpu()
    ae_display = torch.zeros(3, grid_size * 64, grid_size * 64)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j < num_samples:
                ae_display[:, i*64:(i+1)*64, j*64:(j+1)*64] = ae_grid[i*grid_size + j]
    
    axes[1].imshow(ae_display.permute(1, 2, 0))
    axes[1].set_title('Autoencoder Random Samples (Discrete Latent Space)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/random_sampling_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_latent_space_density_plot(vae_codes, ae_codes, save_dir='outputs/visualizations'):
    """
    Create density plots showing the coverage and structure of latent space.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Use first two principal components for density visualization
    pca = PCA(n_components=2)
    vae_pca = pca.fit_transform(vae_codes)
    ae_pca = pca.transform(ae_codes)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # VAE density plot
    axes[0].hexbin(vae_pca[:, 0], vae_pca[:, 1], gridsize=30, cmap='Blues', alpha=0.7)
    axes[0].set_title('VAE Latent Space Density')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # Autoencoder density plot
    axes[1].hexbin(ae_pca[:, 0], ae_pca[:, 1], gridsize=30, cmap='Reds', alpha=0.7)
    axes[1].set_title('Autoencoder Latent Space Density')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_density_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Latent space visualization')
    parser.add_argument('--data_dir', type=str, default='data/celeba',
                       help='Path to CelebA dataset')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Path to trained models')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for analysis')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    dataloader = create_demo_dataloader(args.data_dir, batch_size=32)
    
    # Load models
    vae_model = VAE(latent_dim=args.latent_dim)
    ae_model = StandardAutoencoder(latent_dim=args.latent_dim)
    
    vae_weights_path = os.path.join(args.model_dir, 'vae_final.pth')
    ae_weights_path = os.path.join(args.model_dir, 'autoencoder_final.pth')
    
    load_model_weights(vae_model, vae_weights_path, device)
    load_model_weights(ae_model, ae_weights_path, device)
    
    vae_model.to(device)
    ae_model.to(device)
    
    print("\n" + "="*60)
    print("LATENT SPACE VISUALIZATION")
    print("="*60)
    
    # 1. Latent distribution analysis
    print("\n1. Analyzing latent distributions...")
    vae_codes, ae_codes, vae_stats, ae_stats = visualize_latent_distribution(
        vae_model, ae_model, dataloader, device, args.output_dir
    )
    
    # 2. 2D visualizations
    print("\n2. Creating 2D latent space visualizations...")
    vae_pca, ae_pca, vae_tsne, ae_tsne = create_2d_latent_visualization(
        vae_codes, ae_codes, args.output_dir
    )
    
    # 3. Interactive 3D visualization
    print("\n3. Creating interactive 3D visualization...")
    vae_pca_3d, ae_pca_3d = create_interactive_3d_visualization(
        vae_codes, ae_codes, args.output_dir
    )
    
    # 4. Random sampling comparison
    print("\n4. Comparing random sampling quality...")
    visualize_random_sampling_quality(
        vae_model, ae_model, device, num_samples=25, save_dir=args.output_dir
    )
    
    # 5. Density plots
    print("\n5. Creating density plots...")
    create_latent_space_density_plot(vae_codes, ae_codes, args.output_dir)
    
    print(f"\nAll visualizations completed! Results saved in {args.output_dir}")
    print("\nKey findings to highlight in your presentation:")
    print(f"- VAE latent space standard deviation: {np.mean(vae_stats['std']):.3f}")
    print(f"- Autoencoder latent space standard deviation: {np.mean(ae_stats['std']):.3f}")
    print(f"- VAE latent space range: [{np.min(vae_codes):.2f}, {np.max(vae_codes):.2f}]")
    print(f"- Autoencoder latent space range: [{np.min(ae_codes):.2f}, {np.max(ae_codes):.2f}]")


if __name__ == '__main__':
    main()