"""
Presentation Demo: Variational Autoencoders vs Standard Autoencoders

This script provides an interactive demo perfect for research presentations.
It demonstrates the key differences between discrete and continuous latent spaces.

Usage:
    python demos/presentation_demo.py

Controls during presentation:
    - Press 'n' for next demonstration
    - Press 'q' to quit
    - Press 's' to save current visualization
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from time import sleep
import argparse

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae_model import VAE
from models.autoencoder_model import StandardAutoencoder
from models.utils import get_device, load_model_weights, save_image_grid
from data.celeba_loader import create_demo_dataloader, get_sample_images


class PresentationDemo:
    """
    Interactive presentation demo class.
    Manages the flow of demonstrations and visualizations.
    """
    
    def __init__(self, vae_model, ae_model, sample_images, device):
        self.vae_model = vae_model
        self.ae_model = ae_model
        self.sample_images = sample_images
        self.device = device
        
        # Demo configuration
        self.current_demo = 0
        self.demo_functions = [
            self.demo_reconstruction_comparison,
            self.demo_random_sampling,
            self.demo_interpolation_comparison,
            self.demo_latent_traversal,
            self.demo_continuity_analysis
        ]
        self.demo_titles = [
            "1. Reconstruction Quality Comparison",
            "2. Random Sampling from Latent Space",
            "3. Interpolation Between Images",
            "4. Latent Space Traversal",
            "5. Continuity Analysis"
        ]
        
        plt.ion()  # Interactive mode
    
    def run_presentation(self):
        """Run the full presentation demo"""
        print("="*70)
        print("VARIATIONAL AUTOENCODERS: LATENT SPACE CONTINUITY DEMO")
        print("="*70)
        print("\nThis demo illustrates the importance of continuous latent spaces")
        print("in Variational Autoencoders compared to standard autoencoders.\n")
        
        print("Controls:")
        print("- Press ENTER to continue to next demo")
        print("- Type 'save' to save current visualization")
        print("- Type 'quit' to exit")
        print("\n" + "="*70)
        
        while self.current_demo < len(self.demo_functions):
            self.show_current_demo()
            
            command = input(f"\nPress ENTER for next demo (or 'save'/'quit'): ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'save':
                self.save_current_demo()
            
            self.current_demo += 1
        
        print("\nDemo completed! Thank you for your attention.")
        plt.ioff()
    
    def show_current_demo(self):
        """Show the current demonstration"""
        if self.current_demo < len(self.demo_functions):
            print(f"\n{self.demo_titles[self.current_demo]}")
            print("-" * len(self.demo_titles[self.current_demo]))
            self.demo_functions[self.current_demo]()
    
    def save_current_demo(self):
        """Save current demo visualization"""
        save_dir = 'outputs/presentation_saves'
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/demo_{self.current_demo+1}_{self.demo_titles[self.current_demo].split(".")[1].strip().replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
    
    def demo_reconstruction_comparison(self):
        """Demo 1: Compare reconstruction quality"""
        print("Comparing how well each model reconstructs the input images...")
        
        with torch.no_grad():
            # Get reconstructions
            vae_recon, _, _, _ = self.vae_model(self.sample_images)
            ae_recon, _ = self.ae_model(self.sample_images)
        
        # Create comparison visualization
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        
        for i in range(4):
            if i < self.sample_images.size(0):
                # Original
                img = self.sample_images[i].cpu().permute(1, 2, 0)
                axes[0, i].imshow(img)
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # VAE reconstruction
                vae_img = vae_recon[i].cpu().permute(1, 2, 0)
                axes[1, i].imshow(vae_img)
                axes[1, i].set_title('VAE')
                axes[1, i].axis('off')
                
                # AE reconstruction
                ae_img = ae_recon[i].cpu().permute(1, 2, 0)
                axes[2, i].imshow(ae_img)
                axes[2, i].set_title('Autoencoder')
                axes[2, i].axis('off')
        
        plt.suptitle('Reconstruction Comparison\nTop: Original | Middle: VAE | Bottom: Autoencoder', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("\nKey Observation:")
        print("Both models can reconstruct images reasonably well, but VAE")
        print("maintains better generalization due to regularized latent space.")
    
    def demo_random_sampling(self):
        """Demo 2: Random sampling comparison"""
        print("Sampling random points from the latent space...")
        print("VAE should generate realistic faces, AE may generate artifacts.")
        
        with torch.no_grad():
            vae_samples = self.vae_model.sample(8, self.device)
            ae_samples = self.ae_model.sample(8, self.device)
        
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(8):
            # VAE samples
            vae_img = vae_samples[i].cpu().permute(1, 2, 0)
            axes[0, i].imshow(vae_img)
            axes[0, i].axis('off')
            
            # AE samples
            ae_img = ae_samples[i].cpu().permute(1, 2, 0)
            axes[1, i].imshow(ae_img)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('VAE\n(Continuous)', fontsize=12)
        axes[1, 0].set_ylabel('Autoencoder\n(Discrete)', fontsize=12)
        
        plt.suptitle('Random Sampling from Latent Space', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("\nKey Observation:")
        print("VAE generates coherent faces because every point in its latent")
        print("space corresponds to a valid image. Autoencoder may produce")
        print("artifacts due to gaps in its discrete latent space.")
    
    def demo_interpolation_comparison(self):
        """Demo 3: Interpolation between two images"""
        print("Interpolating between two face images in latent space...")
        
        # Use first two images
        img1 = self.sample_images[0:1]
        img2 = self.sample_images[1:1] if self.sample_images.size(0) > 1 else self.sample_images[0:1]
        
        with torch.no_grad():
            vae_interp = self.vae_model.interpolate(img1, img2, num_steps=10)
            ae_interp = self.ae_model.interpolate(img1, img2, num_steps=10)
        
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        
        for i in range(10):
            # VAE interpolation
            vae_img = vae_interp[i].cpu().permute(1, 2, 0)
            axes[0, i].imshow(vae_img)
            axes[0, i].axis('off')
            
            # AE interpolation
            ae_img = ae_interp[i].cpu().permute(1, 2, 0)
            axes[1, i].imshow(ae_img)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('VAE\n(Smooth)', fontsize=12)
        axes[1, 0].set_ylabel('Autoencoder\n(Abrupt)', fontsize=12)
        
        plt.suptitle('Interpolation Between Two Images', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("\nKey Observation:")
        print("VAE produces smooth, realistic transitions between images.")
        print("Autoencoder shows abrupt changes or artifacts due to")
        print("discontinuities in the latent space.")
    
    def demo_latent_traversal(self):
        """Demo 4: Traverse specific latent dimensions"""
        print("Traversing individual latent dimensions...")
        print("This shows how VAE learns interpretable features.")
        
        base_image = self.sample_images[0:1]
        
        # Test a few different dimensions
        interesting_dims = [5, 23, 45, 67]  # You'd identify these through analysis
        traversal_values = torch.linspace(-2, 2, 9).to(self.device)
        
        fig, axes = plt.subplots(len(interesting_dims), 9, figsize=(18, 8))
        
        for dim_idx, latent_dim in enumerate(interesting_dims):
            print(f"  Traversing dimension {latent_dim}...")
            
            with torch.no_grad():
                for val_idx, value in enumerate(traversal_values):
                    # VAE traversal
                    mu, _ = self.vae_model.encode(base_image)
                    z = mu.clone()
                    z[0, latent_dim] = value
                    traversed = self.vae_model.decode(z)
                    
                    img = traversed[0].cpu().permute(1, 2, 0)
                    axes[dim_idx, val_idx].imshow(img)
                    axes[dim_idx, val_idx].axis('off')
                    
                    if dim_idx == 0:
                        axes[dim_idx, val_idx].set_title(f'{value:.1f}', fontsize=10)
            
            axes[dim_idx, 0].set_ylabel(f'Dim {latent_dim}', fontsize=10)
        
        plt.suptitle('Latent Dimension Traversal (VAE)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("\nKey Observation:")
        print("Each dimension in VAE latent space often corresponds to")
        print("interpretable features like head pose, lighting, or expression.")
    
    def demo_continuity_analysis(self):
        """Demo 5: Analyze continuity of latent space"""
        print("Analyzing the continuity and completeness of latent spaces...")
        
        # Sample points around a base point
        base_image = self.sample_images[0:1]
        
        with torch.no_grad():
            # Get base latent representation
            if hasattr(self.vae_model, 'encode'):
                vae_mu, _ = self.vae_model.encode(base_image)
                base_vae_z = vae_mu
            else:
                _, base_vae_z = self.vae_model(base_image)
            
            base_ae_z = self.ae_model.encode(base_image)
            
            # Sample points in a small neighborhood
            num_samples = 16
            noise_scale = 0.5
            
            vae_neighbors = []
            ae_neighbors = []
            
            for _ in range(num_samples):
                # Add small noise to base latent code
                noise = torch.randn_like(base_vae_z) * noise_scale
                
                vae_neighbor_z = base_vae_z + noise
                ae_neighbor_z = base_ae_z + noise
                
                vae_neighbor_img = self.vae_model.decode(vae_neighbor_z)
                ae_neighbor_img = self.ae_model.decode(ae_neighbor_z)
                
                vae_neighbors.append(vae_neighbor_img)
                ae_neighbors.append(ae_neighbor_img)
        
        # Visualize neighborhoods
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(8):
            # VAE neighborhood
            if i < len(vae_neighbors):
                vae_img = vae_neighbors[i][0].cpu().permute(1, 2, 0)
                axes[0, i].imshow(vae_img)
            axes[0, i].axis('off')
            
            # AE neighborhood
            if i < len(ae_neighbors):
                ae_img = ae_neighbors[i][0].cpu().permute(1, 2, 0)
                axes[1, i].imshow(ae_img)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('VAE\n(Continuous)', fontsize=12)
        axes[1, 0].set_ylabel('Autoencoder\n(Discrete)', fontsize=12)
        
        plt.suptitle('Neighborhood Sampling Around a Base Point', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("\nKey Observation:")
        print("VAE maintains realistic images even when sampling from")
        print("neighborhoods around known points. Autoencoder may show")
        print("more artifacts or unrealistic variations.")


def create_quick_demo():
    """Create a quick demo with minimal setup"""
    device = get_device()
    
    # Create minimal models (for demo without trained weights)
    vae_model = VAE(latent_dim=64)  # Smaller for demo
    ae_model = StandardAutoencoder(latent_dim=64)
    
    vae_model.to(device)
    ae_model.to(device)
    
    # Generate random sample images (placeholder)
    sample_images = torch.randn(4, 3, 64, 64).to(device)
    sample_images = torch.sigmoid(sample_images)  # Normalize to [0,1]
    
    return vae_model, ae_model, sample_images, device


def main():
    parser = argparse.ArgumentParser(description='Presentation demo for VAE research')
    parser.add_argument('--data_dir', type=str, default='data/celeba',
                       help='Path to CelebA dataset')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Path to trained models')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--quick_demo', action='store_true',
                       help='Run quick demo without trained models or real data')
    
    args = parser.parse_args()
    
    print("Initializing presentation demo...")
    
    if args.quick_demo:
        print("Running quick demo mode...")
        vae_model, ae_model, sample_images, device = create_quick_demo()
    else:
        # Full setup with trained models and real data
        device = get_device()
        
        # Load data
        try:
            dataloader = create_demo_dataloader(args.data_dir, batch_size=16)
            sample_images = get_sample_images(dataloader, 4, device)
        except Exception as e:
            print(f"Could not load data: {e}")
            print("Using random sample images for demo...")
            sample_images = torch.randn(4, 3, 64, 64).to(device)
            sample_images = torch.sigmoid(sample_images)
        
        # Load models
        vae_model = VAE(latent_dim=args.latent_dim)
        ae_model = StandardAutoencoder(latent_dim=args.latent_dim)
        
        # Try to load trained weights
        vae_weights_path = os.path.join(args.model_dir, 'vae_final.pth')
        ae_weights_path = os.path.join(args.model_dir, 'autoencoder_final.pth')
        
        vae_loaded = load_model_weights(vae_model, vae_weights_path, device)
        ae_loaded = load_model_weights(ae_model, ae_weights_path, device)
        
        if not (vae_loaded or ae_loaded):
            print("Warning: Using randomly initialized models.")
            print("Train models first for better demo results.")
        
        vae_model.to(device)
        ae_model.to(device)
    
    # Run the presentation
    demo = PresentationDemo(vae_model, ae_model, sample_images, device)
    demo.run_presentation()


if __name__ == '__main__':
    main()