"""
Training script for VAE and Standard Autoencoder models.

This script trains both models on the CelebA dataset and saves the weights
for use in demonstrations and analysis.
"""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae_model import StandardVAE, BetaVAE, beta_vae_loss_function, vae_loss_function
from models.autoencoder_model import StandardAutoencoder, autoencoder_loss_function
from models.utils import get_device, set_seed, save_image_grid
from data.celeba_loader import create_celeba_dataloaders, create_demo_dataloader


def train_vae(model, train_loader, val_loader, device, num_epochs=50, 
              learning_rate=1e-3, model_name='VAE', save_dir='outputs/models'):
    """
    Train VAE models (both StandardVAE and BetaVAE).
    
    Args:
        model: VAE model (StandardVAE or BetaVAE)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        model_name: Name of the model for logging
        save_dir: Directory to save model weights
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Tensorboard logging
    writer = SummaryWriter(f'runs/{model_name.lower()}_beta_{model.beta}')
    
    train_losses = []
    val_losses = []
    
    print(f"Training {model_name} with β={model.beta} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar, _ = model(data)
            
            if isinstance(model, StandardVAE):
                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, data, mu, logvar)
            else:  # BetaVAE
                loss, recon_loss, kl_loss = beta_vae_loss_function(recon_batch, data, mu, logvar, model.beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar, _ = model(data)
                
                if isinstance(model, StandardVAE):
                    loss, recon_loss, kl_loss = vae_loss_function(recon_batch, data, mu, logvar)
                else:  # BetaVAE
                    loss, recon_loss, kl_loss = beta_vae_loss_function(recon_batch, data, mu, logvar, model.beta)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Loss/Train_Reconstruction', train_recon_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/Train_KL', train_kl_loss / len(train_loader.dataset), epoch)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save sample reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_data, _ = next(iter(val_loader))
                sample_data = sample_data[:8].to(device)
                recon_sample, _, _, _ = model(sample_data)
                
                comparison = torch.cat([sample_data, recon_sample])
                save_image_grid(
                    comparison, 
                    nrow=8, 
                    filename=f'{save_dir}/vae_reconstruction_epoch_{epoch+1}.png',
                    title=f'VAE Reconstructions - Epoch {epoch+1}'
                )
                
                # Also save some random samples
                random_samples = model.sample(8, device)
                save_image_grid(
                    random_samples,
                    nrow=8,
                    filename=f'{save_dir}/vae_samples_epoch_{epoch+1}.png',
                    title=f'VAE Random Samples - Epoch {epoch+1}'
                )
    
    # Save final model
    torch.save(model.state_dict(), f'{save_dir}/vae_final.pth')
    writer.close()
    
    return train_losses, val_losses


def train_autoencoder(model, train_loader, val_loader, device, num_epochs=50,
                     learning_rate=1e-3, save_dir='outputs/models'):
    """
    Train the standard autoencoder model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Tensorboard logging
    writer = SummaryWriter('runs/autoencoder')
    
    train_losses = []
    val_losses = []
    
    print(f"Training Standard Autoencoder for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, _ = model(data)
            
            loss = autoencoder_loss_function(recon_batch, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, _ = model(data)
                loss = autoencoder_loss_function(recon_batch, data)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save sample reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_data, _ = next(iter(val_loader))
                sample_data = sample_data[:8].to(device)
                recon_sample, _ = model(sample_data)
                
                comparison = torch.cat([sample_data, recon_sample])
                save_image_grid(
                    comparison, 
                    nrow=8, 
                    filename=f'{save_dir}/ae_reconstruction_epoch_{epoch+1}.png',
                    title=f'Autoencoder Reconstructions - Epoch {epoch+1}'
                )
    
    # Save final model
    torch.save(model.state_dict(), f'{save_dir}/autoencoder_final.pth')
    writer.close()
    
    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train VAE and Autoencoder models')
    parser.add_argument('--data_dir', type=str, default='data/celeba',
                       help='Path to CelebA dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for VAE (higher = more regularization)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--demo_mode', action='store_true',
                       help='Use smaller dataset for quick demo')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    if args.demo_mode:
        print("Running in demo mode with smaller dataset...")
        train_loader = create_demo_dataloader(args.data_dir, args.batch_size)
        val_loader = train_loader  # Use same for demo
    else:
        train_loader, val_loader, _ = create_celeba_dataloaders(
            args.data_dir, args.batch_size
        )
    
    # Create all three models for comprehensive comparison
    standard_vae = StandardVAE(latent_dim=args.latent_dim)
    beta_vae = BetaVAE(latent_dim=args.latent_dim)
    ae_model = StandardAutoencoder(latent_dim=args.latent_dim)
    
    print(f"StandardVAE (β=1.0) parameters: {sum(p.numel() for p in standard_vae.parameters()):,}")
    print(f"BetaVAE (β=4.0) parameters: {sum(p.numel() for p in beta_vae.parameters()):,}")
    print(f"StandardAutoencoder parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    
    # Create output directory
    save_dir = 'outputs/models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Train Standard VAE (β=1.0)
    print("\n" + "="*60)
    print("TRAINING STANDARD VAE (β=1.0 - Continuous but Entangled)")
    print("="*60)
    print("Benefits: Continuous latent space vs Autoencoder")
    standard_vae_train, standard_vae_val = train_vae(
        standard_vae, train_loader, val_loader, device,
        num_epochs=args.epochs, learning_rate=args.learning_rate,
        model_name='StandardVAE', save_dir=save_dir
    )
    
    # Train Beta VAE (β=4.0)
    print("\n" + "="*60)
    print("TRAINING β-VAE (β=4.0 - Continuous AND Disentangled)")
    print("="*60)
    print("Benefits: Superior disentanglement vs Standard VAE")
    beta_vae_train, beta_vae_val = train_vae(
        beta_vae, train_loader, val_loader, device,
        num_epochs=args.epochs, learning_rate=args.learning_rate,
        model_name='BetaVAE', save_dir=save_dir
    )
    
    # Train Standard Autoencoder (Discrete)
    print("\n" + "="*60)
    print("TRAINING STANDARD AUTOENCODER (Discrete Latent Space)")
    print("="*60)
    print("Baseline: Discrete latent space with limitations")
    ae_train_losses, ae_val_losses = train_autoencoder(
        ae_model, train_loader, val_loader, device,
        num_epochs=args.epochs, learning_rate=args.learning_rate,
        save_dir=save_dir
    )
    
    # Plot comprehensive training curves for all three models
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(ae_train_losses, label='AE Train', color='red')
    plt.plot(ae_val_losses, label='AE Val', color='red', linestyle='--')
    plt.title('Standard Autoencoder\n(Discrete Latent Space)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(standard_vae_train, label='Standard VAE Train', color='blue')
    plt.plot(standard_vae_val, label='Standard VAE Val', color='blue', linestyle='--')
    plt.title('Standard VAE (β=1.0)\n(Continuous, Entangled)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(beta_vae_train, label='β-VAE Train', color='green')
    plt.plot(beta_vae_val, label='β-VAE Val', color='green', linestyle='--')
    plt.title('β-VAE (β=4.0)\n(Continuous, Disentangled)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary comparison
    print("\n" + "="*80)
    print("🎯 TRAINING SUMMARY - THREE MODEL COMPARISON")
    print("="*80)
    print(f"1. Standard Autoencoder  - Final Loss: {ae_train_losses[-1]:.4f}")
    print(f"2. Standard VAE (β=1.0)  - Final Loss: {standard_vae_train[-1]:.4f}")
    print(f"3. β-VAE (β=4.0)         - Final Loss: {beta_vae_train[-1]:.4f}")
    print("\n🔬 Research Findings:")
    print("   • Standard AE: Discrete latent space (baseline)")
    print("   • Standard VAE: Continuous latent space (smooth interpolation)")
    print("   • β-VAE: Continuous + Disentangled (semantic control)")
    print("\n💡 Next Steps:")
    print("   • Run analysis: python notebooks/research_analysis.ipynb")
    print("   • Run demo: python demos/presentation_demo.py")
    
    print(f"\nTraining completed! Models saved in {save_dir}")
    print("Run the presentation demo with: python demos/presentation_demo.py")


if __name__ == '__main__':
    main()