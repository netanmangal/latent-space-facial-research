"""
Interactive Demo for Latent Space Exploration

This script provides an interactive interface for exploring the latent space
of VAEs and autoencoders. Perfect for detailed analysis and experimentation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import argparse

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae_model import StandardVAE, BetaVAE
from models.autoencoder_model import StandardAutoencoder
from models.utils import get_device
from data.celeba_loader import create_demo_dataloader, get_sample_images


def load_trained_model(checkpoint_path, model_type, latent_dim=128, image_size=64, device='cpu'):
    """Load a trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Loading untrained model...")
        
        # Create untrained model
        if model_type == 'standard_ae':
            model = StandardAutoencoder(latent_dim=latent_dim, image_channels=3, image_size=image_size)
        elif model_type == 'standard_vae':
            model = StandardVAE(latent_dim=latent_dim, image_channels=3, image_size=image_size)
        elif model_type == 'beta_vae':
            model = BetaVAE(latent_dim=latent_dim, image_channels=3, image_size=image_size, beta=4.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model based on checkpoint info
    if model_type == 'standard_ae':
        model = StandardAutoencoder(
            latent_dim=checkpoint['latent_dim'],
            image_channels=3,
            image_size=checkpoint['image_size']
        )
    elif model_type == 'standard_vae':
        model = StandardVAE(
            latent_dim=checkpoint['latent_dim'],
            image_channels=3,
            image_size=checkpoint['image_size']
        )
    elif model_type == 'beta_vae':
        model = BetaVAE(
            latent_dim=checkpoint['latent_dim'],
            image_channels=3,
            image_size=checkpoint['image_size'],
            beta=checkpoint['beta']
        )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded trained {model_type} from {checkpoint_path}")
    return model


class InteractiveLatentExplorer:
    """
    Interactive GUI for exploring latent space.
    Allows real-time manipulation of latent dimensions.
    """
    
    def __init__(self, vae_model, ae_model, sample_images, device):
        self.vae_model = vae_model
        self.ae_model = ae_model
        self.sample_images = sample_images
        self.device = device
        self.current_model = 'vae'
        self.current_image_idx = 0
        self.latent_dim = vae_model.latent_dim
        
        # Initialize base latent codes
        self.update_base_latent_codes()
        
        # Create GUI
        self.setup_gui()
    
    def update_base_latent_codes(self):
        """Update base latent codes for current image"""
        current_image = self.sample_images[self.current_image_idx:self.current_image_idx+1]
        
        with torch.no_grad():
            if self.current_model == 'vae':
                # For VAE models, get the mean of the latent distribution
                if hasattr(self.vae_model, 'encode'):
                    mu, _ = self.vae_model.encode(current_image)
                    self.base_latent = mu.clone()
                else:
                    # Handle the case where encode is part of forward pass
                    _, mu, logvar, _ = self.vae_model(current_image)
                    self.base_latent = mu.clone()
            else:
                # For autoencoder, get latent directly
                if hasattr(self.ae_model, 'encode'):
                    self.base_latent = self.ae_model.encode(current_image).clone()
                else:
                    # Handle different autoencoder interface
                    latent, _ = self.ae_model(current_image)
                    self.base_latent = latent.clone()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Interactive Latent Space Explorer")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value='vae')
        ttk.Radiobutton(model_frame, text="VAE", variable=self.model_var, 
                       value='vae', command=self.on_model_change).pack(side=tk.LEFT)
        ttk.Radiobutton(model_frame, text="Autoencoder", variable=self.model_var, 
                       value='ae', command=self.on_model_change).pack(side=tk.LEFT)
        
        # Image selection
        image_frame = ttk.Frame(control_frame)
        image_frame.pack(fill=tk.X, pady=5)
        ttk.Label(image_frame, text="Base Image:").pack(side=tk.LEFT)
        self.image_var = tk.IntVar(value=0)
        self.image_spinbox = ttk.Spinbox(image_frame, from_=0, 
                                        to=self.sample_images.size(0)-1,
                                        textvariable=self.image_var,
                                        command=self.on_image_change, width=10)
        self.image_spinbox.pack(side=tk.LEFT)
        
        # Reset button
        ttk.Button(control_frame, text="Reset to Original", 
                  command=self.reset_latent).pack(pady=5)
        
        # Save button
        ttk.Button(control_frame, text="Save Current Image", 
                  command=self.save_current_image).pack(pady=5)
        
        # Sliders frame
        sliders_frame = ttk.LabelFrame(control_frame, text="Latent Dimensions", padding=5)
        sliders_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create scrollable frame for sliders
        canvas = tk.Canvas(sliders_frame, height=400)
        scrollbar = ttk.Scrollbar(sliders_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create sliders for latent dimensions
        self.sliders = []
        self.slider_vars = []
        
        # Show only first 20 dimensions initially (can be expanded)
        num_sliders = min(20, self.latent_dim)
        
        for i in range(num_sliders):
            var = tk.DoubleVar(value=0.0)
            self.slider_vars.append(var)
            
            dim_frame = ttk.Frame(scrollable_frame)
            dim_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dim_frame, text=f"Dim {i:02d}:", width=8).pack(side=tk.LEFT)
            
            slider = ttk.Scale(dim_frame, from_=-3.0, to=3.0, variable=var,
                             orient=tk.HORIZONTAL, length=200,
                             command=lambda val, idx=i: self.on_slider_change(idx, val))
            slider.pack(side=tk.LEFT, padx=5)
            self.sliders.append(slider)
            
            # Value label
            value_label = ttk.Label(dim_frame, text="0.00", width=6)
            value_label.pack(side=tk.LEFT)
            var.trace('w', lambda *args, label=value_label, v=var: 
                     label.config(text=f"{v.get():.2f}"))
        
        # Display frame
        display_frame = ttk.LabelFrame(main_frame, text="Generated Image", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display
        self.image_label = ttk.Label(display_frame)
        self.image_label.pack(expand=True)
        
        # Info label
        self.info_label = ttk.Label(display_frame, text="", font=('Arial', 10))
        self.info_label.pack(pady=5)
        
        # Initial update
        self.update_display()
    
    def on_model_change(self):
        """Handle model selection change"""
        self.current_model = self.model_var.get()
        self.update_base_latent_codes()
        self.reset_sliders()
        self.update_display()
    
    def on_image_change(self):
        """Handle base image change"""
        self.current_image_idx = self.image_var.get()
        self.update_base_latent_codes()
        self.reset_sliders()
        self.update_display()
    
    def on_slider_change(self, dim_idx, value):
        """Handle slider value change"""
        self.update_display()
    
    def reset_latent(self):
        """Reset to original latent code"""
        self.reset_sliders()
        self.update_display()
    
    def reset_sliders(self):
        """Reset all sliders to zero"""
        for var in self.slider_vars:
            var.set(0.0)
    
    def get_current_latent(self):
        """Get current latent code with slider modifications"""
        current_latent = self.base_latent.clone()
        
        for i, var in enumerate(self.slider_vars):
            if i < self.latent_dim:
                current_latent[0, i] += var.get()
        
        return current_latent
    
    def update_display(self):
        """Update the displayed image"""
        current_latent = self.get_current_latent()
        
        with torch.no_grad():
            if self.current_model == 'vae':
                if hasattr(self.vae_model, 'decode'):
                    generated = self.vae_model.decode(current_latent)
                else:
                    generated = self.vae_model.decoder(current_latent)
            else:
                if hasattr(self.ae_model, 'decode'):
                    generated = self.ae_model.decode(current_latent)
                else:
                    generated = self.ae_model.decoder(current_latent)
        
        # Convert to displayable image
        img_tensor = generated[0].cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Resize for display
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.photo)
        
        # Update info
        model_name = "VAE" if self.current_model == 'vae' else "Autoencoder"
        info_text = f"Model: {model_name} | Base Image: {self.current_image_idx}"
        self.info_label.config(text=info_text)
    
    def save_current_image(self):
        """Save the currently displayed image"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Generated Image"
            )
            
            if filename:
                current_latent = self.get_current_latent()
                
                with torch.no_grad():
                    if self.current_model == 'vae':
                        if hasattr(self.vae_model, 'decode'):
                            generated = self.vae_model.decode(current_latent)
                        else:
                            generated = self.vae_model.decoder(current_latent)
                    else:
                        if hasattr(self.ae_model, 'decode'):
                            generated = self.ae_model.decode(current_latent)
                        else:
                            generated = self.ae_model.decoder(current_latent)
                
                # Save full resolution
                img_tensor = generated[0].cpu()
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img_np)
                img_pil.save(filename)
                
                messagebox.showinfo("Success", f"Image saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def run(self):
        """Run the interactive explorer"""
        self.root.mainloop()


class MatplotlibSliderDemo:
    """
    Alternative matplotlib-based slider interface for latent exploration.
    Simpler but still interactive.
    """
    
    def __init__(self, model, sample_images, device, model_name='VAE'):
        self.model = model
        self.sample_images = sample_images
        self.device = device
        self.model_name = model_name
        self.current_image_idx = 0
        
        # Get base latent code
        self.update_base_latent()
        
        # Setup matplotlib interface
        self.setup_matplotlib_interface()
    
    def update_base_latent(self):
        """Update base latent code"""
        current_image = self.sample_images[self.current_image_idx:self.current_image_idx+1]
        
        with torch.no_grad():
            if 'VAE' in self.model_name:
                mu, _ = self.model.encode(current_image)
                self.base_latent = mu.clone()
            else:
                self.base_latent = self.model.encode(current_image).clone()
    
    def setup_matplotlib_interface(self):
        """Setup matplotlib slider interface"""
        self.fig, (self.ax_img, self.ax_controls) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Image display
        self.ax_img.set_title(f'{self.model_name} - Interactive Latent Space')
        self.ax_img.axis('off')
        
        # Controls area
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
        
        # Create sliders for first 8 dimensions
        self.sliders = []
        self.num_sliders = min(8, self.model.latent_dim)
        
        for i in range(self.num_sliders):
            ax_slider = plt.axes([0.6, 0.9 - i * 0.1, 0.3, 0.03])
            slider = Slider(ax_slider, f'Dim {i}', -3.0, 3.0, valinit=0.0)
            slider.on_changed(self.update_image)
            self.sliders.append(slider)
        
        # Reset button
        ax_reset = plt.axes([0.6, 0.02, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_sliders)
        
        # Next image button
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.next_button = Button(ax_next, 'Next Image')
        self.next_button.on_clicked(self.next_image)
        
        # Initial update
        self.update_image(None)
        
        plt.show()
    
    def update_image(self, val):
        """Update the displayed image based on slider values"""
        current_latent = self.base_latent.clone()
        
        # Apply slider modifications
        for i, slider in enumerate(self.sliders):
            current_latent[0, i] += slider.val
        
        # Generate image
        with torch.no_grad():
            if hasattr(self.model, 'decode'):
                generated = self.model.decode(current_latent)
            else:
                generated = self.model.decoder(current_latent)
        
        # Display
        img_np = generated[0].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        self.ax_img.clear()
        self.ax_img.imshow(img_np)
        self.ax_img.set_title(f'{self.model_name} - Image {self.current_image_idx}')
        self.ax_img.axis('off')
        
        self.fig.canvas.draw()
    
    def reset_sliders(self, event):
        """Reset all sliders to zero"""
        for slider in self.sliders:
            slider.reset()
    
    def next_image(self, event):
        """Switch to next base image"""
        self.current_image_idx = (self.current_image_idx + 1) % self.sample_images.size(0)
        self.update_base_latent()
        self.reset_sliders(None)
        self.update_image(None)


def main():
    parser = argparse.ArgumentParser(description='Interactive latent space explorer')
    parser.add_argument('--data_dir', type=str, default='data/celeba',
                       help='Path to CelebA dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='../models/checkpoints',
                       help='Path to trained model checkpoints')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--interface', type=str, default='gui', 
                       choices=['gui', 'matplotlib'],
                       help='Interface type: gui (tkinter) or matplotlib')
    parser.add_argument('--model_type', type=str, default='standard_vae',
                       choices=['standard_ae', 'standard_vae', 'beta_vae'],
                       help='Which model to load')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    print(f"🎯 Interactive Demo - Three-Model Comparison System")
    
    # Load data
    try:
        dataloader = create_demo_dataloader(args.data_dir, batch_size=16)
        sample_images = get_sample_images(dataloader, 8, device)
        print(f"✅ Loaded {sample_images.size(0)} sample images from dataset")
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Using random sample images...")
        sample_images = torch.randn(8, 3, 64, 64).to(device)
        sample_images = torch.sigmoid(sample_images)
    
    # Load all three trained models
    models = {}
    model_names = {
        'standard_ae': 'Standard Autoencoder (Discrete)',
        'standard_vae': 'Standard VAE (β=1.0 - Continuous)',
        'beta_vae': 'β-VAE (β=4.0 - Disentangled)'
    }
    
    for model_type in ['standard_ae', 'standard_vae', 'beta_vae']:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_type}_trained.pth')
        try:
            models[model_type] = load_trained_model(
                checkpoint_path, model_type, args.latent_dim, 64, device
            )
            print(f"✅ {model_names[model_type]} loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {model_names[model_type]}: {e}")
            models[model_type] = None
    
    # Run interface based on available models
    if args.interface == 'gui':
        # For GUI, we'll use the originally designed two-model interface
        # but with the requested model and Standard AE as comparison
        primary_model = models[args.model_type]
        ae_model = models['standard_ae']
        
        if primary_model and ae_model:
            print(f"Starting interactive GUI explorer...")
            print(f"Primary model: {model_names[args.model_type]}")
            print(f"Comparison model: {model_names['standard_ae']}")
            
            explorer = InteractiveLatentExplorer(primary_model, ae_model, sample_images, device)
            explorer.run()
        else:
            print("❌ Could not load required models for GUI interface")
    else:
        # For matplotlib interface, use the requested model
        selected_model = models[args.model_type]
        if selected_model:
            print(f"Starting matplotlib slider interface...")
            print(f"Using: {model_names[args.model_type]}")
            demo = MatplotlibSliderDemo(selected_model, sample_images, device, 
                                      model_names[args.model_type])
        else:
            print(f"❌ Could not load {model_names[args.model_type]}")


if __name__ == '__main__':
    main()