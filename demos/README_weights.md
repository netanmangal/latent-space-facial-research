# Using Trained Model Weights

This guide explains how to use the trained model weights exported from the research notebook in your demonstrations and experiments.

## 🎯 Workflow Overview

1. **Train Models**: Run `notebooks/research_analysis.ipynb` to train all three models
2. **Export Weights**: Models are automatically saved to `models/checkpoints/` 
3. **Use in Demos**: Demos automatically load trained weights for better results

## 📁 Checkpoint Structure

After training, you'll have these files:
```
models/checkpoints/
├── standard_ae_trained.pth      # Standard Autoencoder weights
├── standard_vae_trained.pth     # Standard VAE (β=1.0) weights  
└── beta_vae_trained.pth         # β-VAE (β=4.0) weights
```

Each checkpoint contains:
- `model_state_dict`: Trained model weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `train_losses`: Training loss history
- `recon_losses`: Reconstruction loss history (VAEs only)
- `kl_losses`: KL divergence loss history (VAEs only)
- `model_type`: Model identifier
- `beta`: Beta parameter (VAEs only)
- `latent_dim`: Latent dimension size
- `image_size`: Input image size
- `num_epochs`: Number of training epochs

## 🚀 Using Trained Models in Demos

### Interactive Demo
```bash
# Use trained Standard VAE
python demos/interactive_demo.py --model_type standard_vae

# Use trained β-VAE
python demos/interactive_demo.py --model_type beta_vae

# Use trained Standard Autoencoder
python demos/interactive_demo.py --model_type standard_ae
```

### Three-Model Demo
```bash
# Run comprehensive demo with all trained models
python demos/three_model_demo.py

# Specify custom checkpoint directory
python demos/three_model_demo.py --checkpoint_dir path/to/checkpoints
```

## 🔧 Loading Models in Custom Code

```python
import torch
from models.vae_model import StandardVAE, BetaVAE
from models.autoencoder_model import StandardAutoencoder

def load_trained_model(checkpoint_path, model_type, device='cpu'):
    \"\"\"Load a trained model from checkpoint\"\"\"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
standard_vae = load_trained_model('models/checkpoints/standard_vae_trained.pth', 'standard_vae', device)
```

## 📊 Training Metrics Access

```python
# Load training history from checkpoint
checkpoint = torch.load('models/checkpoints/standard_vae_trained.pth')

# Access training metrics
train_losses = checkpoint['train_losses']
recon_losses = checkpoint['recon_losses']  
kl_losses = checkpoint['kl_losses']

# Plot training curves
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Total Loss')

plt.subplot(1, 3, 2)
plt.plot(recon_losses)
plt.title('Reconstruction Loss')

plt.subplot(1, 3, 3)
plt.plot(kl_losses)
plt.title('KL Divergence')

plt.tight_layout()
plt.show()
```

## ⚠️ Important Notes

1. **Model Architecture Consistency**: Trained weights expect specific model architectures. The demos automatically handle this.

2. **Device Compatibility**: Checkpoints are saved with `map_location` support for CPU/GPU compatibility.

3. **Fallback Behavior**: If trained weights aren't found, demos automatically fall back to untrained models with a warning.

4. **Fair Comparison**: Standard VAE and β-VAE use identical architectures (only β parameter differs) for fair experimental comparison.

## 🎯 Research Benefits

Using trained weights provides:
- **Realistic Demonstrations**: See actual learned representations
- **Meaningful Interpolations**: Smooth transitions between real facial features
- **Semantic Traversals**: Controlled attribute manipulation
- **Quality Reconstructions**: High-fidelity image generation
- **Proper Comparisons**: Fair evaluation across model types

Train your models first in the research notebook, then enjoy the full power of your three-model comparison system! 🚀