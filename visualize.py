import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from train import CSRNet, JHUCrowdDataset

def visualize_predictions(model, dataset, device, num_samples=5, save_path='predictions.png'):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained CSRNet model
        dataset: Dataset to sample from (val_dataset or test_dataset)
        device: torch device
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get random sample
            #idx = np.random.randint(0, len(dataset))
            idx = i
            image, gt_density = dataset[idx]
            
            # Prepare input
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            pred_density = model(image_input)
            
            # Resize prediction to match ground truth if needed
            if pred_density.shape != gt_density.unsqueeze(0).shape:
                pred_density = torch.nn.functional.interpolate(
                    pred_density, 
                    size=gt_density.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Convert to numpy
            image_np = image[:3].permute(1, 2, 0).cpu().numpy()  # RGB only
            gt_density_np = gt_density.squeeze().cpu().numpy()
            pred_density_np = pred_density.squeeze().cpu().numpy()
            
            # Calculate counts
            gt_count = gt_density_np.sum()
            pred_count = pred_density_np.sum()
            error = abs(gt_count - pred_count)
            
            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_density_np, cmap='jet')
            axes[i, 1].set_title(f'GT Density\nCount: {gt_count:.1f}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_density_np, cmap='jet')
            axes[i, 2].set_title(f'Predicted Density\nCount: {pred_count:.1f}')
            axes[i, 2].axis('off')
            
            # Overlay prediction on image
            axes[i, 3].imshow(image_np)
            axes[i, 3].imshow(pred_density_np, cmap='jet', alpha=0.5)
            axes[i, 3].set_title(f'Overlay\nError: {error:.1f}')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions saved to {save_path}")
    #plt.show()

# Usage example:
# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CSRNet(load_weights=False, input_channels=4).to(device)

# Load checkpoint
if (len(sys.argv) == 2):
    checkpoint = torch.load(f'./CSRNETcheckpoints/checkpoint_epoch_{sys.argv[1]}.pth', map_location=device, weights_only=False)
else:
    checkpoint = torch.load('./CSRNETcheckpoints/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Create dataset
val_dataset = JHUCrowdDataset(
    data_root='./content/data/jhu_crowd_v2.0',
    split='val',
    use_edges=True,
    edge_method='canny',
    crop_size=None
)

# Visualize
if (len(sys.argv) == 2):
    visualize_predictions(model, val_dataset, device, num_samples=3, save_path=f"./predictions/val_predictions_epoch_{sys.argv[1]}.png")
else:
    visualize_predictions(model, val_dataset, device, num_samples=3, save_path=f"./predictions/val_predictions.png")