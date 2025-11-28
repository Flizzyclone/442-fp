import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from train import CSRNet, JHUCrowdDataset


def denormalize_image(image_tensor):
    """
    Denormalize ImageNet-normalized image for visualization

    Args:
        image_tensor: Tensor with shape [3, H, W] normalized with ImageNet stats

    Returns:
        Numpy array with shape [H, W, 3] with values in [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Extract RGB channels (first 3 channels)
    image_np = image_tensor[:3].cpu().numpy()  # [3, H, W]

    # Denormalize
    for i in range(3):
        image_np[i] = image_np[i] * std[i] + mean[i]

    # Transpose to [H, W, 3]
    image_np = np.transpose(image_np, (1, 2, 0))

    # Clip to [0, 1] range
    image_np = np.clip(image_np, 0, 1)

    return image_np


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

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(num_samples):
            # Get sample (sequential, not random for reproducibility)
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

            # Convert to numpy - PROPERLY DENORMALIZE
            image_np = denormalize_image(image)  # Now in [0, 1] range
            gt_density_np = gt_density.squeeze().cpu().numpy()
            pred_density_np = pred_density.squeeze().cpu().numpy()

            # Calculate counts
            gt_count = gt_density_np.sum()
            pred_count = pred_density_np.sum()
            error = abs(gt_count - pred_count)

            # Get min/max for consistent color scaling
            vmin = min(gt_density_np.min(), pred_density_np.min())
            vmax = max(gt_density_np.max(), pred_density_np.max())

            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'Input Image')
            axes[i, 0].axis('off')

            im1 = axes[i, 1].imshow(gt_density_np, cmap='jet', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'GT Density\nCount: {gt_count:.1f}')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

            im2 = axes[i, 2].imshow(pred_density_np, cmap='jet', vmin=vmin, vmax=vmax)
            axes[i, 2].set_title(f'Predicted Density\nCount: {pred_count:.1f}')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

            # Overlay prediction on image
            axes[i, 3].imshow(image_np)
            # Only overlay positive values for better visualization
            pred_overlay = np.ma.masked_where(pred_density_np <= 0, pred_density_np)
            axes[i, 3].imshow(pred_overlay, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
            axes[i, 3].set_title(f'Overlay\nError: {error:.1f}')
            axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions saved to {save_path}")
    # plt.show()


# Usage example:
if __name__ == "__main__":
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSRNet(load_weights=False, input_channels=4).to(device)

    # Load checkpoint
    if (len(sys.argv) == 2):
        checkpoint = torch.load(f'./CSRNETcheckpoints/checkpoint_epoch_{sys.argv[1]}.pth',
                                map_location=device, weights_only=False)
    else:
        checkpoint = torch.load('./CSRNETcheckpoints/best_model.pth',
                                map_location=device, weights_only=False)
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
        visualize_predictions(model, val_dataset, device, num_samples=3,
                              save_path=f"./predictions/val_predictions_epoch_{sys.argv[1]}.png")
    else:
        visualize_predictions(model, val_dataset, device, num_samples=3,
                              save_path=f"./predictions/val_predictions.png")