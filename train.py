# @title
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.ndimage
import cv2
import os
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

class JHUCrowdDataset(Dataset):
    def __init__(self, data_root, split='train', use_edges=True,
                 edge_method='sobel', crop_size=512, downsample_ratio=8):
        """
        Args:
            data_root: Path to jhu_crowd_v2.0 directory (e.g., '/content/data')
            split: 'train', 'val', or 'test'
            use_edges: Whether to add edge channel
            edge_method: 'sobel' or 'canny'
            crop_size: Size to crop images (for training)
            downsample_ratio: Ratio for density map (CSRNet uses 8)
        """
        self.data_root = data_root
        self.split = split
        self.use_edges = use_edges
        self.edge_method = edge_method
        self.crop_size = crop_size
        self.downsample_ratio = downsample_ratio

        # Paths
        self.img_dir = os.path.join(data_root, split, 'images')
        self.gt_dir = os.path.join(data_root, split, 'gt')

        # Get list of images
        self.img_files = sorted([f for f in os.listdir(self.img_dir)
                                 if f.endswith('.jpg')])

        print(f"Loaded {len(self.img_files)} images from {split} split")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)

        # Load ground truth points
        gt_name = img_name.replace('.jpg', '.txt')
        gt_path = os.path.join(self.gt_dir, gt_name)
        points = self.load_gt_points(gt_path)

        # Apply random crop for training
        if self.split == 'train' and self.crop_size:
            img_np, points = self.random_crop(img_np, points, self.crop_size)

        # Generate density map
        h, w = img_np.shape[:2]
        density_map = self.generate_density_map((h, w), points)

        # Extract edges if needed
        if self.use_edges:
            edge_channel = self.extract_edges(img_np)
            # Concatenate as 4th channel
            img_np = np.concatenate([img_np, edge_channel[..., None]], axis=2)

        # Normalize image to [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Convert density map to tensor
        density_tensor = torch.from_numpy(density_map).unsqueeze(0).float()

        return img_tensor, density_tensor

    def load_gt_points(self, gt_path):
        """Load ground truth head locations from file"""
        points = []
        if not os.path.exists(gt_path):
            return np.array(points)

        with open(gt_path, 'r') as f:
            for line in f:
                # Format: x y w h o b (space separated)
                values = line.strip().split()
                if len(values) >= 2:
                    x, y = float(values[0]), float(values[1])
                    points.append([x, y])

        return np.array(points)

    def generate_density_map(self, shape, points, sigma=15):
        """
        Generate gaussian density map from point annotations
        Uses adaptive sigma based on k-nearest neighbors for better results
        """
        h, w = shape
        density = np.zeros((h, w), dtype=np.float32)

        if len(points) == 0:
            return density

        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])

            # Check bounds
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            # Adaptive sigma (optional but recommended)
            # Use fixed sigma for simplicity, or implement adaptive version
            sigma_adaptive = sigma

            # Create gaussian kernel
            size = int(6 * sigma_adaptive)
            if size % 2 == 0:
                size += 1

            # Generate 2D gaussian
            ax = np.arange(-size // 2 + 1., size // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_adaptive**2))

            # Normalize the kernel
            kernel = kernel / kernel.sum()

            # Add to density map
            x_start = max(0, x - size // 2)
            x_end = min(w, x + size // 2 + 1)
            y_start = max(0, y - size // 2)
            y_end = min(h, y + size // 2 + 1)

            kernel_x_start = size // 2 - (x - x_start)
            kernel_x_end = kernel_x_start + (x_end - x_start)
            kernel_y_start = size // 2 - (y - y_start)
            kernel_y_end = kernel_y_start + (y_end - y_start)

            density[y_start:y_end, x_start:x_end] += \
                kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]

        return density

    def extract_edges(self, image):
        """Extract edge channel using Sobel or Canny"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.edge_method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            # Normalize to 0-255
            magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
            return magnitude

        elif self.edge_method == 'canny':
            edges = cv2.Canny(gray, 50, 150)
            return edges

        else:
            raise ValueError(f"Unknown edge method: {self.edge_method}")

    def random_crop(self, img, points, crop_size):
        """Randomly crop image and adjust point coordinates"""
        h, w = img.shape[:2]

        # If image is smaller than crop size, pad it
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            h, w = img.shape[:2]

        # Random crop position
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)

        # Crop image
        img_cropped = img[y:y+crop_size, x:x+crop_size]

        # Adjust points
        if len(points) > 0:
            points_cropped = points.copy()
            points_cropped[:, 0] -= x
            points_cropped[:, 1] -= y

            # Keep only points within crop
            mask = (points_cropped[:, 0] >= 0) & (points_cropped[:, 0] < crop_size) & \
                   (points_cropped[:, 1] >= 0) & (points_cropped[:, 1] < crop_size)
            points_cropped = points_cropped[mask]
        else:
            points_cropped = points

        return img_cropped, points_cropped
    
# Create data loaders
def create_data_loaders(data_root, batch_size=4, use_edges=True,
                       edge_method='canny', crop_size=512):
    """
    Create train, val, and test data loaders
    """
    train_dataset = JHUCrowdDataset(
        data_root=data_root,
        split='train',
        use_edges=use_edges,
        edge_method=edge_method,
        crop_size=crop_size
    )

    val_dataset = JHUCrowdDataset(
        data_root=data_root,
        split='val',
        use_edges=use_edges,
        edge_method=edge_method,
        crop_size=None  # No cropping for validation
    )

    test_dataset = JHUCrowdDataset(
        data_root=data_root,
        split='test',
        use_edges=use_edges,
        edge_method=edge_method,
        crop_size=None  # No cropping for test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for validation (variable image sizes)
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

class CSRNet(nn.Module):
    def __init__(self, load_weights=False, input_channels=4):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.input_channels = input_channels
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        # Create frontend with custom input channels
        self.frontend = make_layers(self.frontend_feat, in_channels=input_channels)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._initialize_weights()
            if input_channels == 3:
                # Load pretrained VGG weights for RGB
                mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                self._copy_vgg_weights(mod)
            elif input_channels == 4:
                # Load pretrained VGG weights and handle 4th channel
                mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                self._copy_vgg_weights_with_extra_channel(mod)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _copy_vgg_weights(self, vgg_model):
        """Copy VGG-16 weights to frontend (for 3-channel input)"""
        frontend_dict = self.frontend.state_dict()
        vgg_dict = vgg_model.features.state_dict()

        # Match keys and copy weights
        vgg_keys = list(vgg_dict.keys())
        frontend_keys = list(frontend_dict.keys())

        for i in range(len(frontend_keys)):
            if i < len(vgg_keys):
                frontend_dict[frontend_keys[i]] = vgg_dict[vgg_keys[i]]

        self.frontend.load_state_dict(frontend_dict)

    def _copy_vgg_weights_with_extra_channel(self, vgg_model):
        """Copy VGG-16 weights and initialize 4th channel (for 4-channel input)"""
        vgg_dict = vgg_model.features.state_dict()
        frontend_dict = self.frontend.state_dict()

        for key in frontend_dict.keys():
            if key in vgg_dict.keys():
                if 'weight' in key and frontend_dict[key].shape != vgg_dict[key].shape:
                    # First conv layer - need to handle 4 channels
                    # Copy RGB weights
                    frontend_dict[key][:, :3, :, :] = vgg_dict[key]
                    # Initialize 4th channel as mean of RGB channels
                    frontend_dict[key][:, 3:, :, :] = vgg_dict[key].mean(dim=1, keepdim=True)
                else:
                    frontend_dict[key] = vgg_dict[key]

        self.frontend.load_state_dict(frontend_dict)

def make_layers(cfg, in_channels=4, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, density_maps in pbar:
        images = images.to(device)
        density_maps = density_maps.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Resize density map if needed to match output size
        if outputs.shape != density_maps.shape:
            density_maps = torch.nn.functional.interpolate(
                density_maps, size=outputs.shape[2:], mode='bilinear', align_corners=False
            )

        loss = criterion(outputs, density_maps)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    mae = 0.0  # Mean Absolute Error
    mse = 0.0  # Mean Squared Error

    with torch.no_grad():
        for images, density_maps in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images)

            if outputs.shape != density_maps.shape:
                density_maps = torch.nn.functional.interpolate(
                    density_maps, size=outputs.shape[2:], mode='bilinear', align_corners=False
                )

            loss = criterion(outputs, density_maps)
            running_loss += loss.item()

            # Calculate counting metrics
            pred_count = outputs.sum(dim=(1,2,3))
            gt_count = density_maps.sum(dim=(1,2,3))
            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += ((pred_count - gt_count) ** 2).sum().item()

    avg_loss = running_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_rmse = np.sqrt(mse / len(dataloader.dataset))

    return avg_loss, avg_mae, avg_rmse

def save_checkpoint(model, optimizer, epoch, val_loss, path, train_loss, val_mae, val_rmse, history):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'val_mae': val_mae,
        'val_rmse': val_rmse
    }

    # Add entire history if provided
    if history is not None:
        checkpoint['history'] = history

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    history = checkpoint.get('history', None)
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return epoch, loss, history

def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE
    axes[0, 1].plot(history['val_mae'], label='Val MAE', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # RMSE
    axes[1, 0].plot(history['val_rmse'], label='Val RMSE', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Root Mean Squared Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Combined metrics
    axes[1, 1].plot(history['val_mae'], label='MAE', color='orange')
    axes[1, 1].plot(history['val_rmse'], label='RMSE', color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_title('Combined Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_csrnet(
    data_root,
    batch_size=4,
    crop_size=512,
    num_epochs=200,
    learning_rate=1e-6,
    weight_decay=5e-4,
    checkpoint_dir='checkpoints',
    resume_from=None,
    edge_method='canny'
):
    """
    Main training function for CSRNet on JHU Crowd dataset

    Args:
        data_root: Path to jhu_crowd_v2.0 directory
        batch_size: Batch size for training
        crop_size: Size to crop training images
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        edge_method: 'canny' or 'sobel' for edge detection
    """

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=data_root,
        batch_size=batch_size,
        use_edges=True,
        edge_method=edge_method,
        crop_size=crop_size
    )

    # Model, loss, optimizer
    print("Initializing model...")
    model = CSRNet(load_weights=False, input_channels=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = None

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch, _, loaded_history = load_checkpoint(model, optimizer, resume_from)
        start_epoch += 1

        if loaded_history is not None:
            history = loaded_history
            print(f"Resumed with {len(history['train_loss'])} epochs of history")

    # Training history
    if history is None:
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': []
        }

    print(history['train_loss'])

    best_mae = float('inf')

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, train_loss, val_mae, val_rmse, history)

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, train_loss, val_mae, val_rmse, history)
            print(f"★ New best model! MAE: {best_mae:.2f}")

        # Plot training history every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_path = os.path.join(checkpoint_dir, 'training_history.png')
            plot_training_history(history, plot_path)

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best MAE: {best_mae:.2f}")

    # Final plot
    plot_path = os.path.join(checkpoint_dir, 'final_training_history.png')
    plot_training_history(history, plot_path)

    return model, history


# ============================================================================
# FUNCTION CALL
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_ROOT = './content/data/jhu_crowd_v2.0'
    CHECKPOINT_DIR = './CSRNETcheckpoints'
    BATCH_SIZE = 4
    CROP_SIZE = 512
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 5e-4
    EDGE_METHOD = 'canny'  # or 'sobel'

    # Train model
    model, history = train_csrnet(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        crop_size=CROP_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=CHECKPOINT_DIR,
        edge_method=EDGE_METHOD,
        resume_from=None
        #resume_from='./CSRNETcheckpoints/checkpoint_epoch_2.pth' #set to path to a checkpoint to resume from that checkpoint
    )

    