import torch
import sys
import numpy as np

# Usage example:
if __name__ == "__main__":
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    if (len(sys.argv) == 2):
        checkpoint = torch.load(f'./CSRNETcheckpoints/checkpoint_epoch_{sys.argv[1]}.pth',
                                map_location=device, weights_only=False)
    else:
        checkpoint = torch.load('./CSRNETcheckpoints/best_model.pth',
                                map_location=device, weights_only=False)
    
    print("By val loss:", np.argmin(checkpoint["history"]["val_loss"]))
    print("By train loss:", np.argmin(checkpoint["history"]["train_loss"]))
    print("By val MAE:", np.argmin(checkpoint["history"]["val_mae"]))
    print("Val MAE:", np.min(checkpoint["history"]["val_mae"]))
    print("By val RMSE:", np.argmin(checkpoint["history"]["val_rmse"]))
    print("Val RMSE:", np.min(checkpoint["history"]["val_rmse"]))