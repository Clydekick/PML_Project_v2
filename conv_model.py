# -*- coding: utf-8 -*-
"""
EIT Conductivity Distribution Predictor via CNN-Based Neural Network
Optimized with Unified Native PyTorch Tensors to Eliminate Array Conflicts

Dependencies: pip install -r requirements.txt
Paths:        edit config.py before running
"""

import os
import sys
import time
import random
import datetime
import scipy.io
import numpy as np
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

# Force matplotlib to use a non-interactive backend to prevent GUI hangs
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Import shared path configuration — edit config.py to update paths
from config import IMAGES_BASE_DIR, MATLAB_FOLDER, NPY_FOLDER

# Clear the terminal for a clean run environment
os.system('cls' if os.name == 'nt' else 'clear')

# =============================================================================
# HYPERPARAMETERS — Edit here for each run
# =============================================================================

# --- Output / Visualization ---
PLOT_SAVE        = True
SAVE_UPDATE_FREQ = 50        # How often (in epochs) to save plots

# --- Training ---
EPOCHS      = 50
LR          = 0.0008
GAMMA       = 0.92
BATCH_SIZE  = 64
PATIENCE    = 20
NUM_SAMPLES = 50000

# --- CNN Architecture ---
CNN_CHANNELS  = 128   # Number of convolution channels
CNN_KERNEL    = 7    # Convolution kernel size
BLOCKS        = 2    # Number of stacked ConvBlocks
DROPOUT_CONV  = 0.2  # Channel-wise dropout rate inside each ConvBlock (between the two conv layers)
DROPOUT_FINAL = 0.16  # Dropout rate applied after flatten, before the final linear projection

# =============================================================================

# Define local helper dependency paths (code_functions_v2.py is in the current working directory)
CURRENT_DIR = os.getcwd()
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from code_functions_v2 import plot_EIT, batch_crop


def set_seed(seed: int = 42):
    """Sets environment and PyTorch/NumPy seeds to ensure reproducible training runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_losses(epoch: int, train_losses, val_losses, save_basename: str, save_dir: str):
    """Plots training and validation loss curves on a log scale and saves metrics to disk."""
    plt.figure(figsize=(8, 7), dpi=110)
    plt.yscale("log", base=10)

    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="blue", linestyle="-", marker="o", markevery=1)
    plt.plot(epochs_range, val_losses, label="Val Loss", color="orange", linestyle="-", marker="*", markevery=1)

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, save_basename)
    np.savez(f"{fig_path}.npz", train_losses=train_losses, val_losses=val_losses)
    plt.savefig(f"{fig_path}.png", dpi=300)
    plt.close()


def weighted_bce_loss(pred, target, weight_for_zero=1.25, weight_for_one=1.0):
    """
    Computes binary cross entropy loss, penalizing errors on background pixels (zeros)
    more heavily to combat structural sparsity.
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    weights = torch.where(target == 0, weight_for_zero, weight_for_one)
    return (bce * weights).mean()


class ConvBlock(nn.Module):
    """
    1D residual convolutional block with BatchNorm, LeakyReLU, and channel-wise
    dropout between the two conv layers to prevent feature co-adaptation.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1   = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1     = nn.BatchNorm1d(channels)
        self.act     = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2     = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class CNN_Model(nn.Module):
    """
    1D CNN mapping raw voltage difference vectors to a dense 2D-mesh element
    conductivity distribution. Treats the input vector as a single-channel
    1D signal and processes it through stacked residual convolutional blocks.
    """
    def __init__(self, input_dim: int, output_dim: int, channels: int, kernel_size: int,
                 blocks: int = 3, dropout_conv: float = 0.0, dropout_final: float = 0.0):
        super().__init__()
        padding = kernel_size // 2

        # Project single-channel input into CNN feature space
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
        )
        # Stacked residual conv blocks — spatial length stays input_dim throughout
        self.blocks = nn.ModuleList([ConvBlock(channels, kernel_size, dropout_conv) for _ in range(blocks)])
        # Dropout applied after flatten, before the final linear projection
        self.dropout_final = nn.Dropout(dropout_final)
        # Maps flattened feature map to full element grid size
        self.proj_out = nn.Linear(channels * input_dim, output_dim)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)              # [B, 1, input_dim]
        x = self.stem(x)                # [B, channels, input_dim]
        for blk in self.blocks:
            x = blk(x)
        x = x.flatten(1)               # [B, channels * input_dim]
        x = self.dropout_final(x)
        return self.sigmoid(self.proj_out(x))


def run_conv_training(
    lr=LR,
    gamma=GAMMA,
    batchsz=BATCH_SIZE,
    epochs=EPOCHS,
    patience=PATIENCE,
    num_samples=NUM_SAMPLES,
    cnn_channels=CNN_CHANNELS,
    cnn_kernel=CNN_KERNEL,
    blocks=BLOCKS,
    dropout_conv=DROPOUT_CONV,
    dropout_final=DROPOUT_FINAL,
    images_base_dir=IMAGES_BASE_DIR,
    matlab_folder=MATLAB_FOLDER,
    npy_folder=NPY_FOLDER,
):
    """Main execution pipeline managing data ingestion, network optimization, and image rendering."""
    set_seed(42)
    save_dir = os.path.join(images_base_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    os.chdir(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on target compute device: {device}")

    # --- DATA INGESTION & STRUCTURAL MESH CONFIGURATION ---
    # Load foundational mesh node relationships for plotting
    mesh_data   = scipy.io.loadmat(os.path.join(matlab_folder, 'meshPINN.mat'))
    ElemConnect = np.double(mesh_data["ElemConnect"])
    NodalCoords = np.double(mesh_data["NodalCoords"])

    # Baseline evaluations configurations (No-damage baseline vs distinct damage variants)
    V1 = scipy.io.loadmat(os.path.join(matlab_folder, "Vd_mean_no_dam.mat"))["Vdmean"]
    validation_cases = {
        1: scipy.io.loadmat(os.path.join(matlab_folder, 'Vd_mean_single_dam.mat'))["Vdmean"],
        2: scipy.io.loadmat(os.path.join(matlab_folder, 'Vd_mean_double_dam.mat'))["Vdmean"],
        3: scipy.io.loadmat(os.path.join(matlab_folder, 'Vd_mean_triple_dam.mat'))["Vdmean"]
    }

    # Load high-capacity dataset arrays using memory maps to shield system RAM
    sig_mm = np.load(os.path.join(npy_folder, 'sig.npy'), mmap_mode='r')
    dVT_mm = np.load(os.path.join(npy_folder, 'dVT.npy'), mmap_mode='r')

    sig_samples = np.ascontiguousarray(sig_mm[:, :num_samples])
    dVT_samples = np.ascontiguousarray(dVT_mm[:, :num_samples])
    del sig_mm, dVT_mm  # Immediately flush memory maps from heap

    # Explicitly enforce clean binary targets (0 = undamaged background, 1 = structural defect)
    sig_samples[sig_samples < 10] = 0
    sig_samples[sig_samples > 10] = 1

    # --- TRAIN / VALIDATION DATA SPLITTING ---
    total_sp = sig_samples.shape[1]
    val_sz   = total_sp // 5
    train_sz = total_sp - val_sz

    X_train_raw = np.transpose(dVT_samples[:, :train_sz])
    Y_train_raw = np.transpose(sig_samples[:, :train_sz])
    X_val_raw   = np.transpose(dVT_samples[:, train_sz:total_sp])
    Y_val_raw   = np.transpose(sig_samples[:, train_sz:total_sp])

    # Inject Gaussian White Noise into input vectors to regularize optimization boundaries
    noise_std    = 0.001 * np.std(X_train_raw)
    X_train_raw += np.random.normal(loc=0.0, scale=noise_std, size=X_train_raw.shape)

    # Convert features AND labels to PyTorch tensors upfront to eliminate framework conflicts
    X_train = torch.as_tensor(X_train_raw, dtype=torch.float32)
    X_val   = torch.as_tensor(X_val_raw,   dtype=torch.float32)
    Y_train = torch.as_tensor(Y_train_raw, dtype=torch.float32)
    Y_val   = torch.as_tensor(Y_val_raw,   dtype=torch.float32)

    # --- INSTANTIATE MODEL & OPTIMIZER ---
    input_dim  = X_train.shape[-1]   # 208
    output_dim = Y_train.shape[-1]   # 2067

    model     = CNN_Model(input_dim, output_dim, cnn_channels, cnn_kernel, blocks, dropout_conv, dropout_final).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    torch.set_float32_matmul_precision('high')

    # --- MODEL PERFORMANCE TRACKING LABELS ---
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    worse_streak  = 0

    # Sub-sampling pipeline configurations (Dynamically capped for quick data configurations)
    train_subset_size = min(10240, X_train.shape[0])
    val_subset_size   = val_sz

    # --- CORE TRAINING LOOP ---
    for epoch in trange(epochs, desc="Training Progress"):
        # Dynamically draw randomized mini-batch slices every epoch
        train_idx = np.random.choice(X_train.shape[0], size=train_subset_size, replace=False)
        val_idx   = np.random.choice(X_val.shape[0],   size=val_subset_size,   replace=False)

        # Cast randomized extraction arrays to PyTorch Long tensors for safe indexing operations
        train_idx = torch.as_tensor(train_idx, dtype=torch.long)
        val_idx   = torch.as_tensor(val_idx,   dtype=torch.long)

        # Pure tensor-on-tensor slicing (No array translation required)
        Xs     = X_train[train_idx].to(device, non_blocking=True)
        Ys     = Y_train[train_idx].to(device, non_blocking=True)
        Xs_val = X_val[val_idx].to(device, non_blocking=True)
        Ys_val = Y_val[val_idx].to(device, non_blocking=True)

        train_loader = DataLoader(TensorDataset(Xs, Ys),         batch_size=batchsz, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xs_val, Ys_val), batch_size=batchsz, shuffle=False)

        # Batch Optimization Pass
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss  = weighted_bce_loss(preds, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        # Batch Validation Pass
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_Y_val in val_loader:
                val_preds       = model(batch_X_val)
                epoch_val_loss += weighted_bce_loss(val_preds, batch_Y_val).item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()

        # --- EARLY STOPPING CHECK ---
        if avg_val_loss > best_val_loss - 1e-5:
            worse_streak += 1
        else:
            best_val_loss = avg_val_loss
            worse_streak  = 0

        # --- PERIODIC VISUALIZATION AND PREDICTION RENDERING ---
        if (epoch + 1) % SAVE_UPDATE_FREQ == 0 or worse_streak >= patience:
            if PLOT_SAVE:
                plot_losses(epoch, train_losses, val_losses, f"loss_plot_epoch{epoch+1}", save_dir)

            # Generate forward predictions on standalone validation physical damage profiles
            for q_id, V_damage in validation_cases.items():
                diff_np = V_damage - V1
                Vp = torch.as_tensor(diff_np, dtype=torch.float32, device=device).reshape(1, -1)

                with torch.no_grad():
                    sigP = model(Vp).cpu().numpy().squeeze()

                if PLOT_SAVE:
                    plot_EIT(
                        x_hat=sigP,
                        NodalCoords=NodalCoords,
                        ElemConnect=ElemConnect,
                        q=q_id,
                        epoch=epoch,
                        plot_save=True,
                        trial_number=None,
                        save_directory=save_dir,
                    )

        if worse_streak >= patience:
            print(f"\n[Early Stop Triggered] No validation performance gains detected for {patience} consecutive epochs.")
            break

    # White-space image post-processing step
    batch_crop(save_dir)
    print("\nTraining run safely terminated.")

    # Return best validation loss
    return best_val_loss


if __name__ == "__main__":
    start_time = time.time()

    # -------------------------------------------------------------
    # SMOKE TEST CONFIGURATION: Change to high values for true runs
    # (Edit the constants at the top of this file, not here)
    # -------------------------------------------------------------
    run_conv_training(
        lr=LR,
        gamma=GAMMA,
        batchsz=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        num_samples=NUM_SAMPLES,
        cnn_channels=CNN_CHANNELS,
        cnn_kernel=CNN_KERNEL,
        blocks=BLOCKS,
        dropout_conv=DROPOUT_CONV,
        dropout_final=DROPOUT_FINAL,
    )

    print(f"Total calculation time: {(time.time() - start_time) / 60:.2f} minutes")
