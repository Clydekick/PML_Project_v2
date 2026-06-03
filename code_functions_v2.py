# Luke Grossmann, April 16, 2026

# code_functions_v2.py
import os
import time
import numpy as np
import pandas as pd
import torch

# plotting deps
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
import matplotlib.colors as mcolors

from pathlib import Path
from PIL import Image

__all__ = [
    "vector_to_matrix", # Converts voltage difference input vector to matrix
    "get_valid_indices", # Gets valid (row, col) positions for the 208 measurements
    "export_vector_and_matrix_example", # Saves a demo 208-vector and its 16x16 mapping to an Excel file with 2 sheets
    "plot_losses", # Original loss plot function from Smyl with tweaks by LG
    "plot_EIT", # Original EIT image plot function from Smyl with tweaks by LG
    "difference2matrix", # Converts (V2-V1) vector to 16x16 matrix for loss calculation. This applies to the experimental samples.
    "difference2matrix_standardized", # Same as difference2matrix but also standardizes the data.
    "standardized", # Standardizes (V2-V1) vector based on training mu, sigma.
    "crop_whitespace", # Crops extra white background from EIT images
    "batch_crop" # Crops all PNG files with 'eit' in the filename within `folder`
]

# Crops extra white background from EIT images
def crop_whitespace(in_path, out_path, tol=12, pad=6):
    """
    Crop near-white background from an image and save to out_path.

    tol: pixels with all RGB >= 255 - tol are treated as background.
    pad: extra pixels kept around the detected content.
    """
    im = Image.open(in_path).convert("RGBA")
    arr = np.asarray(im)

    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        near_white_bg = (rgb >= (255 - tol)).all(axis=-1) | (alpha <= tol)
    else:
        rgb = arr[..., :3]
        near_white_bg = (rgb >= (255 - tol)).all(axis=-1)

    fg_mask = ~near_white_bg
    if not np.any(fg_mask):
        # No foreground found; just copy the original
        im.save(out_path)
        return

    ys, xs = np.where(fg_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h, w = arr.shape[:2]
    y_min = max(0, y_min - pad); y_max = min(h - 1, y_max + pad)
    x_min = max(0, x_min - pad); x_max = min(w - 1, x_max + pad)

    cropped = arr[y_min:y_max+1, x_min:x_max+1]
    Image.fromarray(cropped).save(out_path)


from pathlib import Path

def batch_crop(folder, tol=12, pad=6, overwrite=True, make_backup=False):
    """
    Crop all PNG files with 'eit' in the filename within `folder`.
    If overwrite=True, saves back to the SAME file path.
    If overwrite=False, writes a '<name>_cropped.png' sibling instead.
    Optionally make a backup '<name>.orig.png' before overwriting.
    """
    folder = Path(folder)
    processed = 0

    for p in folder.glob("*.png"):
        if "eit" not in p.name.lower():
            continue

        if overwrite:
            if make_backup:
                backup = p.with_name(p.stem + ".orig" + p.suffix)
                if not backup.exists():
                    p.replace(backup)          # move original to backup
                    backup.replace(p)           # move it back so p still exists
            out = p
        else:
            out = p.with_stem(p.stem + "_cropped")

        crop_whitespace(p, out, tol=tol, pad=pad)
        print(f"Cropped: {p} -> {out}")
        processed += 1

    if processed == 0:
        print("No matching PNGs found (need '*.png' with 'eit' in the name).")
    else:
        print(f"Processed {processed} images.")
    return processed


    # ==== How to run ====
    # Example: process all matching images in "Figures" without overwriting originals
    # batch_crop("Figures", tol=12, pad=6, overwrite=False)

    # If you truly want to overwrite originals, set overwrite=True:
    # batch_crop("Figures", overwrite=True)


# Gets valid (row, col) positions for the 208 measurements
def get_valid_indices(device="cpu"):
    inj  = torch.arange(16, device=device)
    m    = torch.arange(13, device=device)
    rows = (inj[:, None] + 2 + m[None, :]) % 16   # (16,13)
    cols = inj[:, None].expand(16, 13)            # (16,13)
    rows208 = rows.reshape(-1)                    # (208,)
    cols208 = cols.reshape(-1)                    # (208,)
    return rows208, cols208

# -----------------------------
# Data layout helper
# -----------------------------
# Put this in code_functions_v2.py (replace the old version)
# Precompute the 208 -> (row,col) mapping once
_INJ = torch.arange(16)
_M   = torch.arange(13)
_ROWS_16x13 = (_INJ[:, None] + 2 + _M[None, :]) % 16        # (16,13)
_COLS_16x13 = _INJ[:, None].expand(16, 13)                  # (16,13)
_ROWS_208   = _ROWS_16x13.reshape(-1)                       # (208,)
_COLS_208   = _COLS_16x13.reshape(-1)                       # (208,)

@torch.no_grad()
# Converts voltage difference input vector to matrix
def vector_to_matrix(vec):
    """
    vec: shape (208,) or (B,208); can be numpy or torch
    returns: (16,16) or (B,16,16)
    """
    t = torch.as_tensor(vec)            # accepts numpy or torch
    if t.ndim == 1:
        if t.numel() != 208:
            raise ValueError("Expected a length-208 vector.")
        t = t[None, :]                  # (1,208) for unified code path

    if t.ndim != 2 or t.shape[1] != 208:
        raise ValueError("vec must be shape (208,) or (B, 208).")

    device = t.device
    dtype  = t.dtype

    # move indices to the same device once
    rows = _ROWS_208.to(device)
    cols = _COLS_208.to(device)

    B = t.shape[0]
    out = torch.zeros((B, 16, 16), dtype=dtype, device=device)
    # scatter values into their (row,col) positions for all B at once
    out[:, rows, cols] = t

    return out[0] if vec.ndim == 1 else out

# Old code for converting a 208-length vector to a 16x16 matrix.
'''
def vector_to_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    Convert a length-208 EIT vector into a 16x16 matrix.
    vec: shape (208,) or (B, 208)
    Returns: shape (16,16) or (B,16,16)
    """
    n_electrodes = 16
    n_meas = n_electrodes - 3  # 13 per injection

    if vec.ndim == 1:
        if vec.numel() != 208:
            raise ValueError("Expected a length-208 vector.")
        mat = torch.zeros((n_electrodes, n_electrodes), dtype=vec.dtype, device=vec.device)
        idx = 0
        for inj in range(n_electrodes):         # columns
            for m in range(n_meas):             # 13 valid rows
                row = (inj + 2 + m) % n_electrodes
                mat[row, inj] = vec[idx]
                idx += 1
        return mat

    if vec.ndim == 2:
        B, D = vec.shape
        if D != 208:
            raise ValueError("Expected input of shape (B, 208).")
        mats = torch.zeros((B, n_electrodes, n_electrodes), dtype=vec.dtype, device=vec.device)
        for b in range(B):
            mats[b] = vector_to_matrix(vec[b])
        return mats

    raise ValueError("vec must be shape (208,) or (B, 208).")
'''


# -----------------------------
# Convenience: export example to Excel
# -----------------------------
def export_vector_and_matrix_example(save_dir: str) -> str:
    """
    Save a demo 208-vector and its 16x16 mapping to an Excel file with 2 sheets.
    Returns the file path.
    """
    x = torch.arange(208, dtype=torch.float32)
    M = vector_to_matrix(x)

    df_vec = pd.DataFrame(x.cpu().numpy(), columns=["Vector (length 208)"])
    df_mat = pd.DataFrame(
        M.cpu().numpy(),
        index=[f"Row_{i+1}" for i in range(16)],
        columns=[f"Col_{j+1}" for j in range(16)],
    )

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "Matrix_example.xlsx")

    with pd.ExcelWriter(file_path, engine="openpyxl") as w:
        df_vec.to_excel(w, sheet_name="Flat_Vector_208", index=False)
        df_mat.to_excel(w, sheet_name="Matrix_16x16")

    return file_path

# -----------------------------
# Plot: losses
# -----------------------------
def plot_losses(
    epoch: int,
    train_losses,
    val_losses,
    #val_losses3,
    train_losses_pinn,
    train_losses_pinn_u,
    train_losses_pinn_du,
    train_losses_pinn_I,
    train_losses_BCE,
    val_losses_BCE,
    train_losses_bloss,
    #train_losses_alpha_controlled,  # LG: Added
    *,
    plot_save: bool = True,
    save_basename: str,
    save_directory
):
    """Plot training/validation losses on log scale."""
    plt.figure(figsize=(8, 7), dpi=110)
    plt.clf()

    sns.set_style("ticks")
    colors = sns.color_palette("bright", n_colors=9)

    plt.yscale("log", base=10)

    def plot_with_markers(data, label, color, linestyle, marker):
        epochs = range(1, len(data) + 1)
        plt.plot(epochs, data, label=label, color=color, linestyle=linestyle, marker=marker, markevery=5)
    
    plot_with_markers(train_losses,            r"Train Loss",                                colors[0], "-",  "o")
    plot_with_markers(train_losses_pinn,       r"Total Physics Loss",                        colors[1], "--", "s")
    plot_with_markers(train_losses_pinn_u,     r"$\beta_1 \|V_p - V_t\|_1$",                 colors[2], "-.", "^")
    plot_with_markers(train_losses_pinn_du,    r"$\beta_2 \|\partial V_p/\partial \Delta V_t\|_1$", colors[3], ":",  "D")
    plot_with_markers(train_losses_pinn_I,     r"$\beta_3 \|u_p - u_t\|_1$",                 colors[4], "-",  "x")
    plot_with_markers(val_losses,              r"Val Loss",                                  colors[5], "-",  "*")
    plot_with_markers(val_losses_BCE,          r"Val BCE Loss",                              colors[6], "-.", "s")
    plot_with_markers(train_losses_BCE,        r"Train BCE Loss",                            colors[7], "-.", "s")
    #plot_with_markers(val_losses3,             r"Val_losses3",              colors[8], "--", "v")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    out_paths = {}
    if plot_save == True:
        fig_path = os.path.join(save_directory or ".", f"{save_basename}")

        #basename = f"{save_basename}_epoch{epoch+1}"
        #fig_path = f"{basename}.png"
        #npz_path = f"{basename}.npz"
        npz_path = os.path.join(save_directory, f"{save_basename}")
        np.savez(
            f"{npz_path}.npz",
            train_losses=train_losses,
            val_losses=val_losses,
            #val_losses3=val_losses3,
            train_losses_pinn=train_losses_pinn,
            train_losses_pinn_u=train_losses_pinn_u,
            train_losses_pinn_du=train_losses_pinn_du,
            train_losses_pinn_I=train_losses_pinn_I,
            train_losses_BCE=train_losses_BCE,
            val_losses_BCE=val_losses_BCE,
            train_losses_bloss=train_losses_bloss,
            #train_losses_alpha_controlled=train_losses_alpha_controlled,  # LG: Added
        )
        plt.savefig(f"{fig_path}.png", dpi=300)
        out_paths = {"figure": f"{fig_path}.png", "npz": f"{npz_path}.npz"}
        plt.close()
        #print("Saved loss plot:", os.path.abspath(f"{fig_path}.png"))
    else:
        plt.show()
        plt.pause(0.001)

    return out_paths

# -----------------------------
# Plot: EIT field
# -----------------------------
def plot_EIT(
    x_hat,
    #gC,  # unused but kept for signature compatibility
    NodalCoords,
    ElemConnect,
    q: int,
    *,
    epoch: int | None = None,
    plot_save: bool = True,
    trial_number = None,
    save_directory: str,
    save_npz: bool = True,
    return_axes=False
):
    """Render conductivity over mesh; optionally save periodically."""
    ElemConnect = np.asarray(ElemConnect, dtype=int) - 1  # to 0-based
    x_hat = np.asarray(x_hat)
    NodalCoords = np.asarray(NodalCoords)

    norm = mcolors.Normalize(vmin=np.min(x_hat), vmax=np.max(x_hat))
    cmap = plt.get_cmap("plasma")

    plt.figure(num=q, figsize=(10, 8), dpi=140)
    plt.clf()
    ax = plt.gca()
    #ax.clear()

    for e in range(ElemConnect.shape[0]):
        indices = ElemConnect[e]
        polygon_coords = NodalCoords[indices]
        polygon = patches.Polygon(polygon_coords, closed=True, edgecolor="none", facecolor=cmap(norm(x_hat[e])))
        ax.add_patch(polygon)

    ax.set_xlim(NodalCoords[:, 0].min(), NodalCoords[:, 0].max())
    ax.set_ylim(NodalCoords[:, 1].min(), NodalCoords[:, 1].max())
    ax.set_aspect("equal", adjustable="box")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.3)
    cbar.set_label(r"$\sigma_P$", fontsize=22)

    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    
    if plot_save == True:
        if trial_number is not None:
            base = f"eit_epoch{epoch+1}_plot{q}_trial{trial_number}"
        else:
            base = f"eit_epoch{epoch+1}_plot{q}"
        
        fig_path = os.path.join(save_directory or ".", f"{base}")
        npz_path = os.path.join(save_directory or ".", f"{base}")
        #basename = f"{save_basename}_epoch{epoch+1}_plot{q}"
        plt.savefig(f"{fig_path}.png", dpi=300)
        if save_npz:
            np.savez(f"{npz_path}.npz", x_hat=x_hat, NodalCoords=NodalCoords, ElemConnect=ElemConnect)

        plt.close(q)
        #print("Saved EIT plot:", os.path.abspath(f"{fig_path}.png"))
    else:
        plt.show()
        plt.pause(0.001)

    if return_axes and plot_save==False:
        return ax
    # If we saved, return the file path; otherwise return figure/axes so callers can keep working interactively
    else:
        return (fig_path + ".png")


def difference2matrix(Vp_like, device):
    """
    Converts (V2-V1) vector to 16x16 matrix for loss calculation.
    Accepts: numpy array or torch.Tensor shaped (208,) or (208,1) etc.
    Returns: torch.Tensor [1, 1, 16, 16] on the current device.
    """
    # Make a torch tensor on the target device, without unnecessary copies
    t = torch.as_tensor(Vp_like, dtype=torch.float32, device=device).reshape(-1)  # [208]
    # vector_to_matrix already handles torch tensors (and numpy), returns [16,16] for 1D
    M = vector_to_matrix(t)            # [16,16]
    return M.unsqueeze(0).unsqueeze(0) # [1,1,16,16]

def difference2matrix_standardized(v_diff_like, mu, sigma, device=None):
    '''
    Converts (V2-V1) vector to 16x16 matrix for loss calculation with standardization.
    Standardization is based on training mu, sigma..
    '''
    if device is None:
        device = mu.device if isinstance(mu, torch.Tensor) else torch.device("cpu")
    v     = torch.as_tensor(v_diff_like, dtype=torch.float32, device=device).reshape(-1)  # [208]
    mu_t  = torch.as_tensor(mu,    dtype=torch.float32, device=device)
    sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    v = (v - mu_t) / sig_t
    M = vector_to_matrix(v)                                                           # [16,16]
    return M.unsqueeze(0).unsqueeze(0)                                                # [1,1,16,16]

def standardized(v_diff_like, mu, sigma, device=None):
    '''
    Standardizes (V2-V1) vector based on training mu, sigma.
    '''
    if device is None:
        device = mu.device if isinstance(mu, torch.Tensor) else torch.device("cpu")
    v     = torch.as_tensor(v_diff_like, dtype=torch.float32, device=device).reshape(-1)  # [208]
    mu_t  = torch.as_tensor(mu,    dtype=torch.float32, device=device)
    sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    v = (v - mu_t) / sig_t
    return v


def load_mesh_data():
    """
    Load mesh data from a .mat file.
    Returns NodalCoords, ElemConnect
    """
    import scipy.io
    folder = r"C:/Users/lagro/Documents/AFRL Code/Data_Files"

    temp = scipy.io.loadmat(os.path.join(folder, 'meshPINN.mat'))

    ElemConnect = temp["ElemConnect"]
    NodalCoords = temp["NodalCoords"]
    
    return NodalCoords, ElemConnect

def load_sig_samples():
    folder = r"C:\Users\lagro\Documents\AFRL Code\Data_Files\dVT"
    sig_samples = np.load(os.path.join(folder, 'sig.npy')) # other files: sig_samples_truss_scattered.npy, sig_samples_val_truss.npy, sig_samples_truss.npy, sig.npy (new data)
    ### Enforce binary conductivity [0,1] ###
    MINVAL = 10
    sig_samples[sig_samples < MINVAL] = 0
    sig_samples[sig_samples > MINVAL] = 1
    
    #### Select Number of Samples ####
    sp = sig_samples.shape[1]  # 50000 total structured samples
    print("Number of samples", sp)
    valsz = sp // 5 # validation samples

    # NN inputs (X) and outputs (Y) for training/validation
    Y = np.transpose(sig_samples[:, :sp])    # conductivity (sigma)
    #Y_val = np.transpose(sig_samples[:, sp-valsz:sp])    # conductivity
    #Z = np.transpose(Uel_samples[:, :sp-valsz])     # electrode potentials (V_t)
    #I = np.transpose(u_samples[:, :sp-valsz])       # internal potential (u_t)
    print("Y shape:", Y.shape)
    return Y


# -----------------------------
# Optional demo when run directly
# -----------------------------
if __name__ == "__main__":
    code_start_time = time.time()

    save_directory = r"C:\Users\lagro\Documents\AFRL Code\Training Conductivity Plots\plot_every_500_NEW_SIG"
    os.makedirs(save_directory, exist_ok=True)
    
    folder = r"C:\Users\lagro\Documents\AFRL Code\Training Conductivity Plots\plot_every_100_full_100k_SIG_SAMPLES_SCATTERED"
    batch_crop(folder)
    breakpoint()
    
    
    NodalCoords, ElemConnect = load_mesh_data()
    Y = load_sig_samples()
    
    # Analyze fraction of damaged elements (zeros) in the dataset
    vals = []
    for j in range(0, Y.shape[0], 1): #, 10_000, 20_000, 30_000, 40_000, 49_999]: #60_000, 70_000, 80_000, 90_000, 99_999]:
        frac_zero = (Y[j, :] == 0).mean()
        #if j % 500 == 0:
        #    print(f"sample {j:>6}: zeros={frac_zero:.4f}") # print the faction of damaged elements
        vals.append(frac_zero)
    
    # average fraction of damaged elements for this data file
    avg_frac_zero = float(np.mean(vals))
    print(f"\nAverage frac_zero over sampled rows = {avg_frac_zero:.4f}") 
    
    if False:
        trial_number = 0
        # Plot the first 100 validation samples
        # epoch will be 1..100 so it becomes the image number in filenames
        count = 0
        #for i, sigP in enumerate(Y[:100], start=0):
        for i in range(0, Y.shape[0], 500):               # i = 0, 100, 200, ... # do 25,000 and on for sig_samples_scattered
            sigP = Y[i, :]                                   # one training sample (per element)
            plot_EIT(
                x_hat=sigP,                # conductivity per element
                NodalCoords=NodalCoords,
                ElemConnect=ElemConnect,
                q=4,                       # 0 for validation, 1 for training, 4 for new training data
                epoch=i,                   # <-- image number in filename
                plot_save=True,
                trial_number=trial_number,         # or set an int if you want it in the filename
                save_directory=str(save_directory),
                save_npz=False
            )
            count += 1
    
    
    
    # Check for blobs vs. scattered data
    def build_element_adjacency(ElemConnect):
        EC = np.asarray(ElemConnect, dtype=int)
        # auto-handle 1-based meshes
        if EC.min() == 1:
            EC = EC - 1
        n_elem = EC.shape[0]
        # map each node -> list of incident elements
        node2elems = {}
        for e, nodes in enumerate(EC):
            for n in nodes:
                node2elems.setdefault(n, []).append(e)
        # neighbors: share >= 2 nodes (same edge)
        neigh = [set() for _ in range(n_elem)]
        for nodes in EC:
            # for each edge (pair) in this polygon
            for a in range(len(nodes)):
                n1, n2 = nodes[a], nodes[(a+1) % len(nodes)]
                # elements touching both n1 and n2
                s1 = set(node2elems[n1])
                s2 = set(node2elems[n2])
                edge_elems = s1 & s2
                for e1 in edge_elems:
                    for e2 in edge_elems:
                        if e1 != e2:
                            neigh[e1].add(e2)
                            neigh[e2].add(e1)
        return [np.fromiter(s, dtype=int) if s else np.array([], dtype=int) for s in neigh]

    def clustered_ratio(sample_bin, neighbors):
        # sample_bin: (n_elem,) with 0 = damaged, 1 = undamaged (per your thresholding)
        damaged = (sample_bin == 0)
        if damaged.sum() == 0:
            return 0.0
        count_clustered = 0
        for e, is_dmg in enumerate(damaged):
            if not is_dmg: continue
            nbrs = neighbors[e]
            if nbrs.size and np.any(damaged[nbrs]):
                count_clustered += 1
        return count_clustered / damaged.sum()

    # Build adjacency once
    neighbors = build_element_adjacency(ElemConnect)

    # Evaluate a bunch of samples quickly
    ratios = []
    for i in range(0, Y.shape[0], 1):     # every 100th sample
        ratios.append(clustered_ratio(Y[i, :], neighbors))
    print(f"Median clustered ratio over checked samples: {np.median(ratios):.2f}")
    
    
    if False:
        # Calculate Wasserstein distances
        from scipy.stats import wasserstein_distance

        # Compare distributions of zero ratios
        zeros_truss = (Y_truss == 0).mean(axis=1)
        zeros_val = (Y_val == 0).mean(axis=1)
        print("Wasserstein distance:", wasserstein_distance(zeros_truss, zeros_val))


    #demo_path = export_vector_and_matrix_example(r"C:\Users\lagro\Documents\AFRL Code\5.33 ARCHITECTURE_CNN_matrix")
    #print(f"Example Excel written to: {demo_path}")
    code_duration = (time.time() - code_start_time) / 3600
    print(f"Code run duration: {code_duration:.2f} hours")
