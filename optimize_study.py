# -*- coding: utf-8 -*-
"""
EIT Hyperparameter Optimization Orchestrator via Optuna
Tunes CNN architecture and training hyperparameters for conductivity prediction.

Dependencies: pip install -r requirements.txt
Paths:        edit config.py before running
"""

import datetime
import torch
import optuna

# Import shared path configuration — edit config.py to update paths
from config import IMAGES_BASE_DIR, MATLAB_FOLDER, NPY_FOLDER

from conv_model import run_conv_training

# -------------------------------------------------------------------------
# 0. TRIAL EXECUTION
# -------------------------------------------------------------------------
trials   = 30
epochs   = 50
patience = 10

def objective(trial):
    # -------------------------------------------------------------------------
    # 1. STANDARD OPTIMIZATION HYPERPARAMETERS
    # -------------------------------------------------------------------------
    lr      = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gamma   = trial.suggest_float("gamma", 0.90, 0.99)
    batchsz = trial.suggest_categorical("batchsz", [64, 128])

    # -------------------------------------------------------------------------
    # 2. CNN ARCHITECTURAL CONFIGURATION
    # -------------------------------------------------------------------------
    cnn_channels  = trial.suggest_int("cnn_channels", 32, 128, step=16)
    cnn_kernel    = trial.suggest_categorical("cnn_kernel", [3, 5, 7, 9])
    blocks        = trial.suggest_int("blocks", 2, 5)

    # -------------------------------------------------------------------------
    # 3. REGULARIZATION
    # -------------------------------------------------------------------------
    dropout_conv  = trial.suggest_float("dropout_conv",  0.0, 0.5)
    dropout_final = trial.suggest_float("dropout_final", 0.0, 0.5)

    print(f"\n[Trial {trial.number}] CNN: Channels={cnn_channels}, Kernel={cnn_kernel}, Blocks={blocks}, "
          f"lr={lr:.2e}, gamma={gamma:.3f}, batch={batchsz}, "
          f"dropout_conv={dropout_conv:.2f}, dropout_final={dropout_final:.2f}")

    try:
        best_val_loss = run_conv_training(
            lr=lr,
            gamma=gamma,
            batchsz=batchsz,
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            blocks=blocks,
            dropout_conv=dropout_conv,
            dropout_final=dropout_final,
            epochs=epochs,        # Capped epochs for efficient parameter sweeps
            patience=patience,    # Slightly lower patience to match shortened trial epochs
            images_base_dir=IMAGES_BASE_DIR,
            matlab_folder=MATLAB_FOLDER,
            npy_folder=NPY_FOLDER,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            print(f"⚠️ Trial {trial.number} failed due to CUDA Out of Memory. Penalizing trial.")
            return float("inf")
        raise e

    print(f"[Trial {trial.number}] Finished: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return best_val_loss


if __name__ == "__main__":
    # Using an SQLite database file preserves trial results if the run is interrupted
    db_path = "sqlite:///eit_optimization_study.db"

    study = optuna.create_study(
        study_name="eit_cnn_study",
        storage=db_path,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler()#seed=42)
    )

    print("🚀 Starting CNN Hyperparameter Optimization Study...")

    # Run the study across a total budget of 30 evaluation iterations
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    # -------------------------------------------------------------------------
    # REPORT STUDY RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("🏆 STUDY COMPLETE 🏆")
    print("="*60)
    print(f"Best Target Loss: {study.best_trial.value:.6f}")
    print("\nOptimal Parameter Selection Configuration:")
    for param_key, param_val in study.best_trial.params.items():
        print(f"  • {param_key:<15}: {param_val}")
    print("="*60)
