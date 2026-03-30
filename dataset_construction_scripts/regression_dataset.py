import h5py
import numpy as np
from sklearn.model_selection import train_test_split

file_path = "Dataset_Specific_labelled_full_only_for_2i.h5"

# ── Step 1: Load ──────────────────────────────────────────────────────────────
with h5py.File(file_path, "r") as f:
    X    = f["jet"][:].astype(np.float32)   # (10000, 125, 125, 8)
    y    = f["Y"][:].reshape(-1).astype(np.int64)        # (10000,)
    mass = f["m"][:].reshape(-1).astype(np.float32)      # (10000,)
    pt   = f["pT"][:].reshape(-1).astype(np.float32)     # (10000,)

print("Loaded:")
print(f"  X:    {X.shape}    dtype={X.dtype}")
print(f"  y:    {y.shape}    classes={np.unique(y)}")
print(f"  mass: {mass.shape} min={mass.min():.2f} max={mass.max():.2f} mean={mass.mean():.2f}")
print(f"  pT:   {pt.shape}   min={pt.min():.2f}   max={pt.max():.2f}   mean={pt.mean():.2f}")

# ── Step 2: Same image preprocessing as Phase 1 ───────────────────────────────
X = np.log1p(X)
X = X / (np.max(X, axis=(1, 2, 3), keepdims=True) + 1e-8)
print("\nImage preprocessing done (log1p + per-sample max norm)")

# ── Step 3: 80/20 stratified split ───────────────────────────────────────────
idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                   random_state=42,
                                   stratify=y)
print(f"\nSplit: train={len(idx_tr)}  test={len(idx_te)}")
print(f"  Train class dist: {np.bincount(y[idx_tr])}")
print(f"  Test  class dist: {np.bincount(y[idx_te])}")

# ── Step 4: Normalise mass and pT using training stats only ───────────────────
mass_mean, mass_std = mass[idx_tr].mean(), mass[idx_tr].std()
pt_mean,   pt_std   = pt[idx_tr].mean(),   pt[idx_tr].std()

mass_norm = (mass - mass_mean) / (mass_std + 1e-8)
pt_norm   = (pt   - pt_mean)   / (pt_std   + 1e-8)

print(f"\nRegression target stats (training split):")
print(f"  mass: mean={mass_mean:.4f}  std={mass_std:.4f}")
print(f"  pT:   mean={pt_mean:.4f}    std={pt_std:.4f}")

# ── Step 5: Save ──────────────────────────────────────────────────────────────
# Images
np.save("X_train_mt.npy",  X[idx_tr])
np.save("X_test_mt.npy",   X[idx_te])

# Class labels
np.save("y_train_mt.npy",  y[idx_tr])
np.save("y_test_mt.npy",   y[idx_te])

# Raw regression targets (GeV) — for reporting metrics
np.save("mass_train_raw.npy", mass[idx_tr])
np.save("mass_test_raw.npy",  mass[idx_te])
np.save("pt_train_raw.npy",   pt[idx_tr])
np.save("pt_test_raw.npy",    pt[idx_te])

# Normalised regression targets — for training
np.save("mass_train_norm.npy", mass_norm[idx_tr])
np.save("mass_test_norm.npy",  mass_norm[idx_te])
np.save("pt_train_norm.npy",   pt_norm[idx_tr])
np.save("pt_test_norm.npy",    pt_norm[idx_te])

# Normalisation stats — needed to convert predictions back to GeV
np.save("norm_stats.npy", np.array([mass_mean, mass_std, pt_mean, pt_std]))

# ── Step 6: Verify ────────────────────────────────────────────────────────────
print("\nSaved files:")
files = [
    "X_train_mt.npy",    "X_test_mt.npy",
    "y_train_mt.npy",    "y_test_mt.npy",
    "mass_train_raw.npy","mass_test_raw.npy",
    "pt_train_raw.npy",  "pt_test_raw.npy",
    "mass_train_norm.npy","mass_test_norm.npy",
    "pt_train_norm.npy", "pt_test_norm.npy",
    "norm_stats.npy",
]
for fname in files:
    arr = np.load(fname)
    print(f"  {fname:25s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")

print("\nTo load in the multitask notebook:")
print("  X_train    = np.load('X_train_mt.npy')")
print("  y_train    = np.load('y_train_mt.npy')")
print("  mass_train = np.load('mass_train_norm.npy')  # normalised, use for training")
print("  pt_train   = np.load('pt_train_norm.npy')    # normalised, use for training")
print("  stats      = np.load('norm_stats.npy')")
print("  mass_mean, mass_std, pt_mean, pt_std = stats")
print("  # To convert predictions back to GeV:")
print("  mass_pred_GeV = mass_pred_norm * mass_std + mass_mean")
print("  pt_pred_GeV   = pt_pred_norm   * pt_std   + pt_mean")

