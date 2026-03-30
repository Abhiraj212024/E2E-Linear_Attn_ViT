import h5py
import numpy as np
from sklearn.model_selection import train_test_split

file_path = "Dataset_Specific_Labelled.h5"

# -------------------------------
# Step 1: Load data (lazy → then convert)
# -------------------------------
with h5py.File(file_path, "r") as f:
    X = f["jet"][:]   # shape: (N, 125, 125, 8)
    y = f["Y"][:].reshape(-1)  # shape: (N,)

print("Loaded data:", X.shape, y.shape)

# -------------------------------
# Step 2: Apply log1p per channel
# -------------------------------
# vectorized (fast)
X = np.log1p(X)

# -------------------------------
# Step 3: Normalize (recommended)
# -------------------------------
# normalize per sample (robust for sparse data)
X = X / (np.max(X, axis=(1,2,3), keepdims=True) + 1e-8)

# -------------------------------
# Step 4: Train-test split (80-20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # important for classification
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# -------------------------------
# Step 5: Save as .npy files
# -------------------------------
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Saved all files successfully.")