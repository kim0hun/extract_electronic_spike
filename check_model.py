import torch
import joblib
import numpy as np
from pathlib import Path

COL = "F10_01"
MODEL_DIR = Path(f"./model/{COL}")

# =========================
# 1. scaler.pkl 조회
# =========================
scaler_path = MODEL_DIR / "scaler.pkl"
scaler = joblib.load(scaler_path)

print("========== scaler.pkl ==========")
print("path:", scaler_path)
print("class:", scaler.__class__.__module__ + "." + scaler.__class__.__name__)

for name in [
    "feature_range",
    "n_features_in_",
    "feature_names_in_",
    "data_min_",
    "data_max_",
    "data_range_",
    "scale_",
    "min_",
]:
    if hasattr(scaler, name):
        print(f"{name}:", getattr(scaler, name))

print("inverse scaled [0, 0.5, 1]:")
print(scaler.inverse_transform([[0.0], [0.5], [1.0]]).reshape(-1))


# =========================
# 2. threshold.npy 조회
# =========================
threshold_path = MODEL_DIR / "threshold.npy"
threshold = np.load(threshold_path)

print("\n========== threshold.npy ==========")
print("path:", threshold_path)
print("threshold:", float(threshold))
print("shape:", threshold.shape)
print("dtype:", threshold.dtype)


# =========================
# 3. model.pth 조회
# =========================
model_path = MODEL_DIR / "model.pth"
state_dict = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True,
)

print("\n========== model.pth ==========")
print("path:", model_path)
print("num tensors:", len(state_dict))

total_params = 0

for name, tensor in state_dict.items():
    num_params = tensor.numel()
    total_params += num_params

    print(
        name,
        "shape=", tuple(tensor.shape),
        "dtype=", tensor.dtype,
        "params=", num_params,
        "min=", float(tensor.min()),
        "max=", float(tensor.max()),
    )

print("total params:", total_params)
