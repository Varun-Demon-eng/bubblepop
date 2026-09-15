"""
=============================================================================
  TRAINING SCRIPT FOR 3D-PHYSNET VIDEO rPPG DEEPFAKE DETECTOR
  Dataset: HuggingFace "UniDataPro/deepfake-videos-dataset"
=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

from deepfake_video_detector import (
    VideoForensicsConfig,
    FacialROIPipeline,
    PhysNet3DCNN,
    VideoDeepfakeRPPGEngine,
)

def main():
    print("=" * 65)
    print("  TRAINING 3D-PHYSNET ON 'UniDataPro/deepfake-videos-dataset'")
    print("=" * 65)

    config = VideoForensicsConfig(
        sample_fps=30,
        sequence_length=64,
        roi_size=(128, 128),
        base_channels=32,
        device="auto",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware accelerator: {device.type.upper()}")

    model = PhysNet3DCNN(config).to(device)
    roi_pipe = FacialROIPipeline(config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    repo_id = "UniDataPro/deepfake-videos-dataset"
    print(f"[INFO] Listing video files from HuggingFace repo '{repo_id}'...")
    
    api = HfApi()
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    
    deepfake_files = [f for f in repo_files if f.startswith("deepfake/") and f.lower().endswith((".mp4", ".mov", ".avi", ".webm"))]
    real_files     = [f for f in repo_files if f.startswith("video/") and f.lower().endswith((".mp4", ".mov", ".avi", ".webm"))]
    
    print(f"[INFO] Discovered {len(deepfake_files)} deepfake video(s) and {len(real_files)} real video(s).")
    
    # Balance train set
    SAMPLES_PER_CLASS = min(10, len(deepfake_files), len(real_files))
    target_files = []
    
    for f in real_files[:SAMPLES_PER_CLASS]:
        target_files.append((f, 0))  # 0: Authentic
    for f in deepfake_files[:SAMPLES_PER_CLASS]:
        target_files.append((f, 1))  # 1: AI-Generated/Deepfake

    print(f"[INFO] Downloading and extracting 5D chromatic tensors for {len(target_files)} videos...")
    X_train, y_train = [], []

    for idx, (filename, label) in enumerate(target_files, 1):
        try:
            local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
            chroma_np = roi_pipe.process_video(local_path)  # (2, T, H, W)
            tensor = torch.from_numpy(chroma_np).unsqueeze(0)  # (1, 2, T, H, W)
            X_train.append(tensor)
            y_train.append(label)
            cls_name = "Authentic" if label == 0 else "Deepfake"
            print(f"  [{idx}/{len(target_files)}] Processed '{filename}' -> Label: {cls_name}")
        except Exception as exc:
            print(f"  [{idx}/{len(target_files)}] Failed '{filename}': {exc}")
            continue

    print(f"\n[INFO] Dataset prepared ({len(X_train)} volumetric video tensors).")
    if len(X_train) == 0:
        print("[ERROR] No videos were processed.")
        return

    # Training Loop
    EPOCHS = 8
    BATCH_SIZE = 2
    num_samples = len(X_train)

    print(f"\n[INFO] Starting 3D-PhysNet training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, num_samples, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            batch_x = torch.cat([X_train[j] for j in batch_idx], dim=0).to(device)
            batch_y = torch.tensor([y_train[j] for j in batch_idx], dtype=torch.long).to(device)

            optimizer.zero_grad()
            bvp_wave, outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_idx)

        train_acc = (correct / total) * 100.0
        avg_loss = epoch_loss / total
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:2d}/{EPOCHS} [{elapsed:4.1f}s] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}%")

    save_path = Path("video_model.pth")
    torch.save(model.state_dict(), str(save_path))
    print(f"\n[INFO] 3D-PhysNet model weights saved to '{save_path.resolve()}'")

if __name__ == "__main__":
    main()
