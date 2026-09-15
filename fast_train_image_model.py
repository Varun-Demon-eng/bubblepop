"""
=============================================================================
  TRAINER FOR FREQUENCY FORENSICS CNN
  Dataset: HuggingFace "itsLeen/deepfake_vs_real_image"
=============================================================================
"""

import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from torchvision import transforms

from deepfake_image_detector import (
    ImageForensicsConfig,
    DCT2DTransformer,
    FrequencyForensicsCNN,
    ImageDeepfakeDetector,
)

def prepare_tensor_from_pil(pil_img: Image.Image, dct_engine: DCT2DTransformer, config: ImageForensicsConfig):
    # Convert PIL -> NumPy grayscale uint8
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Resize to config target_size (512, 512)
    resized = cv2.resize(gray, (config.target_size[1], config.target_size[0]), interpolation=cv2.INTER_AREA)
    norm_spatial = resized.astype(np.float32) / 255.0
    
    # Apply 2D-DCT + log attenuation
    log_dct = dct_engine.transform(norm_spatial)
    
    # Tensor formatting
    tensor = torch.from_numpy(log_dct).unsqueeze(0)  # (1, H, W)
    normaliser = transforms.Normalize(mean=[config.norm_mean], std=[config.norm_std])
    tensor = normaliser(tensor).unsqueeze(0)           # (1, 1, H, W)
    return tensor

def main():
    print("=" * 65)
    print("  TRAINING FREQUENCY FORENSICS CNN ON 'itsLeen/deepfake_vs_real_image'")
    print("=" * 65)

    config = ImageForensicsConfig(
        target_size=(512, 512),
        cnn_base_channels=32,
        dropout_rate=0.4,
        device="auto",
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware accelerator: {device.type.upper()}")
    
    model = FrequencyForensicsCNN(config).to(device)
    dct_engine = DCT2DTransformer(config)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    # Load dataset streaming
    print("[INFO] Streaming dataset samples from HuggingFace ('itsLeen/deepfake_vs_real_image')...")
    ds = load_dataset("itsLeen/deepfake_vs_real_image", streaming=True)
    train_stream = ds["train"]
    test_stream  = ds["test"]
    
    MAX_TRAIN_SAMPLES = 100
    MAX_TEST_SAMPLES  = 30
    
    print(f"[INFO] Ingesting & transforming {MAX_TRAIN_SAMPLES} training images into 2D-DCT frequency matrices...")
    X_train, y_train = [], []
    
    real_count, fake_count = 0, 0
    half_limit = MAX_TRAIN_SAMPLES // 2
    
    for item in train_stream:
        hf_label = item["label"]  # 0: AI_Art, 1: Real_Art
        target_class = 0 if hf_label == 1 else 1  # 0: Authentic, 1: AI-Generated
        
        if target_class == 0 and real_count >= half_limit:
            continue
        if target_class == 1 and fake_count >= half_limit:
            continue
            
        try:
            tensor = prepare_tensor_from_pil(item["image"], dct_engine, config)
            X_train.append(tensor)
            y_train.append(target_class)
            
            if target_class == 0:
                real_count += 1
            else:
                fake_count += 1
                
            print(f"  [Progress] Loaded {len(X_train)}/{MAX_TRAIN_SAMPLES} samples (Real_Art: {real_count}, AI_Art: {fake_count})")
                
            if len(X_train) >= MAX_TRAIN_SAMPLES:
                break
        except Exception:
            continue
            
    print(f"[INFO] Training set prepared ({len(X_train)} frequency matrices).")
    
    # Collect test batch
    print(f"[INFO] Ingesting {MAX_TEST_SAMPLES} test images...")
    X_test, y_test = [], []
    for item in test_stream:
        hf_label = item["label"]
        target_class = 0 if hf_label == 1 else 1
        try:
            tensor = prepare_tensor_from_pil(item["image"], dct_engine, config)
            X_test.append(tensor)
            y_test.append(target_class)
            if len(X_test) >= MAX_TEST_SAMPLES:
                break
        except Exception:
            continue
    print(f"[INFO] Test set prepared ({len(X_test)} samples).")
    
    # Training Loop
    EPOCHS = 10
    BATCH_SIZE = 8
    
    print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
    num_train = len(X_train)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        indices = np.random.permutation(num_train)
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, num_train, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            batch_x = torch.cat([X_train[j] for j in batch_idx], dim=0).to(device)
            batch_y = torch.tensor([y_train[j] for j in batch_idx], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
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
        
    # Evaluation on Test set
    print("\n[INFO] Evaluating on Test Set...")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_x = torch.cat([X_test[j] for j in range(i, min(i+BATCH_SIZE, len(X_test)))], dim=0).to(device)
            batch_y = torch.tensor([y_test[j] for j in range(i, min(i+BATCH_SIZE, len(X_test)))], dtype=torch.long).to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == batch_y).sum().item()
            
    test_acc = (test_correct / len(X_test)) * 100.0
    print(f"[RESULT] Test Set Accuracy: {test_acc:.2f}%")
    
    # Save Model Weights
    save_path = Path("image_model.pth")
    torch.save(model.state_dict(), str(save_path))
    print(f"[INFO] Model weights saved to '{save_path.resolve()}'")
    
    # Re-evaluate user image
    user_img_path = r"C:\Users\raghu\OneDrive\Pictures\Saved Pictures\1000054739.jpg"
    print("\n" + "=" * 65)
    print(f"  RE-TESTING USER IMAGE WITH TRAINED WEIGHTS")
    print(f"  Path: {user_img_path}")
    print("=" * 65)
    
    eval_config = ImageForensicsConfig(weights_path=str(save_path), device="auto")
    detector = ImageDeepfakeDetector(eval_config)
    report = detector.predict_image(user_img_path)
    print(report)

if __name__ == "__main__":
    main()
