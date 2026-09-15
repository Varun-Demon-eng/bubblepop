"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         DEEPFAKE AUDIO DETECTION ENGINE  ·  Phase 1.5                      ║
║         Supervised Training Loop  ·  PyTorch + HuggingFace Datasets        ║
║                                                                              ║
║  Dataset   : Tanishq125/deepfake-audio-detection  (1,866 samples, balanced) ║
║  Schema    : { audio: AudioObject, label: int }  0=Real  1=Fake             ║
║  Optimizer : AdamW + CosineAnnealingLR                                      ║
║  Metrics   : Loss · Accuracy · AUC-ROC · EER                               ║
║  Output    : model.pth  (best val AUC checkpoint, ready for inference)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
─────
  # Standard training (downloads dataset automatically)
  python train.py

  # Custom hyperparameters
  python train.py --epochs 80 --batch_size 16 --lr 1e-4

  # Resume from a checkpoint
  python train.py --resume model.pth

  # Evaluate only (no training)
  python train.py --eval_only --resume model.pth
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import argparse
import warnings
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    sys.exit("[FATAL] PyTorch not found. Run: pip install torch")

try:
    import librosa
except ImportError:
    sys.exit("[FATAL] Librosa not found. Run: pip install librosa")

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
except ImportError:
    sys.exit("[FATAL] scikit-learn not found. Run: pip install scikit-learn")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[FATAL] HuggingFace datasets not found. Run: pip install datasets")

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODULE — the Phase 1 engine
# ─────────────────────────────────────────────────────────────────────────────
try:
    from deepfake_audio_detector import (
        AudioTCNClassifier,
        AudioConfig,
        TCNConfig,
        AUDIO_CFG,
        TCN_CFG,
    )
except ImportError:
    sys.exit(
        "[FATAL] deepfake_audio_detector.py not found.\n"
        "  Ensure train.py is in the same directory as deepfake_audio_detector.py"
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Trainer")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """
    All hyperparameters and training settings in one immutable place.
    Every argument can be overridden via argparse at the command line.
    """
    # Dataset
    hf_dataset_id:    str   = "Tanishq125/deepfake-audio-detection"
    val_split:        float = 0.15   # 15% validation (≈280 samples)
    test_split:       float = 0.05   # 5%  held-out test (≈93 samples)

    # Audio preprocessing (must match AUDIO_CFG)
    sample_rate:      int   = 16_000
    clip_duration_s:  float = 3.0    # Fixed clip length in seconds
    n_mels:           int   = 40
    n_fft:            int   = 1_024
    hop_length:       int   = 512
    top_db:           float = 80.0

    # Training loop
    num_epochs:       int   = 50
    batch_size:       int   = 32
    learning_rate:    float = 3e-4
    weight_decay:     float = 1e-4
    label_smoothing:  float = 0.1    # Prevents overconfident logits
    clip_grad_norm:   float = 1.0    # Gradient norm clipping threshold

    # Scheduler (CosineAnnealingLR)
    eta_min:          float = 1e-6   # Minimum LR at the end of cosine cycle

    # Early stopping
    patience:         int   = 12     # Stop if val AUC doesn't improve for N epochs
    min_delta:        float = 1e-4   # Minimum improvement to count as progress

    # Checkpointing
    checkpoint_path:  str   = "model.pth"

    # Reproducibility
    seed:             int   = 42

    # DataLoader
    num_workers:      int   = 0      # 0 = main process (Windows-safe default)
    pin_memory:       bool  = False  # Set True on CUDA for faster transfers


# =============================================================================
#  SECTION 1 — DATASET
# =============================================================================

class DeepfakeAudioDataset(Dataset):
    """
    PyTorch Dataset wrapping the Tanishq125/deepfake-audio-detection
    HuggingFace dataset.

    Each sample in the HuggingFace dataset has the schema:
        { "audio": {"array": np.ndarray, "sampling_rate": int},
          "label": int  (0=Real, 1=Fake) }

    This class:
    ① Decodes the raw audio array from the HF sample.
    ② Resamples to a fixed target sample rate (16 kHz).
    ③ Crops/pads to a fixed clip duration (3 s) for batch-ability.
    ④ Computes a 40-dim Log-Mel Spectrogram in dB.
    ⑤ Returns a (feature_tensor [40, T], label_int) pair.

    Random cropping during training acts as lightweight data augmentation
    by selecting different temporal windows per epoch.

    Args:
        hf_samples  (list) : List of raw HuggingFace dataset rows.
        cfg         (TrainingConfig): Full training config.
        augment     (bool) : If True, crop start is random (train mode).
                             If False, always take the first N seconds (val/test).
    """

    CLASS_NAMES = {0: "Real (Authentic)", 1: "Fake (Deepfake)"}

    def __init__(
        self,
        hf_samples: list,
        cfg: TrainingConfig,
        augment: bool = False,
    ) -> None:
        self.samples = hf_samples
        self.cfg     = cfg
        self.augment = augment

        # Pre-compute fixed clip length in samples
        self.clip_len = int(cfg.sample_rate * cfg.clip_duration_s)

        logger.debug(
            "Dataset ready | samples=%d | augment=%s | clip_len=%d",
            len(self.samples), augment, self.clip_len,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load, preprocess, and return one (feature_tensor, label) pair.

        Returns:
            features : torch.Tensor of shape [n_mels=40, T_frames]
            label    : int  (0 = Authentic, 1 = Deepfake)
        """
        row   = self.samples[idx]
        label = int(row["label"])

        # ── Step 1: Decode audio from raw bytes (no torchcodec needed) ────────
        # The HF dataset is loaded with Audio(decode=False), so audio_obj is:
        #   { "bytes": bytes_or_None, "path": str_or_None }
        # We decode manually using soundfile (already installed) or librosa.
        audio_obj = row["audio"]
        waveform, source_sr = self._decode_audio(audio_obj)

        # ── Step 2: Resample to target sample rate if needed ──────────────────
        if source_sr != self.cfg.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=source_sr, target_sr=self.cfg.sample_rate
            )

        # ── Step 3: Fixed-length cropping / padding ───────────────────────────
        waveform = self._crop_or_pad(waveform)

        # ── Step 4: Log-Mel Spectrogram extraction ────────────────────────────
        mel = librosa.feature.melspectrogram(
            y          = waveform,
            sr         = self.cfg.sample_rate,
            n_fft      = self.cfg.n_fft,
            hop_length = self.cfg.hop_length,
            n_mels     = self.cfg.n_mels,
            power      = 2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=self.cfg.top_db)
        log_mel = log_mel.astype(np.float32)

        # ── Step 5: Normalise per-sample (zero-mean, unit-variance) ──────────
        # Stabilises gradients; each clip has a different absolute dB scale.
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        features = torch.from_numpy(log_mel)   # [40, T]
        return features, label

    def _crop_or_pad(self, waveform: np.ndarray) -> np.ndarray:
        """
        Crop to self.clip_len samples (random start if augment=True,
        else from the beginning), or zero-pad if shorter.

        Args:
            waveform: 1-D float32 numpy array.

        Returns:
            waveform of exactly shape [clip_len].
        """
        n = len(waveform)
        if n >= self.clip_len:
            # Longer than target: crop a window
            if self.augment:
                # Random start (data augmentation via temporal jitter)
                start = random.randint(0, n - self.clip_len)
            else:
                # Deterministic: always centre-crop for reproducible eval
                start = (n - self.clip_len) // 2
            return waveform[start : start + self.clip_len]
        else:
            # Shorter than target: right-pad with zeros
            pad_width = self.clip_len - n
            return np.pad(waveform, (0, pad_width), mode="constant")

    @staticmethod
    def _decode_audio(audio_obj: dict) -> Tuple[np.ndarray, int]:
        """
        Decode audio from a HuggingFace Audio(decode=False) object.

        When `decode=False`, HuggingFace returns a dict with either:
            { "bytes": <raw audio file bytes>, "path": None }   -- embedded
            { "bytes": None, "path": "<local path>" }           -- file-backed

        We use soundfile (fastest, no torchcodec required) to decode
        the bytes into a float32 numpy array, then fall back to librosa.

        Args:
            audio_obj (dict): The raw audio dict from HF dataset row.

        Returns:
            (waveform, sample_rate): float32 ndarray and original SR.
        """
        import io
        import soundfile as sf

        raw_bytes = audio_obj.get("bytes")
        file_path = audio_obj.get("path")

        if raw_bytes is not None:
            # Embedded bytes path (most common for this dataset)
            try:
                buf = io.BytesIO(raw_bytes)
                waveform, sr = sf.read(buf, dtype="float32", always_2d=False)
                # If stereo, downmix to mono by averaging channels
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
                return waveform, sr
            except Exception as exc:
                # Fallback to librosa (handles more edge-case formats)
                buf = io.BytesIO(raw_bytes)
                waveform, sr = librosa.load(buf, sr=None, mono=True)
                return waveform.astype(np.float32), sr

        elif file_path is not None:
            # File-backed path
            waveform, sr = librosa.load(file_path, sr=None, mono=True)
            return waveform.astype(np.float32), sr

        else:
            raise ValueError(
                "HuggingFace audio object has neither 'bytes' nor 'path'. "
                "Inspect the dataset schema."
            )


# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom DataLoader collator that stacks feature tensors and labels.

    All tensors within a batch will have the same time dimension because
    DeepfakeAudioDataset._crop_or_pad() enforces a fixed clip length,
    so standard stacking is safe here.

    However, minor rounding differences in Librosa's STFT can occasionally
    produce ±1 frame variation. This collator handles that edge case by
    padding along the time axis (dim=1) to the maximum T in the batch.

    Args:
        batch: List of (feature_tensor [40, T_i], label) tuples.

    Returns:
        features_batch : torch.Tensor [B, 40, T_max]
        labels_batch   : torch.LongTensor [B]
    """
    features, labels = zip(*batch)

    # Find the maximum time dimension in this batch
    max_T = max(f.shape[1] for f in features)

    # Pad all tensors to max_T along dim=1
    padded = []
    for f in features:
        t = f.shape[1]
        if t < max_T:
            pad = torch.zeros(f.shape[0], max_T - t, dtype=f.dtype)
            f   = torch.cat([f, pad], dim=1)
        padded.append(f)

    features_batch = torch.stack(padded, dim=0)          # [B, 40, T_max]
    labels_batch   = torch.tensor(labels, dtype=torch.long)  # [B]
    return features_batch, labels_batch


# =============================================================================
#  SECTION 2 — METRICS
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) — the threshold at which the False
    Acceptance Rate (FAR) equals the False Rejection Rate (FRR).

    EER is the gold-standard metric in speaker verification and
    deepfake detection benchmarks (ASVspoof). Lower EER = better.

    Args:
        y_true   : Binary ground-truth labels [N] (0=Real, 1=Fake).
        y_scores : Predicted probability of class 1 (Fake) [N].

    Returns:
        eer (float): EER as a fraction in [0.0, 1.0].
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr  # False Negative Rate = 1 - True Positive Rate

    # EER is where FPR ≈ FNR; find the threshold index closest to this point
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return eer


@dataclass
class EpochMetrics:
    """
    Typed container for all metrics produced in a single epoch.
    Makes passing metrics between functions clean and IDE-friendly.
    """
    loss:     float
    accuracy: float   # as percentage (0-100)
    auc_roc:  float   # 0.0 to 1.0
    eer:      float   # 0.0 to 1.0 (lower is better)

    def __str__(self) -> str:
        return (
            f"Loss={self.loss:.4f}  "
            f"Acc={self.accuracy:.2f}%  "
            f"AUC={self.auc_roc:.4f}  "
            f"EER={self.eer * 100:.2f}%"
        )


# =============================================================================
#  SECTION 3 — EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Monitor a validation metric and signal when training should stop
    due to lack of improvement.

    Tracks the *maximum* of the monitored metric (AUC-ROC in our case).
    If the metric does not improve by at least `min_delta` for `patience`
    consecutive epochs, `should_stop` is set to True.

    Args:
        patience  (int)  : Number of epochs to wait without improvement.
        min_delta (float): Minimum change to count as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_score  = float("-inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Call after each validation epoch.

        Args:
            score (float): Current validation metric value (higher = better).

        Returns:
            improved (bool): True if this is a new best score.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return True   # Improved — caller should save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping triggered after %d epochs without improvement.",
                    self.patience,
                )
            return False


# =============================================================================
#  SECTION 4 — TRAINER
# =============================================================================

class Trainer:
    """
    Full supervised training orchestrator for AudioTCNClassifier.

    Responsibilities
    ────────────────
    ① Build train / val / test DataLoaders from the HuggingFace dataset.
    ② Configure AdamW optimizer + CosineAnnealingLR scheduler.
    ③ Execute the training loop with gradient clipping.
    ④ Evaluate after every epoch: Loss, Accuracy, AUC-ROC, EER.
    ⑤ Save the best checkpoint (by val AUC) to disk.
    ⑥ Apply early stopping to prevent wasteful over-training.
    ⑦ Final evaluation on the held-out test split.

    Args:
        cfg (TrainingConfig): Complete training configuration.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._seed_everything(cfg.seed)

        # ── Hardware ─────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Hardware: %s%s",
            str(self.device).upper(),
            f"  ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "",
        )

        # ── Dataset loading ───────────────────────────────────────────────────
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = AudioTCNClassifier(
            input_channels = TCN_CFG.input_channels,
            num_channels   = list(TCN_CFG.num_channels),
            kernel_size    = TCN_CFG.kernel_size,
            dropout        = TCN_CFG.dropout,
            num_classes    = TCN_CFG.num_classes,
        ).to(self.device)

        # ── Loss ──────────────────────────────────────────────────────────────
        # label_smoothing: prevents overconfident softmax outputs and
        # regularises the model naturally on a small dataset like this one.
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing = cfg.label_smoothing,
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        # AdamW decouples the weight decay from the gradient update, giving
        # better regularisation than vanilla Adam on small datasets.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
        )

        # ── LR Scheduler ──────────────────────────────────────────────────────
        # CosineAnnealingLR decays LR smoothly from lr to eta_min over
        # T_max epochs. No abrupt drops = stable loss curves.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg.num_epochs,
            eta_min = cfg.eta_min,
        )

        # ── Early stopping ─────────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience  = cfg.patience,
            min_delta = cfg.min_delta,
        )

        # ── Training state ─────────────────────────────────────────────────────
        self.start_epoch     = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "val_auc":    [], "val_eer":   [],
            "lr":         [],
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Pin all random sources for full reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("Seed pinned to %d", seed)

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Download (or use cache) the HuggingFace dataset, split into
        train / val / test, and wrap in DataLoaders.

        The dataset has only a 'train' split, so we manually divide it:
            80% train  |  15% validation  |  5% test (held-out)

        Returns:
            (train_loader, val_loader, test_loader)
        """
        logger.info("Loading dataset: %s ...", self.cfg.hf_dataset_id)
        logger.info("  (This may take a few minutes on first run -- downloading ~50 MB)")

        from datasets import Audio as HFAudio

        raw = load_dataset(self.cfg.hf_dataset_id, split="train")

        # CRITICAL: disable HuggingFace's built-in audio decoding.
        # The default decoder requires 'torchcodec' which may not be installed.
        # With decode=False, each row's 'audio' field becomes:
        #   { "bytes": <raw bytes>, "path": str_or_None }
        # Our DeepfakeAudioDataset._decode_audio() handles this manually
        # using soundfile (already installed), with no external codec needed.
        raw = raw.cast_column("audio", HFAudio(decode=False))

        logger.info("  OK Dataset loaded  |  Total samples: %d", len(raw))

        # Log class distribution — use raw['label'] for fast column access
        # (avoids iterating over rows, which would trigger audio decoding)
        label_col  = raw["label"]
        real_count = sum(1 for l in label_col if l == 0)
        fake_count = sum(1 for l in label_col if l == 1)
        logger.info(
            "  Class distribution: Real=%d (%.1f%%)  Fake=%d (%.1f%%)",
            real_count, 100 * real_count / len(raw),
            fake_count, 100 * fake_count / len(raw),
        )

        # Shuffle before splitting (deterministic via seed)
        raw = raw.shuffle(seed=self.cfg.seed)

        # Manual train / val / test split
        n        = len(raw)
        n_test   = max(1, int(n * self.cfg.test_split))
        n_val    = max(1, int(n * self.cfg.val_split))
        n_train  = n - n_val - n_test

        all_rows = list(raw)   # Convert to plain list for slicing
        train_rows = all_rows[:n_train]
        val_rows   = all_rows[n_train : n_train + n_val]
        test_rows  = all_rows[n_train + n_val:]

        logger.info(
            "  Split: Train=%d  Val=%d  Test=%d",
            len(train_rows), len(val_rows), len(test_rows),
        )

        # Wrap in Dataset objects
        train_ds = DeepfakeAudioDataset(train_rows, self.cfg, augment=True)
        val_ds   = DeepfakeAudioDataset(val_rows,   self.cfg, augment=False)
        test_ds  = DeepfakeAudioDataset(test_rows,  self.cfg, augment=False)

        loader_kwargs = dict(
            batch_size  = self.cfg.batch_size,
            collate_fn  = collate_fn,
            num_workers = self.cfg.num_workers,
            pin_memory  = self.cfg.pin_memory and self.device.type == "cuda",
        )

        train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
        val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
        test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

        return train_loader, val_loader, test_loader

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        """
        Resume training from a saved checkpoint.

        The checkpoint stores both model weights and optimizer state,
        allowing training to continue seamlessly from the saved epoch.

        Args:
            path (str): Path to the .pth checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state_dict" in ckpt:
            # Full training checkpoint (has optimizer state, epoch, history)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.history     = ckpt.get("history", self.history)
            self.early_stopping.best_score = ckpt.get("best_val_auc", float("-inf"))
            logger.info(
                "Resumed from checkpoint '%s'  |  Epoch %d  |  Best AUC: %.4f",
                path, self.start_epoch, self.early_stopping.best_score,
            )
        else:
            # Inference-only checkpoint (just model weights from AudioDeepfakeDetector)
            self.model.load_state_dict(ckpt)
            logger.info("Loaded inference checkpoint from '%s'  |  Starting from epoch 0", path)

    def save_checkpoint(self, epoch: int, val_auc: float, inference_only: bool = False) -> None:
        """
        Save the current model to disk.

        Two modes:
            inference_only=False → Full checkpoint (model + optimizer state)
                                   Used for resuming training.
            inference_only=True  → Weights-only state dict.
                                   This is what AudioDeepfakeDetector.predict_audio()
                                   expects when you pass weights_path="model.pth".

        The best checkpoint (model.pth) is always saved as weights-only so
        the inference engine can load it without modification.
        """
        if inference_only:
            # Flat state_dict — directly loadable by AudioDeepfakeDetector
            torch.save(self.model.state_dict(), self.cfg.checkpoint_path)
        else:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_auc":         val_auc,
                "history":              self.history,
                "tcn_cfg":              TCN_CFG,
                "audio_cfg":            AUDIO_CFG,
            }, self.cfg.checkpoint_path.replace(".pth", "_full.pth"))

        logger.info("Checkpoint saved to '%s'  |  Val AUC: %.4f", self.cfg.checkpoint_path, val_auc)

    # ── Epoch-level routines ───────────────────────────────────────────────────

    def _train_epoch(self) -> EpochMetrics:
        """
        One full pass over the training set.

        Returns:
            EpochMetrics with training loss and accuracy.
            (AUC and EER are only computed on val/test to save compute.)
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_probs = [], []

        for batch_features, batch_labels in self.train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels   = batch_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_features)          # [B, 2]
            loss   = self.criterion(logits, batch_labels)

            # Backward pass + gradient clipping
            loss.backward()
            if self.cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * batch_labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == batch_labels).sum().item()
            total      += batch_labels.size(0)

            probs = F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        # Compute AUC on training set as well (useful to detect overfitting)
        all_labels_np = np.array(all_labels)
        all_probs_np  = np.array(all_probs)
        try:
            auc = float(roc_auc_score(all_labels_np, all_probs_np))
            eer = compute_eer(all_labels_np, all_probs_np)
        except Exception:
            auc, eer = 0.5, 0.5   # fallback for degenerate batches

        return EpochMetrics(loss=avg_loss, accuracy=accuracy, auc_roc=auc, eer=eer)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> EpochMetrics:
        """
        One full pass over a validation or test DataLoader.
        No gradient computation — faster and memory-efficient.

        Returns:
            EpochMetrics with all four metrics: loss, accuracy, AUC-ROC, EER.
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_probs = [], []

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(self.device)
            batch_labels   = batch_labels.to(self.device)

            logits = self.model(batch_features)
            loss   = self.criterion(logits, batch_labels)

            total_loss += loss.item() * batch_labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == batch_labels).sum().item()
            total      += batch_labels.size(0)

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        all_labels_np = np.array(all_labels)
        all_probs_np  = np.array(all_probs)
        try:
            auc = float(roc_auc_score(all_labels_np, all_probs_np))
            eer = compute_eer(all_labels_np, all_probs_np)
        except Exception:
            auc, eer = 0.5, 0.5

        return EpochMetrics(loss=avg_loss, accuracy=accuracy, auc_roc=auc, eer=eer)

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Execute the full training loop.

        Flow per epoch:
            1. train_epoch()  → update weights
            2. eval_epoch(val_loader) → compute val metrics
            3. scheduler.step()  → decay LR
            4. early_stopping.step(val_auc):
               - If improved → save_checkpoint() (best model)
               - If patience exceeded → stop training
            5. Print epoch summary
        """
        logger.info("=" * 70)
        logger.info("  Training started  |  Epochs=%d  |  Batch=%d  |  LR=%.1e",
                    self.cfg.num_epochs, self.cfg.batch_size, self.cfg.learning_rate)
        logger.info("=" * 70)

        BOLD, CYAN, GREEN, YELLOW, RESET = (
            "\033[1m", "\033[96m", "\033[92m", "\033[93m", "\033[0m"
        )

        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            train_metrics = self._train_epoch()

            # ── Validate ──────────────────────────────────────────────────────
            val_metrics = self._eval_epoch(self.val_loader)

            # ── LR step ───────────────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # ── Record history ────────────────────────────────────────────────
            self.history["train_loss"].append(train_metrics.loss)
            self.history["train_acc"].append(train_metrics.accuracy)
            self.history["val_loss"].append(val_metrics.loss)
            self.history["val_acc"].append(val_metrics.accuracy)
            self.history["val_auc"].append(val_metrics.auc_roc)
            self.history["val_eer"].append(val_metrics.eer)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0

            # ── Early stopping check ──────────────────────────────────────────
            improved = self.early_stopping.step(val_metrics.auc_roc)
            star     = f" {GREEN}*BEST*{RESET}" if improved else ""

            # Save best model weights (inference-compatible format)
            if improved:
                self.save_checkpoint(epoch, val_metrics.auc_roc, inference_only=True)

            # ── Console summary ───────────────────────────────────────────────
            print(
                f"{BOLD}Epoch [{epoch+1:3d}/{self.cfg.num_epochs}]{RESET}"
                f"  {elapsed:.1f}s"
                f"  LR={current_lr:.2e}"
                f"  |  Train: Loss={train_metrics.loss:.4f}  Acc={train_metrics.accuracy:.1f}%"
                f"  |  Val: Loss={val_metrics.loss:.4f}  Acc={val_metrics.accuracy:.1f}%"
                f"  {CYAN}AUC={val_metrics.auc_roc:.4f}{RESET}"
                f"  EER={val_metrics.eer*100:.1f}%"
                f"{star}"
            )

            if self.early_stopping.should_stop:
                print(f"\n{YELLOW}  Early stopping at epoch {epoch+1}.{RESET}")
                break

        # ── Final evaluation on held-out test set ─────────────────────────────
        self._final_test_report()

    def evaluate_only(self) -> None:
        """
        Skip training and run directly on the validation + test splits.
        Used with --eval_only flag to benchmark a loaded checkpoint.
        """
        logger.info("Evaluation-only mode  |  Running val + test sets ...")
        val_m  = self._eval_epoch(self.val_loader)
        test_m = self._eval_epoch(self.test_loader)

        BOLD, CYAN, GREEN, RESET = "\033[1m", "\033[96m", "\033[92m", "\033[0m"
        print(f"\n{BOLD}  Validation  {RESET}: {val_m}")
        print(f"{BOLD}  Test        {RESET}: {test_m}\n")

    def _final_test_report(self) -> None:
        """
        Evaluate the best saved checkpoint on the held-out test split
        and print a structured final report.
        """
        BOLD, CYAN, GREEN, RESET = "\033[1m", "\033[96m", "\033[92m", "\033[0m"

        # Reload the best checkpoint for final test evaluation
        best_path = self.cfg.checkpoint_path
        if Path(best_path).exists():
            state = torch.load(best_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info("Loaded best checkpoint from '%s' for final test eval.", best_path)

        test_m = self._eval_epoch(self.test_loader)

        print(f"\n{BOLD}{CYAN}+{'='*60}+{RESET}")
        print(f"{BOLD}{CYAN}|         FINAL TEST SET RESULTS (held-out)              |{RESET}")
        print(f"{BOLD}{CYAN}+{'='*60}+{RESET}")
        print(f"  Accuracy : {test_m.accuracy:.2f}%")
        print(f"  AUC-ROC  : {test_m.auc_roc:.4f}")
        print(f"  EER      : {test_m.eer * 100:.2f}%")
        print(f"  Loss     : {test_m.loss:.4f}")
        print(f"{BOLD}{CYAN}+{'='*60}+{RESET}")
        print(f"\n  Best model saved to: '{self.cfg.checkpoint_path}'")
        print(f"  Use it with:")
        print(f"  {CYAN}python test_audio.py --weights {self.cfg.checkpoint_path} your_audio.wav{RESET}\n")


# =============================================================================
#  SECTION 5 — CLI ENTRY POINT
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "train",
        description = "Phase 1.5 — Train the Deepfake Audio TCN on Tanishq125/deepfake-audio-detection",
        formatter_class = argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=50,      help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size",  type=int,   default=32,      help="Batch size (default: 32)")
    parser.add_argument("--lr",          type=float, default=3e-4,    help="Initial learning rate (default: 3e-4)")
    parser.add_argument("--weight_decay",type=float, default=1e-4,    help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--patience",    type=int,   default=12,      help="Early stopping patience in epochs (default: 12)")
    parser.add_argument("--val_split",   type=float, default=0.15,    help="Validation fraction (default: 0.15)")
    parser.add_argument("--checkpoint",  type=str,   default="model.pth", help="Output checkpoint path (default: model.pth)")
    parser.add_argument("--resume",      type=str,   default=None,    help="Resume training from this checkpoint path")
    parser.add_argument("--eval_only",   action="store_true",         help="Skip training; evaluate a loaded checkpoint only")
    parser.add_argument("--seed",        type=int,   default=42,      help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int,   default=0,       help="DataLoader workers (default: 0, Windows-safe)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Build config from defaults + CLI overrides
    cfg = TrainingConfig(
        num_epochs      = args.epochs,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        weight_decay    = args.weight_decay,
        patience        = args.patience,
        val_split       = args.val_split,
        checkpoint_path = args.checkpoint,
        seed            = args.seed,
        num_workers     = args.num_workers,
    )

    BOLD, CYAN, RESET = "\033[1m", "\033[96m", "\033[0m"
    print(f"\n{BOLD}{CYAN}  ============================================================{RESET}")
    print(f"{BOLD}{CYAN}  Deepfake Audio Detection Engine  --  Phase 1.5 Training{RESET}")
    print(f"{BOLD}{CYAN}  Dataset : Tanishq125/deepfake-audio-detection (1,866 clips){RESET}")
    print(f"{BOLD}{CYAN}  ============================================================{RESET}\n")

    trainer = Trainer(cfg)

    if args.resume:
        if not Path(args.resume).exists():
            logger.error("Checkpoint not found: '%s'", args.resume)
            sys.exit(1)
        trainer.load_checkpoint(args.resume)

    if args.eval_only:
        if not args.resume:
            logger.error("--eval_only requires --resume <checkpoint.pth>")
            sys.exit(1)
        trainer.evaluate_only()
    else:
        trainer.train()

    sys.exit(0)
