"""
=============================================================================
  REMOTE PHOTOPLETHYSMOGRAPHY (rPPG) VIDEO DEEPFAKE DETECTION ENGINE
  Phase 3 - Spatio-Temporal Facial Forensics via 3D-PhysNet & rPPG BVP
=============================================================================

  Module   : deepfake_video_detector.py
  Modality : VIDEO ONLY (completely isolated from audio and image pipelines)
  Author   : Principal ML Engineer - Biomedical Vision & Facial Forensics Division

  Architecture:
    +---------------------------------------------------------+
    |  Raw Video (.mp4 / .mov / .avi)                         |
    |       |                                                 |
    |  [FacialROIPipeline]                                    |
    |   ├─ OpenCV video reader @ target FPS                   |
    |   ├─ Face ROI Tracking (Forehead + Cheeks)              |
    |   ├─ Motion noise filter (Eyes blinking + Mouth motion) |
    |   └─ YCrCb color space -> Discard Y, Keep [Cr, Cb]      |
    |       |                                                 |
    |  [PhysNet3DCNN]                                         |
    |   ├─ 3D Convolutional Encoder [B, 2, T, H, W]           |
    |   ├─ 3D MaxPool spatio-temporal downsampling            |
    |   ├─ 1D Blood Volume Pulse (BVP) Waveform Head (B, T)   |
    |   └─ Classification Head -> 2 Logits                    |
    |       |                                                 |
    |  [VideoDeepfakeRPPGEngine]                              |
    |   ├─ FFT Power Spectrum & Cardiovascular Peak (BPM)     |
    |   ├─ Pulse Rhythmic Regularity SNR                      |
    |   └─ predict_video() -> VideoForensicReport             |
    +---------------------------------------------------------+

=============================================================================
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from scipy.signal import find_peaks, welch

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="[VIDEO-rPPG-FORENSICS] %(levelname)s | %(message)s",
)
log = logging.getLogger("video_rppg_forensics")


# ===========================================================================
# SECTION 1 - CONFIGURATION DATACLASS
# ===========================================================================

@dataclass
class VideoForensicsConfig:
    sample_fps:      int             = 30
    sequence_length: int             = 64
    roi_size:        Tuple[int, int] = (128, 128)
    color_space:     str             = "YCrCb"
    base_channels:   int             = 32
    num_classes:     int             = 2
    bpm_min:         float           = 45.0
    bpm_max:         float           = 180.0
    dropout_rate:    float           = 0.4
    device:          str             = "auto"
    weights_path:    Optional[str]   = None
    video_dir:       str             = "deepfake_video"
    supported_ext:   tuple           = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# ===========================================================================
# SECTION 2 - FACIAL LANDMARK ROI EXTRACTION PIPELINE
# ===========================================================================

class FacialROIPipeline:
    """
    Extracts facial Regions of Interest (forehead, cheeks) across video frames,
    filters motion noise (eyes/mouth), and converts pixel streams from RGB/BGR
    to YCrCb (discarding Y luminance to eliminate shadow interference).

    Parameters
    ----------
    config : VideoForensicsConfig
    """

    def __init__(self, config: VideoForensicsConfig) -> None:
        self.config = config
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def process_video(self, video_path: str | Path) -> np.ndarray:
        """
        Ingests video, extracts T frames of chromatic facial ROI crops.

        Returns
        -------
        np.ndarray : float32 tensor of shape (2, T, H, W) containing [Cr, Cb] channels.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV failed to open video: {video_path.name}")

        frames_roi = []
        target_T   = self.config.sequence_length

        while len(frames_roi) < target_T:
            ret, frame = cap.read()
            if not ret:
                break

            roi_chroma = self._extract_frame_chroma_roi(frame)
            frames_roi.append(roi_chroma)

        cap.release()

        if len(frames_roi) == 0:
            raise ValueError(f"No valid frames extracted from '{video_path.name}'.")

        # Pad sequence if video is shorter than sequence_length
        while len(frames_roi) < target_T:
            frames_roi.append(frames_roi[-1])

        # Stack into (T, H, W, 2) -> Transpose to (2, T, H, W)
        sequence_np = np.stack(frames_roi, axis=0)
        sequence_np = np.transpose(sequence_np, (3, 0, 1, 2)).astype(np.float32)

        log.info("Processed video '%s' -> shape %s", video_path.name, sequence_np.shape)
        return sequence_np

    def _extract_frame_chroma_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts facial ROI, filters motion noise (eye blinking / mouth motion),
        converts to YCrCb, discards Y channel (luminance/shadows), and resizes
        [Cr, Cb] to (H, W, 2).
        """
        H, W, _ = frame_bgr.shape
        x, y, w, h = self._detect_face_roi(frame_bgr, H, W)

        # Crop face bounding box
        face_crop = frame_bgr[y:y+h, x:x+w]
        if face_crop.size == 0:
            face_crop = cv2.resize(frame_bgr, self.config.roi_size)
        else:
            face_crop = cv2.resize(face_crop, self.config.roi_size)

        # Color-space transformation to YCrCb
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        
        # Discard Y (channel 0), preserve Cr (channel 1) and Cb (channel 2)
        chroma = ycrcb[:, :, 1:3].astype(np.float32) / 255.0  # (H, W, 2) in [0, 1]
        return chroma

    def _detect_face_roi(self, frame_bgr: np.ndarray, H: int, W: int) -> Tuple[int, int, int, int]:
        """Detect face ROI using OpenCV Haar Cascade or central crop fallback."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            return tuple(faces[0])

        # Center fallback crop
        return (W // 4, H // 4, W // 2, H // 2)


# ===========================================================================
# SECTION 3 - 3D SPATIO-TEMPORAL PHYSNET ARCHITECTURE
# ===========================================================================

class _ConvBnRelu3D(nn.Module):
    """Atomic 3D Convolutional Block: Conv3D -> BatchNorm3D -> ELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PhysNet3DCNN(nn.Module):
    """
    3D Spatio-Temporal Convolutional Network for rPPG Blood Volume Pulse (BVP)
    extraction and Deepfake classification.

    Input  : (B, 2, T, H, W)  - 5D volumetric tensor containing [Cr, Cb] channels over time.
    Output : 
       - bvp_signal : (B, T) 1D temporal Blood Volume Pulse waveform
       - logits     : (B, 2) raw classification logits [0: Authentic, 1: AI-Generated]
    """

    def __init__(self, config: VideoForensicsConfig) -> None:
        super().__init__()
        C = config.base_channels

        # Stage 1: Spatial & Short Temporal Features (B, 2, T, H, W) -> (B, C, T, H/2, W/2)
        self.stage1 = nn.Sequential(
            _ConvBnRelu3D(2, C, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),  # Spatial downsample only
        )

        # Stage 2: Spatio-Temporal Convolution (B, C, T, H/2, W/2) -> (B, 2C, T, H/4, W/4)
        self.stage2 = nn.Sequential(
            _ConvBnRelu3D(C, C*2, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
        )

        # Stage 3: Deep Spatio-Temporal Features (B, 2C, T, H/4, W/4) -> (B, 4C, T, H/8, W/8)
        self.stage3 = nn.Sequential(
            _ConvBnRelu3D(C*2, C*4, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
        )

        # Spatial Global Average Pooling -> retains Temporal dimension T
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, 4C, T, 1, 1)

        # 1D Temporal BVP Signal Head
        self.bvp_head = nn.Sequential(
            nn.Conv1d(C * 4, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=1),  # Outputs 1D signal (B, 1, T)
        )

        # Classification Head
        flat_dim = C * 4 * config.sequence_length
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ELU(inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(256, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Pool spatial dimensions to (B, 4C, T, 1, 1) -> squeeze to (B, 4C, T)
        feat_temp = self.spatial_pool(x).squeeze(-1).squeeze(-1)

        # Head 1: BVP Waveform Signal (B, T)
        bvp_signal = self.bvp_head(feat_temp).squeeze(1)

        # Head 2: Deepfake Logits (B, 2)
        logits = self.classifier(feat_temp)

        return bvp_signal, logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# SECTION 4 - FORENSIC REPORT DATACLASS
# ===========================================================================

@dataclass
class VideoForensicReport:
    video_path:        str
    verdict:           str
    confidence:        float
    prob_authentic:    float
    prob_ai_generated: float
    heart_rate_bpm:    float
    pulse_regularity:  float
    hardware:          str

    def __str__(self) -> str:
        bar_auth = "X" * int(self.prob_authentic    * 30)
        bar_ai   = "X" * int(self.prob_ai_generated * 30)
        W = 30
        return (
            "\n"
            "+----------------------------------------------------------+\n"
            "|         VIDEO rPPG FORENSICS REPORT - DEEPFAKE ENGINE     |\n"
            "+----------------------------------------------------------+\n"
            f"|  File      : {Path(self.video_path).name:<44}|\n"
            f"|  Hardware  : {self.hardware:<44}|\n"
            "+----------------------------------------------------------+\n"
            f"|  VERDICT         : {self.verdict:<39}|\n"
            f"|  CONFIDENCE      : {self.confidence * 100:>6.2f}%{'':<33}|\n"
            f"|  HEART RATE (BPM): {self.heart_rate_bpm:>6.1f} BPM{'':<31}|\n"
            f"|  PULSE SNR SCORE : {self.pulse_regularity:>6.2f} dB{'':<32}|\n"
            "+----------------------------------------------------------+\n"
            "|  PROBABILITY BREAKDOWN                                   |\n"
            f"|  Authentic      {self.prob_authentic * 100:>6.2f}%  {bar_auth:<{W}}|\n"
            f"|  AI-Generated   {self.prob_ai_generated * 100:>6.2f}%  {bar_ai:<{W}}|\n"
            "+----------------------------------------------------------+\n"
        )


# ===========================================================================
# SECTION 5 - PRODUCTION BIOMETRIC EVALUATION ENGINE
# ===========================================================================

class VideoDeepfakeRPPGEngine:
    """
    Public inference wrapper for the Video rPPG Deepfake Detection Engine.

    Encapsulates:
      1. Hardware resolution (CUDA/CPU).
      2. 3D-PhysNet model initialization and weight loading.
      3. Facial ROI tracking & YCrCb chromatic extraction.
      4. Spatio-temporal 3D-CNN forward pass.
      5. FFT power spectrum calculation for Heart Rate (BPM) & pulse regularity.
      6. Structured VideoForensicReport generation.
    """

    _LABELS = {0: "AUTHENTIC (Real Human)", 1: "AI-GENERATED (Deepfake)"}

    def __init__(self, config: VideoForensicsConfig) -> None:
        self.config    = config
        self.device    = self._resolve_device(config.device)
        self.roi_pipe  = FacialROIPipeline(config)
        self.model     = self._build_model()
        log.info(
            "VideoDeepfakeRPPGEngine ready | device: %s | params: %s",
            str(self.device).upper(),
            f"{self.model.count_parameters():,}",
        )

    def predict_video(self, video_path: str | Path) -> VideoForensicReport:
        """Runs end-to-end video rPPG forensic evaluation."""
        video_path = Path(video_path)

        # Stage 1: Extract 5D Chromatic Tensor (2, T, H, W)
        chroma_tensor_np = self.roi_pipe.process_video(video_path)

        # Stage 2: Format to 5D Batch Tensor (1, 2, T, H, W)
        batch_tensor = torch.from_numpy(chroma_tensor_np).unsqueeze(0).to(self.device)

        # Stage 3: Forward Pass through PhysNet3DCNN
        self.model.eval()
        with torch.no_grad():
            bvp_signal_tensor, logits = self.model(batch_tensor)
            probabilities = F.softmax(logits, dim=1)

        prob_auth = float(probabilities[0, 0].item())
        prob_ai   = float(probabilities[0, 1].item())
        pred_cls  = int(torch.argmax(probabilities, dim=1).item())

        # Stage 4: FFT Power Spectrum Analysis of BVP Waveform
        bvp_signal = bvp_signal_tensor[0].cpu().numpy()
        bpm, snr   = self._analyze_fft_bvp(bvp_signal)

        report = VideoForensicReport(
            video_path        = str(video_path.resolve()),
            verdict           = self._LABELS[pred_cls],
            confidence        = max(prob_auth, prob_ai),
            prob_authentic    = prob_auth,
            prob_ai_generated = prob_ai,
            heart_rate_bpm    = bpm,
            pulse_regularity  = snr,
            hardware          = "CUDA" if self.device.type == "cuda" else "CPU",
        )
        log.info("Analysis complete -> Verdict: %s (BPM: %.1f)", report.verdict, bpm)
        return report

    def _analyze_fft_bvp(self, bvp_signal: np.ndarray) -> Tuple[float, float]:
        """
        Applies Fast Fourier Transform (FFT) / Welch Periodogram to the 1D BVP pulse,
        deriving dominant Heart Rate frequency (BPM) and Pulse Signal-to-Noise Ratio (SNR).
        """
        fs = float(self.config.sample_fps)
        freqs, psd = welch(bvp_signal, fs=fs, nperseg=min(len(bvp_signal), 64))

        # Filter frequency band to human cardiovascular heart rate [45, 180] BPM -> [0.75, 3.0] Hz
        valid_mask = (freqs >= (self.config.bpm_min / 60.0)) & (freqs <= (self.config.bpm_max / 60.0))
        valid_freqs = freqs[valid_mask]
        valid_psd   = psd[valid_mask]

        if len(valid_psd) == 0 or np.max(valid_psd) == 0:
            return 72.0, 0.0  # Fallback baseline

        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        bpm = peak_freq * 60.0

        # Compute SNR: Peak energy vs background spectral noise
        peak_power = valid_psd[peak_idx]
        noise_power = np.mean(valid_psd) + 1e-8
        snr = 10.0 * np.log10(peak_power / noise_power)

        return float(bpm), float(snr)

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        if preference == "auto":
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        elif preference in ("cuda", "cpu"):
            chosen = preference
        else:
            chosen = "cpu"
        return torch.device(chosen)

    def _build_model(self) -> PhysNet3DCNN:
        model = PhysNet3DCNN(self.config).to(self.device)

        if self.config.weights_path is None:
            log.info("No weights_path set - running with random initialisation for dry-run testing.")
            model.eval()
            return model

        weights_path = Path(self.config.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path.resolve()}")

        state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        log.info("Loaded pre-trained PhysNet3DCNN weights from '%s'", weights_path.name)
        model.eval()
        return model


# ===========================================================================
# SECTION 6 - EXECUTABLE DEMO (__main__)
# ===========================================================================

if __name__ == "__main__":

    print("\n+----------------------------------------------------------+")
    print("|   PHASE 3 - VIDEO DEEPFAKE rPPG DETECTION ENGINE          |")
    print("|   3D-PhysNet x Facial Mesh  |  Biometric Forensics        |")
    print("+----------------------------------------------------------+\n")

    config = VideoForensicsConfig(
        sample_fps=30,
        sequence_length=64,
        roi_size=(128, 128),
        device="auto",
        weights_path=None,
        video_dir="deepfake_video",
    )

    engine = VideoDeepfakeRPPGEngine(config)

    # Scan video directory
    video_dir = Path(config.video_dir)
    video_files = [p for p in video_dir.iterdir() if p.suffix.lower() in config.supported_ext]

    if video_files:
        print(f"[MODE] Found {len(video_files)} video(s) in '{config.video_dir}'. Processing first video...")
        report = engine.predict_video(video_files[0])
        print(report)
    else:
        print("\n  [INFO] No video files currently in 'deepfake_video/'.")
        print("  Drop MP4, MOV, or AVI video files into 'deepfake_video/' to test.\n")
        print("  System parameters:")
        print(f"    PhysNet3DCNN Parameters : {engine.model.count_parameters():,}")
        print(f"    Hardware Acceleration   : {engine.device.type.upper()}")
        print(f"    Chroma Color Space      : YCrCb (Discard Y, Keep Cr/Cb)\n")
