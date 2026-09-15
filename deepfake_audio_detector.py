"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DEEPFAKE AUDIO DETECTION ENGINE  ·  Phase 1 of N                  ║
║          1D Temporal Convolutional Network (1D-TCN)  ·  PyTorch             ║
║                                                                              ║
║  Architecture  : Causal Dilated TCN → Global Average Pooling → Binary Head  ║
║  Features      : Log-Mel Spectrogram (40-dim, 16 kHz, dB-scaled)            ║
║  Classes       : 0 → Authentic Human Speech | 1 → Synthetic / Deepfake      ║
║                                                                              ║
║  Philosophy    : "import antigravity" — elegant, modular, ready to fly.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # Use the modern parametrizations API (PyTorch >= 1.11).
    # Fall back to the legacy path on older builds to stay compatible.
    if hasattr(torch.nn.utils, "parametrizations"):
        from torch.nn.utils.parametrizations import weight_norm
    else:
        from torch.nn.utils import weight_norm   # pragma: no cover
except ImportError as e:
    sys.exit(
        f"[FATAL] PyTorch is not installed. Run: pip install torch\n  -> {e}"
    )

try:
    import librosa
    import librosa.display
except ImportError as e:
    sys.exit(
        f"[FATAL] Librosa is not installed. Run: pip install librosa\n  -> {e}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepfakeAudioDetector")

# Suppress noisy librosa UserWarnings (e.g. PySoundFile fallback)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS & GLOBAL CONSTANTS  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AudioConfig:
    """
    Immutable configuration bundle for the audio pre-processing pipeline.

    All downstream components read from this dataclass so that a single
    change here propagates everywhere — no magic numbers scattered in code.
    """
    sample_rate: int   = 16_000   # Target sample rate for standardisation (Hz)
    n_mels:      int   = 40       # Number of Mel filter-bank channels
    n_fft:       int   = 1_024    # FFT window length (samples)
    hop_length:  int   = 512      # Hop between consecutive STFT frames (samples)
    top_db:      float = 80.0     # Dynamic range clamp when converting to dB
    mono:        bool  = True     # Always downmix to a single channel


@dataclass(frozen=True)
class TCNConfig:
    """
    Immutable configuration bundle for the 1D-TCN architecture.

    'num_channels' defines the feature-map width at each dilation level.
    'num_levels'   controls how many TemporalBlocks are stacked, which
    determines the maximum receptive-field = 2^(num_levels) x kernel_size.
    """
    input_channels:  int       = 40    # Must match AudioConfig.n_mels
    num_channels:    List[int] = field(
        default_factory=lambda: [64, 64, 128, 128, 256, 256]
    )                                   # Channel widths per TCN level
    kernel_size:     int       = 3      # Conv kernel size (small -> fast; bigger -> wider RF)
    dropout:         float     = 0.2    # Spatial dropout probability per TemporalBlock
    num_classes:     int       = 2      # Binary: 0=Authentic, 1=Deepfake


AUDIO_CFG = AudioConfig()
TCN_CFG   = TCNConfig()


# =============================================================================
#  SECTION 1 — TCN CORE ARCHITECTURE
# =============================================================================

class Chomp1d(nn.Module):
    """
    Causal Padding Clipper.

    PyTorch's Conv1d with `padding=(kernel_size-1)*dilation` adds 'chomp_size'
    extra time-steps at the *right* tail of the output tensor.  For strictly
    causal inference (i.e., the model must NEVER peek at future audio frames),
    we strip those trailing time-steps after every convolution.

    Without this, a convolution at time-step t would see inputs from t+1,
    t+2, ... which is forbidden in a real-time forensic detection pipeline.

    Args:
        chomp_size (int): Number of trailing samples to remove.
                          Equals (kernel_size - 1) * dilation.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slice the tensor along the time dimension (dim=2) to remove the
        causal-padding tail.

        Args:
            x: Shape [B, C, T + chomp_size]

        Returns:
            Tensor of shape [B, C, T]  — temporally aligned, causal output.
        """
        # [:, :, :-self.chomp_size] — keep everything except the last
        # `chomp_size` time-steps which represent "future leakage".
        return x[:, :, : -self.chomp_size].contiguous()


# ─────────────────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """
    A single residual Temporal Block — the fundamental building unit of the TCN.

    Internal data-flow per block
    ────────────────────────────
    Input -> Conv1d (dilated, causal) -> Chomp1d -> ReLU -> Dropout
          -> Conv1d (dilated, causal) -> Chomp1d -> ReLU -> Dropout
          -> [+ Residual (1x1 conv if channels change)] -> ReLU -> Output

    Key design choices
    ──────────────────
    * Weight Normalisation (`weight_norm`): Decouples the magnitude of
      the weight tensor from its direction, which stabilises gradient flow
      and accelerates convergence — especially important for deep stacks.

    * Dilation (2^i): Each successive TemporalBlock doubles its dilation
      factor, causing the receptive field to grow *exponentially* with depth.
      A stack of 6 blocks with kernel_size=3 reaches:
         RF = 2 * (2^0 + 2^1 + ... + 2^5) * (3-1) + 1 = 127 time-steps.

    * Residual Connection: Adds the block input directly to its output
      (optionally through a 1x1 conv to match channel dimensions), preventing
      vanishing/exploding gradients in deep networks.

    Args:
        n_inputs   (int)  : Number of input feature channels.
        n_outputs  (int)  : Number of output feature channels.
        kernel_size(int)  : Convolution kernel width.
        stride     (int)  : Convolution stride (kept at 1 for TCN).
        dilation   (int)  : Dilation factor — must be 2^block_index.
        padding    (int)  : Pre-applied causal padding = (kernel_size-1)*dilation.
        dropout    (float): Dropout probability for regularisation.
    """

    def __init__(
        self,
        n_inputs:    int,
        n_outputs:   int,
        kernel_size: int,
        stride:      int,
        dilation:    int,
        padding:     int,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()

        # -- First dilated causal convolution --------------------------------
        # padding is applied *before* the conv; Chomp1d removes the tail.
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1   = Chomp1d(padding)          # Strip future-leaking tail
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # -- Second dilated causal convolution (deeper feature extraction) ---
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # -- Sequential net (main branch) ------------------------------------
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )

        # -- Residual / Skip connection --------------------------------------
        # If input and output channel counts differ we need a 1x1 projection
        # conv to align dimensions before the residual addition.
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
            else None
        )

        # Final activation *after* residual addition
        self.relu = nn.ReLU()

        # Initialise all conv weights to a sensible default
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming (He) uniform initialisation for both convolutional layers.
        Appropriate for ReLU-activated networks as it accounts for the
        non-linearity's effect on variance during forward propagation.
        """
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Temporal Block.

        Args:
            x: Input tensor of shape [B, C_in, T]

        Returns:
            Output tensor of shape [B, C_out, T]  (time dimension preserved)
        """
        # Main branch: two stacked dilated causal convolutions
        out = self.net(x)

        # Residual branch: project input channels if needed
        residual = x if self.downsample is None else self.downsample(x)

        # Element-wise addition followed by final ReLU
        return self.relu(out + residual)


# ─────────────────────────────────────────────────────────────────────────────

class AudioTCNClassifier(nn.Module):
    """
    Full 1D Temporal Convolutional Network for Binary Audio Authenticity
    Classification.

    Architecture overview
    ─────────────────────
    Log-Mel Features [B, 40, T]
         |
         v
    +----------------------------------+
    |  TemporalBlock #0  dilation=2^0  |  (causal, weight-normed)
    |  TemporalBlock #1  dilation=2^1  |
    |  TemporalBlock #2  dilation=2^2  |
    |  TemporalBlock #3  dilation=2^3  |
    |  TemporalBlock #4  dilation=2^4  |
    |  TemporalBlock #5  dilation=2^5  |
    +----------------------------------+
         |  [B, 256, T]
         v
    Global Average Pooling (dim=2)
         |  [B, 256]
         v
    LayerNorm + Dropout
         |
         v
    Linear(256 -> 2)   — binary logit head
         |  [B, 2]
         v
    Softmax (at inference) -> [P(Authentic), P(Deepfake)]

    Args:
        input_channels (int)      : Feature dimension (= n_mels = 40).
        num_channels   (List[int]): Hidden channel widths per TCN level.
        kernel_size    (int)      : Conv kernel size.
        dropout        (float)    : Dropout probability.
        num_classes    (int)      : Number of output classes (2).
    """

    def __init__(
        self,
        input_channels: int       = TCN_CFG.input_channels,
        num_channels:   List[int] = None,
        kernel_size:    int       = TCN_CFG.kernel_size,
        dropout:        float     = TCN_CFG.dropout,
        num_classes:    int       = TCN_CFG.num_classes,
    ) -> None:
        super().__init__()

        if num_channels is None:
            num_channels = TCN_CFG.num_channels

        # -- Build the stack of TemporalBlocks --------------------------------
        temporal_blocks: List[nn.Module] = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            # Exponential dilation: doubles the historical receptive field
            # at each successive depth level.
            dilation_size = 2 ** i

            # Channel count: first block reads raw features; subsequent
            # blocks read from the previous block's output channels.
            in_channels  = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Causal padding = (kernel_size - 1) * dilation ensures the
            # conv output length equals the input length after Chomp1d.
            padding = (kernel_size - 1) * dilation_size

            temporal_blocks.append(
                TemporalBlock(
                    n_inputs    = in_channels,
                    n_outputs   = out_channels,
                    kernel_size = kernel_size,
                    stride      = 1,
                    dilation    = dilation_size,
                    padding     = padding,
                    dropout     = dropout,
                )
            )
            logger.debug(
                "TemporalBlock #%d | in=%d -> out=%d | dilation=2^%d=%d | padding=%d",
                i, in_channels, out_channels, i, dilation_size, padding,
            )

        self.tcn = nn.Sequential(*temporal_blocks)

        # -- Classification Head ---------------------------------------------
        # After temporal modelling we collapse the time dimension via
        # global average pooling — making the model robust to variable
        # audio clip lengths without any fixed-size flattening.
        final_channels = num_channels[-1]

        self.classifier = nn.Sequential(
            nn.LayerNorm(final_channels),   # Stabilise pre-linear activations
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes),
        )

        # Log total trainable parameter count at construction time
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "AudioTCNClassifier built  |  Levels=%d  |  Params=%s",
            num_levels,
            f"{total_params:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full TCN classifier.

        Args:
            x: Input Log-Mel tensor of shape [B, n_mels=40, T]

        Returns:
            Raw logits of shape [B, num_classes=2].
            Apply Softmax externally for probability output.
        """
        # -- TCN Temporal Feature Extraction ---------------------------------
        # x: [B, 40, T]  ->  features: [B, 256, T]
        features = self.tcn(x)

        # -- Global Average Pooling across the time axis ---------------------
        # Collapses T dimension: [B, 256, T]  ->  [B, 256]
        # This makes inference length-agnostic (any audio duration works).
        pooled = features.mean(dim=2)

        # -- Classification Head ---------------------------------------------
        # [B, 256]  ->  [B, 2]
        logits = self.classifier(pooled)

        return logits


# =============================================================================
#  SECTION 2 — AUDIO PREPROCESSING PIPELINE
# =============================================================================

def ingest_audio(file_path: str, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """
    Audio Ingestion & Standardisation.

    Loads any supported audio file (.wav, .mp3, .flac, .ogg, etc.) via
    Librosa and standardises it to a single, consistent format required
    by the feature extraction pipeline:
        * Sample Rate : 16 000 Hz  (telephone-quality; sufficient for speech)
        * Channels    : 1 (mono)   (stereo information is forensically irrelevant here)

    Librosa handles all format-specific decoding internally (via soundfile /
    audioread / ffmpeg), so no manual codec management is required.

    Args:
        file_path (str)     : Absolute or relative path to the audio file.
        cfg       (AudioConfig): Pipeline configuration (sample rate, etc.).

    Returns:
        waveform (np.ndarray): 1-D float32 array of shape [num_samples].
                               Values are normalised to the range [-1.0, +1.0].

    Raises:
        FileNotFoundError : If the audio file does not exist.
        RuntimeError      : If Librosa cannot decode the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"[INGEST] Audio file not found: '{file_path}'\n"
            "  -> Please provide a valid .wav or .mp3 file path."
        )

    logger.info("Ingesting audio: '%s'", path.name)

    try:
        waveform, sr_original = librosa.load(
            str(path),
            sr=cfg.sample_rate,   # Resample to target SR during load
            mono=cfg.mono,        # Downmix stereo -> mono if needed
        )
    except Exception as exc:
        raise RuntimeError(
            f"[INGEST] Failed to decode '{file_path}': {exc}"
        ) from exc

    logger.info(
        "  OK Loaded  |  Original SR: %d Hz  ->  Resampled to: %d Hz  |  "
        "Duration: %.2f s  |  Samples: %d",
        sr_original if sr_original else cfg.sample_rate,
        cfg.sample_rate,
        len(waveform) / cfg.sample_rate,
        len(waveform),
    )

    return waveform


def extract_log_mel_features(
    waveform: np.ndarray,
    cfg: AudioConfig = AUDIO_CFG,
) -> np.ndarray:
    """
    Log-Mel Spectrogram Feature Extraction.

    Converts a raw waveform into a 2-D acoustic feature matrix that the
    TCN can learn from.  The pipeline is:

        Waveform [T_samples]
            |
            v  Short-Time Fourier Transform (STFT)
        Complex Spectrogram  [n_fft/2+1, T_frames]
            |
            v  Mel filter-bank projection (n_mels=40 triangular filters)
        Mel Power Spectrogram  [40, T_frames]
            |
            v  Power -> dB conversion  (log-compression)
        Log-Mel Spectrogram  [40, T_frames]   <- this is what we return

    Why Log-Mel?
    ─────────────
    * Mel scale mirrors human auditory perception; important spectral
      artefacts introduced by vocoders (HiFi-GAN, WaveGlow, etc.) appear
      as anomalous energy patterns in this representation.
    * Log compression (dB) is perceptually meaningful and compresses
      the dynamic range, making gradient-based optimisation more stable.
    * 40 channels is a well-established sweet-spot for speech: broad enough
      to capture phonetic content, compact enough to keep the model lean.

    Args:
        waveform (np.ndarray): 1-D float32 waveform, normalised to [-1, +1].
        cfg (AudioConfig)    : Pipeline configuration.

    Returns:
        log_mel (np.ndarray): 2-D float32 matrix of shape [n_mels=40, T_frames].
                              Values are in decibels, clipped to top_db range.
    """
    logger.info("Extracting Log-Mel features  |  n_mels=%d  n_fft=%d  hop=%d",
                cfg.n_mels, cfg.n_fft, cfg.hop_length)

    # Step 1: Compute the Mel power spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y          = waveform,
        sr         = cfg.sample_rate,
        n_fft      = cfg.n_fft,        # FFT window -> frequency resolution
        hop_length = cfg.hop_length,   # Frame shift -> temporal resolution
        n_mels     = cfg.n_mels,       # Mel filter bank channels
        power      = 2.0,              # Power (not amplitude) spectrogram
    )

    # Step 2: Convert power spectrogram to decibels (log-compression)
    # ref=np.max normalises to 0 dB peak; top_db clips the dynamic range.
    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref    = np.max,
        top_db = cfg.top_db,
    )

    logger.info(
        "  OK Feature matrix shape: %s  |  Range: [%.1f, %.1f] dB",
        log_mel.shape,
        log_mel.min(),
        log_mel.max(),
    )

    return log_mel.astype(np.float32)


# =============================================================================
#  SECTION 3 — THE PRODUCTION ENGINE & UTILITIES
# =============================================================================

@dataclass
class DetectionResult:
    """
    Structured container for a single inference result.

    Carrying results in a typed dataclass (instead of raw dicts or tuples)
    makes downstream reporting, logging, and API serialisation clean and safe.
    """
    file_path:          str
    authentic_prob:     float   # P(Class 0 — Real Human Speech)
    deepfake_prob:      float   # P(Class 1 — Synthetic / Deepfake)
    predicted_class:    int     # 0 = Authentic | 1 = Deepfake
    predicted_label:    str     # "AUTHENTIC" | "DEEPFAKE"
    confidence:         float   # Max(authentic_prob, deepfake_prob) * 100 %
    device_used:        str     # "cuda" or "cpu"
    feature_shape:      Tuple[int, int]   # (n_mels, T_frames)

    @property
    def is_deepfake(self) -> bool:
        return self.predicted_class == 1


# ─────────────────────────────────────────────────────────────────────────────

class AudioDeepfakeDetector:
    """
    Production Orchestration Wrapper for the 1D-TCN Deepfake Audio Engine.

    This class is the single public interface for all external callers (CLI,
    API server, batch pipeline).  It handles:

    (1) Hardware selection   — automatically chooses CUDA GPU over CPU.
    (2) Model instantiation  — builds AudioTCNClassifier from TCNConfig.
    (3) Weight loading       — safely loads a .pth checkpoint with full
                               exception handling; falls back to random weights
                               with a prominent warning if the file is absent.
    (4) Inference            — exposes `predict_audio()` for single-file
                               and `predict_batch()` for multi-file scoring.

    Usage
    ─────
    >>> detector = AudioDeepfakeDetector(weights_path="model.pth")
    >>> result   = detector.predict_audio("suspect_clip.wav")
    >>> print(result.predicted_label, result.confidence)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        audio_cfg:    AudioConfig   = AUDIO_CFG,
        tcn_cfg:      TCNConfig     = TCN_CFG,
    ) -> None:
        """
        Initialise the detection engine.

        Args:
            weights_path (str, optional): Path to a .pth model checkpoint.
                If None or the file is missing, the model runs with untrained
                (random) weights and a warning is emitted. In production,
                always supply a valid checkpoint path.
            audio_cfg (AudioConfig): Audio preprocessing parameters.
            tcn_cfg   (TCNConfig)  : TCN architecture parameters.
        """
        self.audio_cfg = audio_cfg
        self.tcn_cfg   = tcn_cfg

        # -- Hardware Accelerator Selection ----------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Hardware target: %s%s",
            str(self.device).upper(),
            f"  ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "",
        )

        # -- Model Construction ----------------------------------------------
        self.model = AudioTCNClassifier(
            input_channels = tcn_cfg.input_channels,
            num_channels   = list(tcn_cfg.num_channels),
            kernel_size    = tcn_cfg.kernel_size,
            dropout        = tcn_cfg.dropout,
            num_classes    = tcn_cfg.num_classes,
        ).to(self.device)

        # -- Weight Loading (with graceful degradation) ----------------------
        self._load_weights(weights_path)

        # Lock the model into evaluation mode — disables dropout, batchnorm
        # running-stats updates, and gradient computation.
        self.model.eval()
        logger.info("Model locked in EVAL mode  |  Inference-ready OK")

    def _load_weights(self, weights_path: Optional[str]) -> None:
        """
        Safely load pre-trained weights from a .pth checkpoint file.

        Handles three scenarios robustly:
            A) weights_path is None     -> warn and continue (random weights).
            B) File does not exist      -> warn and continue (random weights).
            C) File exists but corrupt  -> raise RuntimeError (hard fail).

        The production contract is that inference with random weights will
        always run (useful for integration tests), but results will be
        forensically meaningless — the warning makes this explicit.

        Args:
            weights_path (str | None): Path to the .pth checkpoint.
        """
        if weights_path is None:
            logger.warning(
                "[!] No weights_path supplied.  Model running with RANDOM weights.\n"
                "    Results are NOT forensically valid.  "
                "Supply a trained .pth checkpoint for production use."
            )
            return

        path = Path(weights_path)

        if not path.exists():
            logger.warning(
                "[!] Weights file not found: '%s'\n"
                "    Model running with RANDOM weights.  "
                "Train the model and supply a .pth checkpoint.",
                weights_path,
            )
            return

        try:
            # map_location ensures a GPU-trained checkpoint loads on CPU too
            state_dict = torch.load(
                str(path),
                map_location=self.device,
                weights_only=True,     # Security: refuse arbitrary pickle code
            )
            self.model.load_state_dict(state_dict)
            logger.info("OK Pre-trained weights loaded from: '%s'", weights_path)

        except RuntimeError as exc:
            # Architecture mismatch (e.g., different num_channels config)
            raise RuntimeError(
                f"[WEIGHTS] Architecture mismatch loading '{weights_path}'.\n"
                f"  Ensure TCNConfig matches the checkpoint's config.\n"
                f"  Detail: {exc}"
            ) from exc

        except Exception as exc:
            # Corrupt file, permission error, etc.
            raise RuntimeError(
                f"[WEIGHTS] Failed to load '{weights_path}': {exc}"
            ) from exc

    # ── Public Inference Interface ─────────────────────────────────────────

    def predict_audio(self, file_path: str) -> DetectionResult:
        """
        Run end-to-end deepfake detection on a single audio file.

        Pipeline
        ────────
        file_path (str)
            -> ingest_audio()             : decode + resample -> waveform [T]
            -> extract_log_mel_features() : STFT + Mel + dB  -> matrix [40, T_frames]
            -> tensor reshape             : [40, T_frames]   -> [1, 40, T_frames]  (3-D)
            -> model(x)                   : logits [1, 2]
            -> Softmax                    : probabilities [P_authentic, P_deepfake]
            -> DetectionResult            : structured output

        Args:
            file_path (str): Path to the audio file to analyse.

        Returns:
            DetectionResult: Structured inference result with probabilities,
                             predicted label, confidence, and metadata.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError     : If decoding or inference fails.
        """
        logger.info("-- Inference Start -----------------------------------------")

        # -- Step 1: Audio Ingestion & Standardisation -----------------------
        waveform = ingest_audio(file_path, self.audio_cfg)

        # -- Step 2: Feature Extraction (Log-Mel Spectrogram) ----------------
        log_mel = extract_log_mel_features(waveform, self.audio_cfg)
        # log_mel.shape == (n_mels=40, T_frames)

        # -- Step 3: Tensor Preparation --------------------------------------
        # The TCN expects a 3-D batch tensor: [Batch, Channels, Time]
        # We add a batch dimension of 1 for single-sample inference.
        tensor = (
            torch.from_numpy(log_mel)          # np.ndarray -> torch.Tensor
            .unsqueeze(0)                       # [40, T] -> [1, 40, T]
            .to(self.device)                    # Route to GPU/CPU
        )
        logger.info(
            "  Tensor routed to %s  |  Shape: %s",
            str(self.device).upper(), list(tensor.shape),
        )

        # -- Step 4: Model Forward Pass (no gradient tracking) ---------------
        with torch.no_grad():
            logits = self.model(tensor)         # Shape: [1, 2]

        # -- Step 5: Probability Calculation via Softmax ---------------------
        probs    = F.softmax(logits, dim=1)        # [1, 2] -> normalised [0,1]
        probs_np = probs.squeeze(0).cpu().numpy()  # [2]

        authentic_prob = float(probs_np[0])
        deepfake_prob  = float(probs_np[1])

        predicted_class = int(np.argmax(probs_np))
        predicted_label = "AUTHENTIC" if predicted_class == 0 else "DEEPFAKE"
        confidence      = float(np.max(probs_np)) * 100.0

        logger.info(
            "  OK Inference complete  |  %s  |  Confidence: %.2f%%",
            predicted_label, confidence,
        )
        logger.info("-- Inference End -------------------------------------------")

        return DetectionResult(
            file_path       = str(file_path),
            authentic_prob  = authentic_prob,
            deepfake_prob   = deepfake_prob,
            predicted_class = predicted_class,
            predicted_label = predicted_label,
            confidence      = confidence,
            device_used     = str(self.device),
            feature_shape   = (log_mel.shape[0], log_mel.shape[1]),
        )

    def predict_batch(self, file_paths: List[str]) -> List[DetectionResult]:
        """
        Run inference over a list of audio files sequentially.

        In a production environment this would be replaced by a true
        batched DataLoader pipeline, but for Phase 1 the sequential
        approach is clean, predictable, and easy to debug.

        Args:
            file_paths (List[str]): Paths to audio files.

        Returns:
            List[DetectionResult]: One result per valid input file.
                                   Failed files are logged and skipped.
        """
        results: List[DetectionResult] = []
        total = len(file_paths)

        for idx, fp in enumerate(file_paths, start=1):
            logger.info("[%d/%d] Processing: %s", idx, total, fp)
            try:
                results.append(self.predict_audio(fp))
            except (FileNotFoundError, RuntimeError) as exc:
                logger.error("  SKIP '%s': %s", fp, exc)

        return results


# =============================================================================
#  SECTION 4 — REPORTING UTILITIES
# =============================================================================

def render_forensic_report(result: DetectionResult) -> str:
    """
    Render a structured, human-readable forensic report for stdout.

    The report is styled to resemble a simplified digital forensics output
    brief — clear sections, confidence bars, and verdict highlighting — so
    that both engineers and non-technical stakeholders can interpret results
    at a glance.

    Args:
        result (DetectionResult): The inference result to render.

    Returns:
        report (str): Multi-line formatted string, ready to print.
    """
    # ANSI escape codes for terminal colour output
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"

    verdict_color = GREEN if not result.is_deepfake else RED

    report = (
        f"\n"
        f"{BOLD}{CYAN}+======================================================================+{RESET}\n"
        f"{BOLD}{CYAN}|       DEEPFAKE AUDIO DETECTION ENGINE  *  FORENSIC REPORT           |{RESET}\n"
        f"{BOLD}{CYAN}+======================================================================+{RESET}\n"
        f"\n"
        f"{BOLD}  FILE ANALYSED{RESET}\n"
        f"  +-- Path          : {result.file_path}\n"
        f"  +-- Feature Shape : {result.feature_shape[0]} mel-bands x {result.feature_shape[1]} time-frames\n"
        f"  +-- Hardware Used : {result.device_used.upper()}\n"
        f"\n"
        f"{BOLD}  ACOUSTIC PROBABILITIES{RESET}\n"
        f"  +-- [REAL]  Real Human Speech  : {result.authentic_prob * 100:6.2f}%  "
        f"{_build_bar(result.authentic_prob * 100, 20)}\n"
        f"  +-- [FAKE]  Synthetic / Clone  : {result.deepfake_prob  * 100:6.2f}%  "
        f"{_build_bar(result.deepfake_prob  * 100, 20)}\n"
        f"\n"
        f"{BOLD}  OVERALL CONFIDENCE{RESET}\n"
        f"  +-- {_build_bar(result.confidence, 30)}  {result.confidence:.1f}%\n"
        f"\n"
        f"{BOLD}  VERDICT{RESET}\n"
        f"  +-- {verdict_color}{BOLD}"
        f"{'[PASS] AUTHENTIC  -- No synthetic artefacts detected.' if not result.is_deepfake else '[FAIL] DEEPFAKE   -- Synthetic voice cloning suspected.'}"
        f"{RESET}\n"
        f"\n"
        f"{BOLD}{CYAN}+======================================================================+{RESET}\n"
        f"{MAGENTA}  [!] This result is probabilistic.  Always corroborate with a second   {RESET}\n"
        f"{MAGENTA}      modality (Phase 2: Visual Lip-Sync Analysis) before legal action. {RESET}\n"
        f"{BOLD}{CYAN}+======================================================================+{RESET}\n"
    )

    return report


def _build_bar(value: float, width: int = 20, fill: str = "#", empty: str = "-") -> str:
    """
    Build a simple ASCII progress bar scaled to 0-100%.

    Args:
        value (float) : A percentage value in [0, 100].
        width (int)   : Total character width of the bar.
        fill  (str)   : Character for the filled portion.
        empty (str)   : Character for the empty portion.

    Returns:
        str: A bar string like '[##########----------]'.
    """
    filled = int(round(max(0.0, min(value, 100.0)) / 100.0 * width))
    bar    = fill * filled + empty * (width - filled)
    return f"[{bar}]"


# =============================================================================
#  SECTION 5 — EXECUTABLE DEMO  (__main__ entry point)
# =============================================================================

def _generate_mock_audio_file(path: str, duration_s: float = 3.0) -> str:
    """
    Generate a synthetic sine-wave .wav file for demo/testing purposes.

    This allows the demo to run without requiring an actual audio recording,
    making the full pipeline testable in any CI environment or local setup.

    The signal: 440 Hz tone (concert A) at 16 kHz sample rate.

    Args:
        path       (str)  : Where to save the generated .wav file.
        duration_s (float): Length of the generated clip in seconds.

    Returns:
        str: The path to the generated .wav file (empty if unsaveable).
    """
    sr = AUDIO_CFG.sample_rate
    t  = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # 440 Hz sine wave (concert A) — clean, deterministic, device-agnostic
    waveform_float = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Attempt 1: soundfile (most reliable, supports float32 WAV natively)
    try:
        import soundfile as sf
        sf.write(path, waveform_float, sr)
        logger.info("Mock audio file written via soundfile: '%s'", path)
        return path
    except ImportError:
        pass

    # Attempt 2: scipy.io.wavfile (int16 WAV)
    try:
        from scipy.io import wavfile as sciwav
        waveform_int = (waveform_float * 32767).astype(np.int16)
        sciwav.write(path, sr, waveform_int)
        logger.info("Mock audio file written via scipy: '%s'", path)
        return path
    except ImportError:
        pass

    # Attempt 3: wave stdlib (pure Python, no third-party dependency)
    try:
        import wave
        import struct
        waveform_int = (waveform_float * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)           # mono
            wf.setsampwidth(2)           # 16-bit samples = 2 bytes
            wf.setframerate(sr)
            wf.writeframes(
                struct.pack(f"<{len(waveform_int)}h", *waveform_int)
            )
        logger.info("Mock audio file written via stdlib wave: '%s'", path)
        return path
    except Exception as exc:
        logger.warning("Could not write mock .wav file: %s", exc)
        return ""   # Signal to caller: file not written, use in-memory fallback


if __name__ == "__main__":
    """
    End-to-End Demo
    ───────────────────────────────────────────────────────────────────────────
    This entry point demonstrates the complete Phase-1 pipeline:

    1. Engine initialisation (no checkpoint supplied -> random weights demo)
    2. Mock audio generation (440 Hz sine wave -> 16 kHz WAV)
    3. Full inference pass (ingest -> feature extract -> TCN -> Softmax)
    4. Forensic report rendering to stdout

    In production, replace `mock_audio_path` with a real suspect audio
    file, and supply a trained checkpoint path to `AudioDeepfakeDetector`.
    ───────────────────────────────────────────────────────────────────────────
    """

    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RESET   = "\033[0m"

    print(f"\n{BOLD}{CYAN}  ============================================================{RESET}")
    print(f"{BOLD}{CYAN}  Deepfake Audio Detection Engine  --  Phase 1 Demo{RESET}")
    print(f"{BOLD}{CYAN}  1D Temporal Convolutional Network (1D-TCN)  *  PyTorch{RESET}")
    print(f"{BOLD}{CYAN}  ============================================================{RESET}\n")

    # ── Step 1: Initialise the Detection Engine ───────────────────────────
    print(f"{BOLD}[STEP 1/4]{RESET}  Initialising detection engine ...")

    detector = AudioDeepfakeDetector(
        weights_path = None,   # <- No checkpoint: demo with random weights
        audio_cfg    = AUDIO_CFG,
        tcn_cfg      = TCN_CFG,
    )

    # ── Step 2: Prepare a Mock Audio File ─────────────────────────────────
    print(f"\n{BOLD}[STEP 2/4]{RESET}  Generating mock audio signal (440 Hz sine, 3 s) ...")

    mock_audio_path = "demo_suspect_audio.wav"
    generated_path  = _generate_mock_audio_file(mock_audio_path, duration_s=3.0)

    # ── Step 3: Run Inference ─────────────────────────────────────────────
    print(f"\n{BOLD}[STEP 3/4]{RESET}  Running full TCN inference pipeline ...")

    if generated_path and Path(generated_path).exists():
        # Happy path: file was written, run the full orchestrated pipeline
        result = detector.predict_audio(generated_path)

    else:
        # Graceful fallback: generate waveform in-memory and bypass ingest_audio
        logger.info("In-memory fallback: bypassing disk I/O for demo inference.")

        sr  = AUDIO_CFG.sample_rate
        t   = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        mock_waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        log_mel = extract_log_mel_features(mock_waveform, AUDIO_CFG)
        tensor  = torch.from_numpy(log_mel).unsqueeze(0).to(detector.device)

        with torch.no_grad():
            logits = detector.model(tensor)

        probs        = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        auth_p       = float(probs[0])
        fake_p       = float(probs[1])
        pred_cls     = int(np.argmax(probs))

        result = DetectionResult(
            file_path       = mock_audio_path + "  [in-memory fallback]",
            authentic_prob  = auth_p,
            deepfake_prob   = fake_p,
            predicted_class = pred_cls,
            predicted_label = "AUTHENTIC" if pred_cls == 0 else "DEEPFAKE",
            confidence      = float(np.max(probs)) * 100.0,
            device_used     = str(detector.device),
            feature_shape   = (log_mel.shape[0], log_mel.shape[1]),
        )

    # ── Step 4: Render the Forensic Report ────────────────────────────────
    print(f"\n{BOLD}[STEP 4/4]{RESET}  Rendering forensic report ...")
    print(render_forensic_report(result))

    # ── Cleanup: remove the temporary demo file ────────────────────────────
    demo_file = Path(mock_audio_path)
    if demo_file.exists():
        demo_file.unlink()
        logger.info("Cleaned up temporary demo file: '%s'", mock_audio_path)

    print(f"{BOLD}{YELLOW}  NOTE: Results above used RANDOM (untrained) weights.{RESET}")
    print(f"{YELLOW}  Supply a trained .pth checkpoint for forensically valid output.{RESET}")
    print(f"\n{BOLD}  Demo complete.  See forensic report above for the verdict.{RESET}\n")
    sys.exit(0)
