"""
test_audio.py — Real-Audio CLI Tester for the Deepfake Audio Detection Engine
───────────────────────────────────────────────────────────────────────────────
Usage
─────
  # Single file
  python test_audio.py path/to/audio.wav

  # Multiple files at once
  python test_audio.py clip1.wav clip2.mp3 recording.flac

  # With a trained checkpoint
  python test_audio.py --weights model.pth suspect.wav

Notes
─────
* Supports: .wav  .mp3  .flac  .ogg  .m4a  (anything librosa can decode)
* Without --weights the model uses RANDOM weights (pipeline test only).
* With a trained .pth checkpoint, results are forensically meaningful.
───────────────────────────────────────────────────────────────────────────────
"""

import sys
import argparse
from pathlib import Path

# ── Import the engine from the sibling module ────────────────────────────────
try:
    from deepfake_audio_detector import (
        AudioDeepfakeDetector,
        AUDIO_CFG,
        TCN_CFG,
        render_forensic_report,
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import deepfake_audio_detector.py.\n"
        f"  Make sure test_audio.py is in the same folder as deepfake_audio_detector.py.\n"
        f"  Detail: {e}"
    )


# ── Argument Parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "test_audio",
        description = "Deepfake Audio Detection Engine — Real-File CLI Tester",
        formatter_class = argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "audio_files",
        nargs   = "+",
        metavar = "AUDIO_FILE",
        help    = "Path(s) to audio file(s) to analyse (.wav, .mp3, .flac, etc.)",
    )
    parser.add_argument(
        "--weights", "-w",
        default = None,
        metavar = "PATH",
        help    = "Path to a trained .pth checkpoint (optional).\n"
                  "Without this flag the model uses random weights\n"
                  "(useful for pipeline validation, not forensic use).",
    )
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    # ── Validate that all supplied paths exist before initialising the model ─
    missing = [f for f in args.audio_files if not Path(f).exists()]
    if missing:
        print(f"\n{BOLD}[ERROR]{RESET} The following file(s) were not found:")
        for m in missing:
            print(f"  ✗  {m}")
        print("\n  Double-check the path and try again.")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}  Deepfake Audio Detection Engine — Real-Audio Test{RESET}")
    print(f"{CYAN}  Files to analyse : {len(args.audio_files)}{RESET}")
    if args.weights:
        print(f"{CYAN}  Checkpoint       : {args.weights}{RESET}")
    else:
        print(f"{YELLOW}  Checkpoint       : NONE (random weights — pipeline test only){RESET}")
    print()

    # ── Initialise the detector (once; shared across all files) ──────────────
    detector = AudioDeepfakeDetector(
        weights_path = args.weights,
        audio_cfg    = AUDIO_CFG,
        tcn_cfg      = TCN_CFG,
    )

    # ── Run inference on each file ───────────────────────────────────────────
    results = detector.predict_batch(args.audio_files)

    # ── Print a forensic report per file ────────────────────────────────────
    for result in results:
        print(render_forensic_report(result))

    # ── Batch summary ────────────────────────────────────────────────────────
    if len(results) > 1:
        deepfakes  = sum(1 for r in results if r.is_deepfake)
        authentic  = len(results) - deepfakes

        print(f"{BOLD}{CYAN}  BATCH SUMMARY{RESET}")
        print(f"  +-- Files processed : {len(results)}")
        print(f"  +-- Authentic       : {authentic}")
        print(f"  +-- Deepfake        : {deepfakes}")
        avg_conf = sum(r.confidence for r in results) / len(results)
        print(f"  +-- Avg confidence  : {avg_conf:.1f}%")
        print()


if __name__ == "__main__":
    main()
