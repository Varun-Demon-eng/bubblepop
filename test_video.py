"""
=============================================================================
  TEST / VALIDATION SCRIPT FOR VIDEO rPPG DEEPFAKE DETECTOR
=============================================================================
  Usage:
      python test_video.py <path_to_video.mp4> [weights_path]
=============================================================================
"""

import sys
from pathlib import Path
from deepfake_video_detector import VideoForensicsConfig, VideoDeepfakeRPPGEngine

def validate_video(video_path: str, weights_path: str = None):
    config = VideoForensicsConfig(
        device="auto",
        weights_path=weights_path if (weights_path and Path(weights_path).exists()) else None,
        video_dir="deepfake_video"
    )
    engine = VideoDeepfakeRPPGEngine(config)
    report = engine.predict_video(video_path)
    print(report)
    return report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Scan video_dir for first video
        vdir = Path("deepfake_video")
        vfiles = [p for p in vdir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
        if vfiles:
            target_path = str(vfiles[0])
            print(f"[INFO] Testing video: {target_path}")
        else:
            print("[INFO] Usage: python test_video.py <path_to_video.mp4>")
            sys.exit(0)

    weights = sys.argv[2] if len(sys.argv) > 2 else "video_model.pth"
    validate_video(target_path, weights)
