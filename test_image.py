"""
=============================================================================
  TEST / VALIDATION SCRIPT FOR IMAGE DEEPFAKE DETECTOR
=============================================================================
  Usage:
      python test_image.py <path_to_image>
      python test_image.py deepfake_image/sample_real_photo.png
=============================================================================
"""

import sys
from pathlib import Path
from deepfake_image_detector import ImageForensicsConfig, ImageDeepfakeDetector

def validate_image(image_path: str, weights_path: str = None):
    config = ImageForensicsConfig(
        device="auto",
        weights_path=weights_path,
        image_dir="deepfake_image"
    )
    detector = ImageDeepfakeDetector(config)
    report = detector.predict_image(image_path)
    print(report)
    return report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default test if no argument provided
        target_path = "deepfake_image/sample_real_photo.png"
        print(f"[INFO] No image path provided. Testing default sample: {target_path}")

    weights = sys.argv[2] if len(sys.argv) > 2 else None
    validate_image(target_path, weights)
