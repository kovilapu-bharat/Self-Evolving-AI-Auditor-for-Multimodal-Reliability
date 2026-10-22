"""
Standalone runner script for ResNet50 feature extraction.
Logs progress to console and outputs/logs/image_feature_extraction.log.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.image_feature_extractor import ResNet50FeatureExtractor

def main():
    log_dir = project_root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "image_feature_extraction.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("ExtractImageFeatures")
    logger.info("Starting ResNet50 feature extraction pipeline...")

    output_dir = project_root / "data" / "processed"
    extractor = ResNet50FeatureExtractor(
        output_dir=output_dir,
        batch_size=64,
        num_threads=8,
        device="cpu"
    )
    extractor.extract_all()
    logger.info("Finished ResNet50 feature extraction.")

if __name__ == "__main__":
    main()
