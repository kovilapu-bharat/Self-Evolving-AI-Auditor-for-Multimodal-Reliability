"""
Script to prepare and validate the MM-IMDb dataset for SEAA.
Supports smoke testing and full dataset execution.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import MMIMDbDataPipeline

def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare MM-IMDb Dataset for SEAA")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test on 200 samples")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    args = parser.parse_args()

    log_path = project_root / "outputs" / "logs" / "data_pipeline.log"
    setup_logging(log_path)
    logger = logging.getLogger("PrepareDataset")
    logger.info("=== Starting MM-IMDb Dataset Preparation ===")

    max_samples = 200 if args.smoke_test else args.samples
    pipeline = MMIMDbDataPipeline(str(project_root))
    
    summary = pipeline.run(max_samples=max_samples, seed=args.seed)
    
    logger.info("=== Dataset Preparation Completed Successfully ===")
    logger.info(f"Total samples processed: {summary['total_samples']:,}")
    logger.info(f"Train samples: {summary['split_sizes']['train']:,}")
    logger.info(f"Val samples: {summary['split_sizes']['val']:,}")
    logger.info(f"Test samples: {summary['split_sizes']['test']:,}")
    logger.info(f"Metadata CSV: {project_root / 'data' / 'metadata.csv'}")
    logger.info(f"Label Mapping: {project_root / 'data' / 'label_mapping.json'}")
    logger.info(f"Split Indices: {project_root / 'data' / 'split_indices.json'}")
    logger.info(f"Summary JSON: {project_root / 'outputs' / 'reports' / 'dataset_summary.json'}")
    logger.info(f"Quality Report: {project_root / 'outputs' / 'reports' / 'dataset_quality_report.md'}")

if __name__ == "__main__":
    main()
