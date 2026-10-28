"""
SEAA: Self-Evolving AI Auditor for Multimodal Reliability
Master Unified Application & Pipeline Runner

A single, unified command-center entry point to execute, evaluate, benchmark,
and audit the complete end-to-end SEAA multimodal reliability framework.

Modes:
  --mode full                Execute complete end-to-end pipeline (audit -> recovery -> evaluation).
  --mode audit               Run inline reliability auditing & multi-class failure diagnosis.
  --mode recovery            Run adaptive contextual bandit recovery across 8 failure regimes.
  --mode baselines           Train and benchmark unimodal (text/image) and late-fusion baselines.
  --mode ablations           Run component ablations isolating outcome learning & diagnosis.
  --mode latency             Run high-precision micro-benchmark profiling decision latency & throughput.
  --mode evaluate-diagnosis  Run diagnostic accuracy evaluation and confusion matrix generation.
  --mode verify              Verify environment, dataset accessibility, and model checkpoints.
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SEAA_MasterRunner")


def run_command(cmd_args, desc):
    """Executes a subprocess command with clear logging and status handling."""
    logger.info(f">>> [SEAA Pipeline] Executing: {desc}")
    logger.info(f"    Command: {' '.join(cmd_args)}")
    res = subprocess.run(cmd_args, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        logger.error(f"[FAILED] {desc} returned exit code {res.returncode}")
        sys.exit(res.returncode)
    logger.info(f">>> [SUCCESS] {desc} completed successfully.\n")


def mode_verify():
    """Verifies environment, dependencies, dataset, and checkpoints."""
    logger.info("=" * 80)
    logger.info("SEAA SYSTEM VERIFICATION: Environment, Hardware, Dataset & Checkpoints")
    logger.info("=" * 80)
    verify_script = PROJECT_ROOT / "scripts" / "verify_environment.py"
    run_command([sys.executable, str(verify_script)], "System & Environment Verification")

    # Check key project checkpoints
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    required_ckpts = [
        ckpt_dir / "text_baseline.joblib",
        ckpt_dir / "image_baseline.joblib",
        ckpt_dir / "multimodal_baseline.json"
    ]
    all_exist = all(p.exists() for p in required_ckpts)
    if all_exist:
        logger.info("[VERIFICATION] All required core model checkpoints are present.")
    else:
        logger.warning("[VERIFICATION] One or more checkpoints are missing in outputs/checkpoints/.")


def mode_baselines():
    """Trains and evaluates baseline models."""
    logger.info("=" * 80)
    logger.info("SEAA BASELINE ENGINE: Text, Image & Multimodal Late Fusion")
    logger.info("=" * 80)
    runner = PROJECT_ROOT / "scripts" / "run_baselines.py"
    run_command([sys.executable, str(runner)], "Baseline Training & Evaluation Suite")


def mode_audit(samples=1000, seed=42):
    """Runs inline reliability auditing and failure diagnosis."""
    logger.info("=" * 80)
    logger.info("SEAA INLINE AUDITING: Cross-Modal Disagreement, Uncertainty & Diagnosis")
    logger.info("=" * 80)
    audit_script = PROJECT_ROOT / "scripts" / "run_auditing.py"
    run_command([
        sys.executable, str(audit_script),
        "--sample_size", str(samples),
        "--seed", str(seed)
    ], "Inline Reliability Auditing & Failure Diagnosis")


def mode_recovery(sample_size=500, val_train_size=500, seed=42):
    """Runs adaptive recovery contextual bandit policy evaluation."""
    logger.info("=" * 80)
    logger.info("SEAA ADAPTIVE RECOVERY: Contextual Bandit Policy across 8 Failure Regimes")
    logger.info("=" * 80)
    recovery_script = PROJECT_ROOT / "scripts" / "run_recovery.py"
    run_command([
        sys.executable, str(recovery_script),
        "--sample_size", str(sample_size),
        "--val_train_size", str(val_train_size),
        "--seed", str(seed)
    ], "Adaptive Contextual Bandit Recovery Evaluation")


def mode_ablations():
    """Runs component ablations isolating outcome learning and diagnosis."""
    logger.info("=" * 80)
    logger.info("SEAA COMPONENT ABLATION: Isolating Outcome Learning & Context State")
    logger.info("=" * 80)
    ablation_script = PROJECT_ROOT / "scripts" / "run_ablation_benchmarks.py"
    run_command([sys.executable, str(ablation_script)], "Component Ablation Studies Suite")


def mode_latency():
    """Runs high-precision latency profiling."""
    logger.info("=" * 80)
    logger.info("SEAA EFFICIENCY PROFILING: High-Precision Latency & Throughput Benchmark")
    logger.info("=" * 80)
    latency_script = PROJECT_ROOT / "scripts" / "benchmark_latency_efficiency.py"
    run_command([sys.executable, str(latency_script)], "Latency & Efficiency Profiling")


def mode_evaluate_diagnosis():
    """Evaluates multi-class diagnostic accuracy and contingency tables."""
    logger.info("=" * 80)
    logger.info("SEAA DIAGNOSTIC ACCURACY: Confusion Matrix & Anomaly Detection Metrics")
    logger.info("=" * 80)
    diag_script = PROJECT_ROOT / "scripts" / "evaluate_diagnosis_performance.py"
    run_command([sys.executable, str(diag_script)], "Failure Diagnosis Accuracy Evaluation")


def mode_sensitivity(sample_size=500, seed=42):
    """Runs continuous corruption severity sensitivity benchmark."""
    logger.info("=" * 80)
    logger.info("SEAA SENSITIVITY BENCHMARK: Continuous Corruption Severity Sweeps")
    logger.info("=" * 80)
    sens_script = PROJECT_ROOT / "scripts" / "run_sensitivity_benchmark.py"
    run_command([
        sys.executable, str(sens_script),
        "--sample_size", str(sample_size),
        "--seed", str(seed)
    ], "Continuous Severity Sensitivity Benchmark")


def mode_ood(sample_size=500, seed=42):
    """Runs out-of-distribution (OOD) unseen corruption generalization benchmark."""
    logger.info("=" * 80)
    logger.info("SEAA OOD BENCHMARK: Generalization to Unseen Corruptions")
    logger.info("=" * 80)
    ood_script = PROJECT_ROOT / "scripts" / "run_ood_benchmark.py"
    run_command([
        sys.executable, str(ood_script),
        "--sample_size", str(sample_size),
        "--seed", str(seed)
    ], "OOD Generalization Benchmark")


def mode_case_studies():
    """Generates qualitative multi-modal audit trails and case studies."""
    logger.info("=" * 80)
    logger.info("SEAA QUALITATIVE INSPECTOR: Generating Case Study Reports")
    logger.info("=" * 80)
    case_script = PROJECT_ROOT / "scripts" / "generate_case_studies.py"
    run_command([sys.executable, str(case_script)], "Qualitative Case Study Telemetry Generator")


def mode_full(sample_size=500, seed=42):
    """Executes the complete unified SEAA pipeline end-to-end."""
    logger.info("=" * 80)
    logger.info("SEAA UNIFIED PIPELINE: Complete End-to-End Framework Execution")
    logger.info("=" * 80)
    mode_verify()
    mode_audit(samples=sample_size, seed=seed)
    mode_recovery(sample_size=sample_size, val_train_size=sample_size, seed=seed)
    mode_evaluate_diagnosis()
    logger.info("=" * 80)
    logger.info("SEAA COMPLETE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("All audit, recovery, and diagnostic artifacts have been generated.")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="SEAA Master Unified Pipeline Runner (Single Complete Project Entry Point)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "full",
            "audit",
            "recovery",
            "baselines",
            "ablations",
            "latency",
            "evaluate-diagnosis",
            "sensitivity",
            "ood",
            "case-studies",
            "verify"
        ],
        help="SEAA execution mode (default: full)."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Sample size for evaluation (default: 500)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)."
    )

    args = parser.parse_args()

    if args.mode == "verify":
        mode_verify()
    elif args.mode == "baselines":
        mode_baselines()
    elif args.mode == "audit":
        mode_audit(samples=args.sample_size, seed=args.seed)
    elif args.mode == "recovery":
        mode_recovery(sample_size=args.sample_size, seed=args.seed)
    elif args.mode == "ablations":
        mode_ablations()
    elif args.mode == "latency":
        mode_latency()
    elif args.mode == "evaluate-diagnosis":
        mode_evaluate_diagnosis()
    elif args.mode == "sensitivity":
        mode_sensitivity(sample_size=args.sample_size, seed=args.seed)
    elif args.mode == "ood":
        mode_ood(sample_size=args.sample_size, seed=args.seed)
    elif args.mode == "case-studies":
        mode_case_studies()
    elif args.mode == "full":
        mode_full(sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
