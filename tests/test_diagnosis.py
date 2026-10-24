"""
Unit tests for Reliability Diagnosis and Failure Classification (src/auditing/diagnosis.py).
Tests:
1. DiagnosticConfig calibration strictly on validation distributions without test data.
2. Clean diagnosis identification when both modalities agree with low uncertainty.
3. Missing modality detection: MISSING_TEXT, MISSING_IMAGE, and MISSING_BOTH.
4. Asymmetric failure diagnosis: TEXT_FAILURE vs IMAGE_FAILURE.
5. Conflict and uncertainty diagnosis: CROSS_MODAL_CONFLICT and HIGH_UNCERTAINTY.
6. Binary corruption detection evaluation (Precision, Recall, F1, Specificity, Confusion Matrix).
7. Multi-class failure classification evaluation (Macro-F1, Per-Class metrics, Confusion Matrix).
"""

import unittest
import numpy as np

from src.auditing.diagnosis import (
    DiagnosticConfig,
    calibrate_diagnostic_thresholds_on_val,
    MultimodalReliabilityAuditor,
    evaluate_diagnosis_performance
)


class TestReliabilityDiagnosis(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_classes = 23
        self.text_thresh = np.full(self.n_classes, 0.45)
        self.image_thresh = np.full(self.n_classes, 0.50)

    def test_01_val_calibration_sets_valid_thresholds(self):
        """Calibration strictly determines 95th percentiles from validation arrays."""
        n_val = 100
        val_t_probs = np.random.uniform(0.1, 0.8, size=(n_val, self.n_classes))
        val_i_probs = np.random.uniform(0.1, 0.8, size=(n_val, self.n_classes))

        config = calibrate_diagnostic_thresholds_on_val(
            val_text_probs=val_t_probs,
            val_image_probs=val_i_probs,
            text_thresholds=self.text_thresh,
            image_thresholds=self.image_thresh,
            percentile=95.0
        )

        self.assertTrue(config.calibrated_on_val)
        self.assertGreater(config.jsd_threshold, 0.0)
        self.assertLessEqual(config.jsd_threshold, 1.0)
        self.assertGreater(config.jaccard_threshold, 0.0)
        self.assertLessEqual(config.jaccard_threshold, 1.0)
        self.assertGreater(config.text_uncertainty_threshold, 0.0)

    def test_02_diagnose_clean_sample(self):
        """A clean sample with high agreement and low uncertainty is diagnosed as CLEAN."""
        # Clean sample: text and image agree on genre 0 and 1 with high confidence, rest 0
        t_probs = np.array([[0.95, 0.90] + [0.05] * 21])
        i_probs = np.array([[0.90, 0.92] + [0.08] * 21])

        auditor = MultimodalReliabilityAuditor(
            text_thresholds=self.text_thresh,
            image_thresholds=self.image_thresh
        )
        res = auditor.audit_batch(t_probs, i_probs)

        self.assertEqual(res["diagnoses"][0], "CLEAN")
        self.assertFalse(res["is_anomalous"][0])
        self.assertLess(res["anomaly_scores"][0], 0.3)

    def test_03_diagnose_missing_modalities(self):
        """Missing modalities are immediately identified with high anomaly score."""
        t_probs = np.array([[0.8, 0.1] + [0.05] * 21] * 3)
        i_probs = np.array([[0.1, 0.8] + [0.05] * 21] * 3)

        auditor = MultimodalReliabilityAuditor(
            text_thresholds=self.text_thresh,
            image_thresholds=self.image_thresh
        )

        res = auditor.audit_batch(
            text_probs=t_probs,
            image_probs=i_probs,
            has_text=[False, True, False],
            has_image=[True, False, False]
        )

        self.assertEqual(res["diagnoses"][0], "MISSING_TEXT")
        self.assertTrue(res["is_anomalous"][0])

        self.assertEqual(res["diagnoses"][1], "MISSING_IMAGE")
        self.assertTrue(res["is_anomalous"][1])

        self.assertEqual(res["diagnoses"][2], "MISSING_BOTH")
        self.assertTrue(res["is_anomalous"][2])

    def test_04_diagnose_text_vs_image_failure(self):
        """Identifies asymmetric failure: high text uncertainty vs high image uncertainty."""
        # 1. Text Failure: text is near threshold 0.5 (high entropy), image is confident
        t_noisy = np.full((1, 23), 0.48)  # very high entropy
        i_confident = np.array([[0.95, 0.90] + [0.02] * 21])

        # 2. Image Failure: image is near threshold 0.5, text is confident
        t_confident = np.array([[0.95, 0.90] + [0.02] * 21])
        i_noisy = np.full((1, 23), 0.52)

        config = DiagnosticConfig(
            jsd_threshold=0.10,
            jaccard_threshold=0.20,
            text_uncertainty_threshold=0.50,
            image_uncertainty_threshold=0.50
        )
        auditor = MultimodalReliabilityAuditor(
            config=config,
            text_thresholds=self.text_thresh,
            image_thresholds=self.image_thresh
        )

        res_tf = auditor.audit_batch(t_noisy, i_confident)
        self.assertEqual(res_tf["diagnoses"][0], "TEXT_FAILURE")
        self.assertTrue(res_tf["is_anomalous"][0])

        res_imgf = auditor.audit_batch(t_confident, i_noisy)
        self.assertEqual(res_imgf["diagnoses"][0], "IMAGE_FAILURE")
        self.assertTrue(res_imgf["is_anomalous"][0])

    def test_05_diagnose_conflict_and_high_uncertainty(self):
        """Cross-modal conflict occurs when both confident but disagree; high uncertainty when both ambiguous."""
        # Conflict: text confident on genre 0, image confident on genre 1
        t_conf = np.array([[0.95, 0.05] + [0.05] * 21])
        i_conf = np.array([[0.05, 0.95] + [0.05] * 21])

        # Both ambiguous
        t_amb = np.full((1, 23), 0.50)
        i_amb = np.full((1, 23), 0.50)

        config = DiagnosticConfig(
            jsd_threshold=0.15,
            jaccard_threshold=0.30,
            text_uncertainty_threshold=0.70,
            image_uncertainty_threshold=0.70
        )
        auditor = MultimodalReliabilityAuditor(
            config=config,
            text_thresholds=self.text_thresh,
            image_thresholds=self.image_thresh
        )

        res_conf = auditor.audit_batch(t_conf, i_conf)
        self.assertEqual(res_conf["diagnoses"][0], "CROSS_MODAL_CONFLICT")

        res_amb = auditor.audit_batch(t_amb, i_amb)
        self.assertEqual(res_amb["diagnoses"][0], "HIGH_UNCERTAINTY")

    def test_06_evaluate_diagnosis_metrics(self):
        """Validates detection and multi-class classification metric calculations."""
        y_true_bin = [0, 0, 1, 1, 1]
        y_pred_bin = [0, 1, 1, 1, 0]  # 1 FP, 1 FN -> 3 correct

        y_true_types = ["CLEAN", "CLEAN", "TEXT_FAILURE", "IMAGE_FAILURE", "MISSING_TEXT"]
        y_pred_types = ["CLEAN", "CROSS_MODAL_CONFLICT", "TEXT_FAILURE", "IMAGE_FAILURE", "CLEAN"]

        eval_res = evaluate_diagnosis_performance(
            y_true_binary=y_true_bin,
            y_pred_binary=y_pred_bin,
            y_true_types=y_true_types,
            y_pred_types=y_pred_types
        )

        # Binary checks
        det = eval_res["binary_corruption_detection"]
        self.assertEqual(det["accuracy"], 0.6)
        self.assertEqual(det["confusion_matrix"]["true_positives"], 2)
        self.assertEqual(det["confusion_matrix"]["true_negatives"], 1)
        self.assertEqual(det["confusion_matrix"]["false_positives"], 1)
        self.assertEqual(det["confusion_matrix"]["false_negatives"], 1)

        # Multi-class checks
        cls_eval = eval_res["multiclass_failure_classification"]
        self.assertIsNotNone(cls_eval)
        self.assertEqual(cls_eval["accuracy"], 0.6)
        self.assertIn("TEXT_FAILURE", cls_eval["per_class"])
        self.assertEqual(cls_eval["per_class"]["TEXT_FAILURE"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
