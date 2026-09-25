"""
ChronoEye Infinity - ANPR Verification Evaluation Runner
Calculates summary metrics from data/diagnostic_crops/anpr_verification.csv
after human ground truth annotations have been filled.

Usage:
E:\\chronoeye\\chronoeye\\Scripts\\python.exe evaluate_anpr_verification.py
"""

import sys
import os

# Add backend to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.perception.anpr_verification import ANPRVerifier


def main():
    csv_path = os.path.abspath(r"data/diagnostic_crops/anpr_verification.csv")
    print(f"=" * 60)
    print("CHRONOEYE INFINITY - ANPR VERIFICATION EVALUATION REPORT")
    print(f"=" * 60)
    print(f"Target CSV: {csv_path}\n")

    metrics = ANPRVerifier.evaluate_verification_csv(csv_path)

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        sys.exit(1)

    total = metrics.get("total_samples", 0)
    print(f"Total Representative Samples: {total}")
    if total == 0:
        print("No samples found in CSV. Run video detection to generate verification crops first.")
        sys.exit(0)

    print(f"Correct Recognitions:         {metrics['correct_recognitions']}")
    print(f"Wrong Recognitions:           {metrics['wrong_recognitions']}")
    print(f"Unclear Samples:              {metrics['unclear_samples']}")
    print(f"Clear Samples Evaluated:      {metrics['clear_samples']}")
    print(f"Recognition Accuracy (Clear): {metrics['accuracy_among_clear'] * 100:.2f}%")
    print(f"Indian Format Pass Rate:      {metrics['indian_format_pass_rate'] * 100:.2f}%")
    print(f"Average OCR Confidence:       {metrics['average_ocr_confidence']:.4f}")

    print(f"\n--- SPECIAL CASE SUMMARY ---")
    print(f"Rejected by Format, but Human-Correct: {metrics['rejected_but_correct_count']}")
    for ex in metrics.get("rejected_but_correct_examples", []):
        print(f"  - [{ex.get('crop_id')}] Raw: '{ex.get('raw_ocr')}' -> Ground Truth: '{ex.get('human_ground_truth')}'")

    print(f"\nHigh Confidence (>= 0.70), but Human-Wrong: {metrics['high_conf_wrong_count']}")
    for ex in metrics.get("high_conf_wrong_examples", []):
        print(f"  - [{ex.get('crop_id')}] Raw: '{ex.get('raw_ocr')}' (conf={ex.get('ocr_confidence')}) -> Ground Truth: '{ex.get('human_ground_truth')}'")

    print(f"=" * 60)


if __name__ == "__main__":
    main()
