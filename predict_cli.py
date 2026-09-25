"""
Command-Line Interface (CLI) for Student Mark Prediction.
Usage:
  python predict_cli.py --preset "Top Achiever (High Marks)"
  python predict_cli.py --preset "At-Risk Student (Needs Support)"
  python predict_cli.py --subject math --g1 15 --g2 16 --studytime 3 --absences 2
"""

import sys
from pathlib import Path
import argparse
import json

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.predict import StudentMarkPredictor, PRESET_STUDENTS


def format_cli_output(result: dict):
    """Render a clean CLI summary of the prediction."""
    mark = result["mark"]
    pct = result["percentage"]
    grade = result["letter_grade"]
    standing = result["standing"]
    status = result["status"]
    subject = result["subject"].upper()

    border = "=" * 62
    print(f"\n{border}")
    print(f"             STUDENT MARK PREDICTION REPORT [{subject}]")
    print(border)
    print(f"  * Predicted Final Mark (G3) :  {mark:.2f} / 20.00")
    print(f"  * Scaled Percentage         :  {pct:.1f}%")
    print(f"  * Academic Letter Grade     :  Grade {grade}")
    print(f"  * Performance Standing      :  {standing}")
    print(f"  * Outcome Status            :  {status.upper()}")
    print("-" * 62)
    print("  KEY RECOMMENDATIONS & INSIGHTS:")
    for r in result.get("recommendations", []):
        clean_r = r.replace("**", "").replace("⭐", "[*]").replace("⚠️", "[!]").replace("✅", "[+]").replace("📚", "[*]").replace("🎯", "[*]").replace("📈", "[+]").replace("📉", "[-]").replace("🎓", "[*]").replace("👍", "[+]")
        print(f"   {clean_r}")
    print(f"{border}\n")


def main():
    parser = argparse.ArgumentParser(description="Student Mark Prediction CLI")
    parser.add_argument(
        "--subject",
        type=str,
        default="math",
        choices=["math", "por", "combined"],
        help="Subject dataset model to use ('math', 'por', or 'combined')",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Use a predefined student persona: 'Top Achiever (High Marks)', 'Average Student (Moderate Marks)', 'At-Risk Student (Needs Support)'",
    )
    parser.add_argument("--g1", type=float, default=None, help="Period 1 grade (0-20)")
    parser.add_argument("--g2", type=float, default=None, help="Period 2 grade (0-20)")
    parser.add_argument("--studytime", type=int, default=2, help="Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    parser.add_argument("--absences", type=int, default=4, help="Number of absences (0-93)")
    parser.add_argument("--failures", type=int, default=0, help="Past class failures (0-4)")

    args = parser.parse_args()

    predictor = StudentMarkPredictor(subject=args.subject)

    if args.preset:
        matched_preset = None
        for key in PRESET_STUDENTS:
            if args.preset.lower() in key.lower():
                matched_preset = PRESET_STUDENTS[key].copy()
                print(f"[+] Loaded preset profile: '{key}'")
                break
        if matched_preset is None:
            print(f"[!] Unknown preset '{args.preset}'. Available presets:")
            for k in PRESET_STUDENTS:
                print(f"    - {k}")
            return
        matched_preset["subject"] = args.subject
        result = predictor.predict(matched_preset)
        format_cli_output(result)
        return

    # Use baseline template and override with CLI args
    student_data = PRESET_STUDENTS["Average Student (Moderate Marks)"].copy()
    student_data["subject"] = args.subject
    student_data["studytime"] = args.studytime
    student_data["absences"] = args.absences
    student_data["failures"] = args.failures

    if args.g1 is not None:
        student_data["G1"] = args.g1
    if args.g2 is not None:
        student_data["G2"] = args.g2

    result = predictor.predict(student_data)
    format_cli_output(result)


if __name__ == "__main__":
    main()
