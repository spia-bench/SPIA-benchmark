"""
Inter-Annotator Agreement Evaluation Pipeline

Evaluates agreement between two human annotators.
Three-step pipeline:
1. Align cross-annotations
2. Align subjects between annotators
3. Evaluate PII-level agreement

Configuration: Uses YAML config file for all parameters
Default config: config/evaluate_inter_config.yaml
"""

import subprocess
import argparse
import sys
import os
import yaml
from pathlib import Path


def run(cmd: list[str]):
    """Execute command and exit on failure"""
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate_inter(config: dict):
    """Execute inter-annotator evaluation pipeline using config"""

    # Extract config sections by script names
    input_cfg = config['input']
    align_cfg = config.get('align_subject_across_annotators', {})
    eval_cfg = config.get('check_human_labels_inter', {})

    # Setup paths from annotators_dir
    annotators_dir = Path(input_cfg['annotators_dir'])
    if not annotators_dir.exists():
        print(f"Error: Annotators directory not found: {annotators_dir}")
        sys.exit(1)

    # Extract data name from annotators_dir parent (e.g., TAB -> tab)
    data = annotators_dir.parent.name.lower()
    inter_dir = annotators_dir.parent / "inter_annotator"
    inter_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths (all files in inter_annotator directory)
    aligned_file = inter_dir / f"{data}_inter_aligned_{align_cfg.get('api_provider', 'anthropic')}_{align_cfg.get('model', 'claude-sonnet-4')}.jsonl"
    report_file = inter_dir / f"{data}_inter_evaluation.json"

    print(f"\n{'='*60}")
    print(f"Inter-Annotator Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Dataset: {data}")
    print(f"Annotators Directory: {annotators_dir}")
    print(f"Output Directory: {inter_dir}")
    print(f"Config: API={align_cfg.get('api_provider')}, Model={align_cfg.get('model')}")
    print(f"{'='*60}\n")

    # Step 1: Align cross-annotations
    print("\n[Step 1/3] Aligning cross-annotations...")
    run([
        sys.executable, "src/evaluate/inter_annotator/align_cross_annotations.py",
        "--annotators_dir", str(annotators_dir)
    ])

    # Step 2: Align subjects
    print("\n[Step 2/3] Aligning subjects between annotators...")
    align_cmd = [
        sys.executable, "src/evaluate/inter_annotator/align_subject_across_annotators.py",
        "--data", data,
        "--api_provider", align_cfg.get('api_provider', 'anthropic'),
        "--model", align_cfg.get('model', 'claude-sonnet-4'),
        "--processing_mode", align_cfg.get('processing_mode', 'concurrent'),
        "--num_workers", str(align_cfg.get('num_workers', 20))
    ]
    if align_cfg.get('document_limit'):
        align_cmd.extend(["--document_limit", str(align_cfg['document_limit'])])
    run(align_cmd)

    # Step 3: Evaluate PII-level agreement
    print("\n[Step 3/3] Evaluating PII-level agreement...")
    eval_cmd = [
        sys.executable, "src/evaluate/inter_annotator/check_human_labels_inter.py",
        "--data", data,
        "--path", str(aligned_file),
        "--api_provider", eval_cfg.get('api_provider', 'openai'),
        "--model", eval_cfg.get('model', 'gpt-4o-mini'),
        "--min_certainty", str(eval_cfg.get('min_certainty', 3)),
        "--location_max_depth", str(eval_cfg.get('location_max_depth', 4)),
        "--decider", eval_cfg.get('decider', 'rule_model'),
        "--num_workers", str(eval_cfg.get('num_workers', 20)),
        "--case_log_filter", eval_cfg.get('case_log_filter', 'low_score')
    ]
    if eval_cfg.get('tags_to_evaluate'):
        eval_cmd.extend(["--tags"] + eval_cfg['tags_to_evaluate'])
    if not eval_cfg.get('show_case_logs', True):
        eval_cmd.append("--no_case_logs")
    run(eval_cmd)

    print(f"\n{'='*60}")
    print(f"Inter-Annotator Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {report_file}")

    # Check if consensus annotation files were generated
    agreed = inter_dir / f"{data}_inter_agreed.jsonl"
    conflict = inter_dir / f"{data}_inter_conflict.jsonl"

    if agreed.exists() or conflict.exists():
        print(f"\nConsensus Annotation Files:")
        if agreed.exists():
            print(f"  Agreed: {agreed}")
        if conflict.exists():
            print(f"  Conflict: {conflict}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate inter-annotator agreement using YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended)
  python src/evaluate/inter_annotator/evaluate_inter.py --config config/evaluate_inter_config.yaml

  # Override annotators directory
  python src/evaluate/inter_annotator/evaluate_inter.py \\
    --config config/evaluate_inter_config.yaml \\
    --annotators_dir data/spia/annotators/panorama

Configuration:
  All parameters are defined in the YAML config file.
  See config/evaluate_inter_config.yaml for full documentation.

  Config sections:
    - input: Annotators directory path
    - align_cross_annotations: No additional parameters
    - align_subject_across_annotators: API provider, model, processing settings
    - check_human_labels_inter: Certainty, tags, decider, workers, logging

Output: data/spia/inter_annotator/{data}_inter_evaluation.json
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluate_inter_config.yaml",
        help="Path to YAML config file (default: config/evaluate_inter_config.yaml)"
    )

    # Override options
    parser.add_argument(
        "--annotators_dir",
        type=str,
        help="Override annotators directory from config"
    )

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print(f"Please create a config file or use the default: config/evaluate_inter_config.yaml")
        sys.exit(1)

    config = load_config(args.config)

    # Apply command-line overrides
    if args.annotators_dir:
        config['input']['annotators_dir'] = args.annotators_dir

    # Validate annotators directory exists
    annotators_dir = Path(config['input']['annotators_dir'])
    if not annotators_dir.exists():
        print(f"Error: Annotators directory not found: {annotators_dir}")
        sys.exit(1)

    evaluate_inter(config)
