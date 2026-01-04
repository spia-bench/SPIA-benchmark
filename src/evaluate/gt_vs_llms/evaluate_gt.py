"""
Ground Truth Evaluation Pipeline

Evaluates LLM annotation performance against human ground truth annotations.
Three-step pipeline:
1. Combine GT and LLM annotations
2. Align subjects between GT and LLM
3. Evaluate PII-level agreement

Configuration: Uses YAML config file for all parameters
Default config: config/evaluate_gt_config.yaml
"""

import subprocess
import argparse
import sys
import os
import json
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


def evaluate_gt(config: dict):
    """Execute GT evaluation pipeline using config"""

    # Extract config sections by script names
    input_cfg = config['input']
    output_cfg = config.get('output', {})
    align_cfg = config.get('align_subject_based_ground_truth', {})
    eval_cfg = config.get('check_human_labels_gt', {})

    # Extract dataset name and setup output directory from GT file path
    gt_path = Path(input_cfg['gt_file'])
    data = gt_path.parent.name.lower()  # TAB -> tab, PANORAMA -> panorama

    # Check if LLM file contains anonymized_text (auto-detect anon mode)
    is_anonymized = False
    try:
        with open(input_cfg['llm_file'], 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                first_record = json.loads(first_line)
                is_anonymized = 'anonymized_text' in first_record
    except Exception:
        pass  # Default to non-anonymized if can't read

    # Setup output directory based on anonymization status
    if is_anonymized:
        output_dir = gt_path.parent / "gt_vs_llms_anonymized"
    else:
        output_dir = gt_path.parent / "gt_vs_llms"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = output_cfg.get('suffix', '')

    # Generate concise base filename
    if suffix:
        base_name = f"{data}_gt_vs_llm_{suffix}"
    else:
        base_name = f"{data}_gt_vs_llm"

    # Define intermediate file paths
    combined_file = output_dir / f"{base_name}_combined.jsonl"
    aligned_file = output_dir / f"{base_name}_aligned.jsonl"
    report_file = output_dir / f"{base_name}_evaluation.json"

    print(f"\n{'='*60}")
    print(f"GT Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Ground Truth: {input_cfg['gt_file']}")
    print(f"LLM Annotated: {input_cfg['llm_file']}")
    print(f"Output Directory: {output_dir}")
    print(f"Config: API={align_cfg.get('api_provider')}, Model={align_cfg.get('model')}")
    print(f"{'='*60}\n")

    # Step 1: Combine GT and LLM annotations
    print("\n[Step 1/3] Combining GT and LLM annotations...")
    run([
        sys.executable, "src/evaluate/gt_vs_llms/align_human_llm_annotations.py",
        "--gt_file", input_cfg['gt_file'],
        "--llm_file", input_cfg['llm_file'],
        "--output_file", str(combined_file)
    ])

    # Step 2: Align subjects
    print("\n[Step 2/3] Aligning subjects between GT and LLM...")
    align_cmd = [
        sys.executable, "src/evaluate/gt_vs_llms/align_subject_based_ground_truth.py",
        "--input_file", str(combined_file),
        "--output_file", str(aligned_file),
        "--api_provider", align_cfg.get('api_provider', 'ollama'),
        "--model", align_cfg.get('model', 'qwen3:32b'),
        "--processing_mode", align_cfg.get('processing_mode', 'concurrent'),
        "--num_workers", str(align_cfg.get('num_workers', 20))
    ]
    if align_cfg.get('document_limit'):
        align_cmd.extend(["--document_limit", str(align_cfg['document_limit'])])
    run(align_cmd)

    # Step 3: Evaluate PII-level agreement
    print("\n[Step 3/3] Evaluating PII-level agreement...")
    eval_cmd = [
        sys.executable, "src/evaluate/gt_vs_llms/check_human_labels_gt.py",
        "--input_file", str(aligned_file),
        "--output_file", str(report_file),
        "--api_provider", eval_cfg.get('api_provider', 'openai'),
        "--model", eval_cfg.get('model', 'gpt-4o-mini'),
        "--min_certainty", str(eval_cfg['min_certainty']),
        "--location_max_depth", str(eval_cfg['location_max_depth']),
        "--decider", eval_cfg['decider'],
        "--num_workers", str(eval_cfg['num_workers']),
        "--case_log_filter", eval_cfg['case_log_filter']
    ]

    # Add optional parameters
    if eval_cfg.get('tags_to_evaluate'):
        eval_cmd.extend(["--tags"] + eval_cfg['tags_to_evaluate'])
    if not eval_cfg.get('show_case_logs', True):
        eval_cmd.append("--no_case_logs")

    run(eval_cmd)

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {report_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM annotations against ground truth using YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended)
  python src/evaluate/gt_vs_llms/evaluate_gt.py --config config/evaluate_gt_config.yaml

  # Override specific config values
  python src/evaluate/gt_vs_llms/evaluate_gt.py \\
    --config config/evaluate_gt_config.yaml \\
    --gt_file data/spia/panorama_151_human_gt.jsonl \\
    --llm_file data/spia/inference/panorama_151_inferred.jsonl

Configuration:
  All parameters are defined in the YAML config file.
  See config/evaluate_gt_config.yaml for full documentation.

  Config sections:
    - input: GT and LLM file paths
    - align_subject_based_ground_truth: API provider, model, processing settings
    - check_human_labels_gt: Certainty, tags, decider, workers, logging

Output: data/spia/gt_vs_llms/{gt}_vs_{llm}_evaluation.json
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluate_gt_config.yaml",
        help="Path to YAML config file (default: config/evaluate_gt_config.yaml)"
    )

    # Override options (optional - override config values)
    parser.add_argument("--gt_file", type=str, help="Override GT file path from config")
    parser.add_argument("--llm_file", type=str, help="Override LLM file path from config")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print(f"Please create a config file or use the default: config/evaluate_gt_config.yaml")
        sys.exit(1)

    config = load_config(args.config)

    # Apply command-line overrides
    if args.gt_file:
        config['input']['gt_file'] = args.gt_file
    if args.llm_file:
        config['input']['llm_file'] = args.llm_file

    # Validate input files exist
    if not os.path.exists(config['input']['gt_file']):
        print(f"Error: GT file not found: {config['input']['gt_file']}")
        sys.exit(1)
    if not os.path.exists(config['input']['llm_file']):
        print(f"Error: LLM file not found: {config['input']['llm_file']}")
        sys.exit(1)

    evaluate_gt(config)