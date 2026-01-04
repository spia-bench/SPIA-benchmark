"""
Utility Evaluation Script

Evaluates the utility of anonymized text using BLEU, ROUGE, and LLM-based metrics.
Reads text pairs from JSONL files with 'text' and 'anonymized_text' fields.

Output:
    - Individual results: {input_basename}_utility.jsonl (JSONL with per-item scores)
    - Summary report: {input_basename}_utility_report.json (JSON with config + aggregates)
"""

import json
import argparse
import sys
import os
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluate.utility.utility_evaluator import UtilityEvaluator


# =============================================================================
# Filename Utilities (inlined from filename_utils.py)
# =============================================================================

def normalize_model_name(model: str) -> str:
    """
    Normalize model name for consistent filenames.
    Replaces special characters (: . /) with hyphens.
    """
    return model.replace(":", "-").replace(".", "-").replace("/", "-")


def parse_anonymized_filename(filepath: str) -> Dict[str, str]:
    """
    Parse anonymized filename to extract baseline, model, and dataset info.

    Handles various filename patterns:
    - panorama_151_adversarial_gemma3:27b_20251210_193858.jsonl
    - tab_144_deid_gpt_gpt-4.1-mini_20251211_162624.jsonl
    - tab_144_adversarial_claude-sonnet-4-5_20251214_024049.jsonl

    Returns:
        Dict with keys: dataset, dataset_size, baseline, model
    """
    filename = Path(filepath).stem
    name = filename.lower()

    info = {
        "dataset": "unknown",
        "dataset_size": "unknown",
        "baseline": "unknown",
        "model": "unknown"
    }

    # Detect dataset and set size
    if "panorama" in name:
        info["dataset"] = "panorama"
        info["dataset_size"] = "151"
    elif "tab" in name:
        info["dataset"] = "tab"
        info["dataset_size"] = "144"

    # Parse baseline
    if "adversarial" in name:
        info["baseline"] = "adversarial"
    elif "deid_gpt" in name or "deid-gpt" in name:
        info["baseline"] = "deid_gpt"
    elif "dp_prompt" in name or "dp-prompt" in name:
        info["baseline"] = "dp_prompt"
    elif "longformer" in name:
        info["baseline"] = "longformer"

    # Parse model (order matters - check specific patterns first)
    # Claude models (check longer patterns first)
    if "claude-sonnet-4-5" in name or "claude_sonnet_4_5" in name:
        info["model"] = "claude-sonnet-4-5"
    elif "claude-haiku-4-5" in name or "claude_haiku_4_5" in name:
        info["model"] = "claude-haiku-4-5"
    elif "claude-sonnet" in name or "claude_sonnet" in name:
        info["model"] = "claude-sonnet-4-5"
    elif "claude-3-5-haiku" in name or "claude_haiku" in name or "claude-haiku" in name:
        info["model"] = "claude-haiku-4-5"
    # GPT models (check mini before base)
    elif "gpt-4.1-mini" in name or "gpt-4-1-mini" in name or "gpt4-mini" in name or "gpt4_mini" in name:
        info["model"] = "gpt-4-1-mini"
    elif "gpt-4.1" in name or "gpt-4-1" in name or "gpt4" in name:
        info["model"] = "gpt-4-1"
    # Open source models
    elif "gemma3:27b" in name or "gemma3-27b" in name or "gemma3_27b" in name:
        info["model"] = "gemma3-27b"
    elif "llama3.1:8b" in name or "llama3-1-8b" in name or "llama3_8b" in name or "llama3-8b" in name:
        info["model"] = "llama3-1-8b"
    elif "stablelm" in name:
        info["model"] = "stablelm-7b"
    elif "flan-t5-xl" in name or "flan_t5_xl" in name:
        info["model"] = "flan-t5-xl"
    # longformer is both baseline and model (no separate LLM model)
    elif "longformer" in name:
        info["model"] = "longformer"

    return info


def generate_output_filename(
    eval_type: str,
    dataset: str,
    dataset_size: str,
    baseline: str,
    model: str,
    timestamp: str = None,
    extension: str = ".json"
) -> str:
    """
    Generate standardized output filename.

    Args:
        eval_type: Type of evaluation (ppr, recall, utility)
        dataset: Dataset name (panorama, tab)
        dataset_size: Dataset size (151, 144)
        baseline: Anonymization baseline (adversarial, deid_gpt, dp_prompt)
        model: Model name (already normalized)
        timestamp: Timestamp string (default: current time)
        extension: File extension (default: .json)

    Returns:
        Standardized filename string
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{eval_type}_{dataset}_{dataset_size}_{baseline}_{model}_{timestamp}{extension}"


def get_output_directory(dataset: str, base_path: str = "data/SPIA") -> Path:
    """
    Get output directory path based on dataset type.

    Args:
        dataset: Dataset name (panorama, tab, PANORAMA, TAB)
        base_path: Base path for data directory

    Returns:
        Path to baseline_evaluation directory
    """
    dataset_upper = dataset.upper()
    return Path(base_path) / dataset_upper / "baseline_evaluation"


# =============================================================================
# End of Filename Utilities
# =============================================================================

# Thread-safe lock for file writing
file_lock = threading.Lock()


def evaluate_single_item(
    item: Dict[str, Any],
    idx: int,
    evaluator: UtilityEvaluator
) -> Dict[str, Any]:
    """
    Evaluate a single item (for concurrent processing).

    Args:
        item: Data item with 'text' and 'anonymized_text' fields
        idx: Item index
        evaluator: UtilityEvaluator instance

    Returns:
        Dict with status and result/error
    """
    data_id = item.get('metadata', {}).get('data_id', f'item_{idx}')
    original = item.get('text', '')
    anonymized = item.get('anonymized_text', '')

    if not original or not anonymized:
        return {
            'data_id': data_id,
            'status': 'error',
            'error': "Missing 'text' or 'anonymized_text' field",
            'item': item,
            'idx': idx
        }

    try:
        scores = evaluator.evaluate(original, anonymized)
        result = {
            **item,
            "utility_scores": scores
        }
        return {
            'data_id': data_id,
            'status': 'success',
            'result': result,
            'idx': idx
        }
    except Exception as e:
        return {
            'data_id': data_id,
            'status': 'error',
            'error': str(e),
            'item': item,
            'idx': idx
        }


def save_result_concurrent(output_file: str, result: Dict[str, Any]):
    """Thread-safe save of result to file."""
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def evaluate_from_file_concurrent(
    input_file: str,
    output_file: str = None,
    use_llm: bool = False,
    api_provider: str = "openai",
    model: str = "gpt-4",
    verbose: bool = False,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Evaluate utility for text pairs from a JSONL file using concurrent processing.

    Args:
        input_file: Path to input JSONL file
        output_file: Optional path to save results
        use_llm: Whether to use LLM-based evaluation
        api_provider: API provider for LLM evaluation
        model: Model for LLM evaluation
        verbose: Print detailed output
        max_workers: Number of concurrent workers

    Returns:
        List of evaluation results
    """
    evaluator = UtilityEvaluator(
        api_provider=api_provider,
        model=model,
        use_llm_evaluation=use_llm
    )

    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} text pairs from {input_file}")
    print(f"LLM Evaluation: {'Enabled' if use_llm else 'Disabled'}")
    if use_llm:
        print(f"API Provider: {api_provider}, Model: {model}")
        print(f"Processing mode: CONCURRENT with {max_workers} workers")
    print()

    # Prepare items for processing
    items_to_process = [(item, idx) for idx, item in enumerate(data)]

    results = []
    success_count = 0
    error_count = 0

    # Clear output file if exists (we'll write incrementally)
    if output_file and os.path.exists(output_file):
        os.remove(output_file)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(evaluate_single_item, item, idx, evaluator): (item, idx)
            for item, idx in items_to_process
        }

        for future in tqdm(as_completed(future_to_item),
                          total=len(items_to_process),
                          desc="Evaluating utility"):
            try:
                result = future.result()
            except Exception as e:
                item, idx = future_to_item[future]
                data_id = item.get('metadata', {}).get('data_id', f'item_{idx}')
                result = {
                    'data_id': data_id,
                    'status': 'error',
                    'error': str(e),
                    'item': item,
                    'idx': idx
                }

            if result['status'] == 'success':
                results.append(result['result'])
                success_count += 1

                # Save incrementally
                if output_file:
                    save_result_concurrent(output_file, result['result'])

                if verbose:
                    scores = result['result']['utility_scores']
                    print(f"  ✅ {result['data_id']}: BLEU={scores['utility_bleu']:.4f}, ROUGE-L={scores['utility_rouge']:.4f}")
            else:
                error_count += 1
                print(f"  ⚠️ {result['data_id']}: {result['error']}")

    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed:.1f}s")
    print(f"  Success: {success_count}, Errors: {error_count}")
    if success_count > 0:
        print(f"  Avg time per item: {elapsed/success_count:.2f}s")

    if output_file:
        print(f"Results saved to {output_file}")

    return results


def evaluate_from_file(
    input_file: str,
    output_file: str = None,
    use_llm: bool = False,
    api_provider: str = "openai",
    model: str = "gpt-4",
    verbose: bool = False,
    max_workers: int = 1
) -> List[Dict[str, Any]]:
    """
    Evaluate utility for text pairs from a JSONL file.

    Expected input format (JSONL):
        {"text": "original text", "anonymized_text": "anonymized text", ...}

    Args:
        input_file: Path to input JSONL file
        output_file: Optional path to save results
        use_llm: Whether to use LLM-based evaluation
        api_provider: API provider for LLM evaluation
        model: Model for LLM evaluation
        verbose: Print detailed output
        max_workers: Number of concurrent workers (1 = sequential)

    Returns:
        List of evaluation results
    """
    # Use concurrent processing if max_workers > 1
    if max_workers > 1:
        return evaluate_from_file_concurrent(
            input_file=input_file,
            output_file=output_file,
            use_llm=use_llm,
            api_provider=api_provider,
            model=model,
            verbose=verbose,
            max_workers=max_workers
        )

    # Sequential processing (original logic)
    evaluator = UtilityEvaluator(
        api_provider=api_provider,
        model=model,
        use_llm_evaluation=use_llm
    )

    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} text pairs from {input_file}")
    print(f"LLM Evaluation: {'Enabled' if use_llm else 'Disabled'}")
    if use_llm:
        print(f"API Provider: {api_provider}, Model: {model}")
    print()

    results = []
    for item in tqdm(data, desc="Evaluating utility"):
        original = item.get('text', '')
        anonymized = item.get('anonymized_text', '')

        if not original or not anonymized:
            print(f"Warning: Skipping item with missing 'text' or 'anonymized_text' field")
            continue

        # Evaluate
        scores = evaluator.evaluate(original, anonymized)

        # Combine with original data
        result = {
            **item,
            "utility_scores": scores
        }
        results.append(result)

        if verbose:
            print(f"\n--- Sample {len(results)} ---")
            print(f"Original: {original[:100]}...")
            print(f"Anonymized: {anonymized[:100]}...")
            print(f"BLEU: {scores['utility_bleu']:.4f}, ROUGE-L: {scores['utility_rouge']:.4f}")

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nResults saved to {output_file}")

    return results


def compute_aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate statistics across multiple evaluations."""
    if not results:
        return {}

    # Collect all scores
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    readability_scores = []
    meaning_scores = []
    hallucination_scores = []
    utility_comb_scores = []

    for r in results:
        scores = r.get('utility_scores', r)
        bleu_scores.append(scores.get('utility_bleu', 0))
        rouge1_scores.append(scores.get('rouge', {}).get('rouge1', 0))
        rougeL_scores.append(scores.get('rouge', {}).get('rougeL', 0))
        utility_comb_scores.append(scores.get('utility_comb', 0))

        if 'utility_readability' in scores:
            readability_scores.append(scores['utility_readability'])
            meaning_scores.append(scores['utility_meaning'])
            hallucination_scores.append(scores['utility_hallucination'])

    n = len(results)
    aggregates = {
        "count": n,
        "avg_bleu": sum(bleu_scores) / n,
        "avg_rouge1": sum(rouge1_scores) / n,
        "avg_rougeL": sum(rougeL_scores) / n,
        "avg_utility_comb": sum(utility_comb_scores) / n,
    }

    if readability_scores:
        aggregates.update({
            "avg_readability": sum(readability_scores) / len(readability_scores),
            "avg_meaning": sum(meaning_scores) / len(meaning_scores),
            "avg_hallucination": sum(hallucination_scores) / len(hallucination_scores),
        })

    return aggregates


def print_aggregate_scores(aggregates: Dict[str, float]):
    """Print aggregate scores."""
    print("\n" + "=" * 50)
    print(f"Aggregate Scores (n={aggregates['count']})")
    print("=" * 50)
    print(f"  Avg BLEU:       {aggregates['avg_bleu']:.4f}")
    print(f"  Avg ROUGE-1:    {aggregates['avg_rouge1']:.4f}")
    print(f"  Avg ROUGE-L:    {aggregates['avg_rougeL']:.4f}")

    if 'avg_readability' in aggregates:
        print(f"  Avg Readability: {aggregates['avg_readability']:.2f}/10")
        print(f"  Avg Meaning:     {aggregates['avg_meaning']:.2f}/10")
        print(f"  Avg Hallucination: {aggregates['avg_hallucination']:.2f}")

    print(f"  Avg Combined:   {aggregates['avg_utility_comb']:.4f}")
    print("=" * 50)


def main():
    # ============================================================
    # Default Configuration - Modify these values as needed
    # ============================================================
    INPUT_FILE = "data/anonymized_output.jsonl"  # Input JSONL file path
    OUTPUT_FILE = None  # Output file path (None = {input_basename}_utility.jsonl, same directory as input)
    USE_LLM = False  # Use LLM-based evaluation (requires API key)
    API_PROVIDER = "openai"  # "openai", "anthropic", or "ollama"
    MODEL = "gpt-4.1-mini"  # Model for LLM evaluation
    VERBOSE = False  # Print detailed output per sample
    MAX_WORKERS = 10  # Number of concurrent workers (1 = sequential)
    # ============================================================

    parser = argparse.ArgumentParser(
        description="Evaluate utility of anonymized text from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (configured in main function)
  python evaluate_utility.py

  # Specify input file
  python evaluate_utility.py --input data.jsonl

  # With LLM-based evaluation
  python evaluate_utility.py --input data.jsonl --use-llm --api-provider openai

  # With concurrent processing (10 workers)
  python evaluate_utility.py --input data.jsonl --use-llm --workers 10

  # Save results to file
  python evaluate_utility.py --input data.jsonl --output results.jsonl

Input JSONL format:
  {"text": "original text", "anonymized_text": "anonymized text", ...}

Output files (when OUTPUT_FILE=None, auto-generated in same directory as input):
  - {input_basename}_utility.jsonl: Individual evaluation results per item
  - {input_basename}_utility_report.json: Summary report with config and aggregates
        """
    )

    parser.add_argument('--input', '-i', type=str, default=INPUT_FILE,
                        help=f'Input JSONL file (default: {INPUT_FILE})')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_FILE,
                        help='Output JSONL file (default: {input_basename}_utility.jsonl)')
    parser.add_argument('--use-llm', action='store_true', default=USE_LLM,
                        help='Use LLM-based evaluation (requires API key)')
    parser.add_argument('--api-provider', type=str, default=API_PROVIDER,
                        choices=['openai', 'anthropic', 'ollama'],
                        help=f'API provider for LLM evaluation (default: {API_PROVIDER})')
    parser.add_argument('--model', type=str, default=MODEL,
                        help=f'Model for LLM evaluation (default: {MODEL})')
    parser.add_argument('--verbose', '-v', action='store_true', default=VERBOSE,
                        help='Show detailed output per sample')
    parser.add_argument('--workers', '-w', type=int, default=MAX_WORKERS,
                        help=f'Number of concurrent workers (default: {MAX_WORKERS}, 1=sequential)')

    args = parser.parse_args()

    # Auto-generate output file name if not specified
    # Output to baseline_evaluation directory with standardized naming
    output_file = args.output
    if output_file is None:
        input_path = args.input

        # Parse input filename for components
        file_info = parse_anonymized_filename(input_path)

        # Get output directory
        output_dir = get_output_directory(file_info["dataset"])
        os.makedirs(output_dir, exist_ok=True)

        # Generate standardized filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = generate_output_filename(
            eval_type="utility",
            dataset=file_info["dataset"],
            dataset_size=file_info["dataset_size"],
            baseline=file_info["baseline"],
            model=file_info["model"],
            timestamp=timestamp,
            extension=".jsonl"
        )
        output_file = os.path.join(output_dir, output_filename)

    # Report file path (same base as output, but _report.json extension)
    report_file = output_file.replace(".jsonl", "_report.json")

    print("=" * 60)
    print("Utility Evaluation")
    print("=" * 60)
    print(f"Input file:  {args.input}")
    print(f"Output file: {output_file}")
    print(f"Report file: {report_file}")
    print(f"Workers:     {args.workers} ({'concurrent' if args.workers > 1 else 'sequential'})")
    print("=" * 60)

    # Build configuration dict
    config = {
        "input_file": args.input,
        "output_file": output_file,
        "use_llm": args.use_llm,
        "api_provider": args.api_provider,
        "model": args.model,
        "verbose": args.verbose,
        "max_workers": args.workers,
        "timestamp": datetime.now().isoformat()
    }

    # Run evaluation
    results = evaluate_from_file(
        input_file=args.input,
        output_file=output_file,
        use_llm=args.use_llm,
        api_provider=args.api_provider,
        model=args.model,
        verbose=args.verbose,
        max_workers=args.workers
    )

    # Compute and print aggregate scores
    if results:
        aggregates = compute_aggregate_scores(results)
        print_aggregate_scores(aggregates)

        # Save summary report as JSON
        report = {
            "config": config,
            "aggregates": aggregates
        }
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {report_file}")
    else:
        print("No results to aggregate.")

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
