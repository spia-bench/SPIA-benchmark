"""
DP-Prompt Text Anonymization

Simple function to anonymize text using DP-Prompt technique.

Usage:
    python src/anonymize/anonymize_dp_prompt.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_dp_prompt.py --input data.jsonl --model gpt-4.1-mini --temperature 1.5
"""

import argparse
from typing import List, Union
from src.baseline.dp_prompt.anonymizer import DPPromptAnonymizer
from tqdm import tqdm
from datetime import datetime
import json
import sys
import os

from src.anonymize.utils import check_length_anomaly


def anonymize_dp_prompt(
    input_file: str,
    output_file: str = None,
    api_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 1.0
):
    """
    Anonymize texts from JSONL file using DP-Prompt technique.

    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path (None for auto-generated)
        api_provider: API provider ("openai", "anthropic", or "ollama")
        model_name: Model name (e.g., "gpt-3.5-turbo", "llama2")
        temperature: Sampling temperature (default: 1.0, recommended: 1.0-2.0)
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        json_list = [json.loads(line) for line in f]

    print(f"Loaded {len(json_list)} documents from {input_file}")
    print("=" * 60)
    print("DP-Prompt Batch Anonymization")
    print("=" * 60)
    print(f"\nProcessing {len(json_list)} texts with {model_name}...\n")

    # Initialize anonymizer once for efficiency
    anonymizer = DPPromptAnonymizer(
        api_provider=api_provider,
        model_name=model_name,
        temperature=temperature
    )

    results = []
    errors = []
    for json_data in tqdm(json_list):
        data_id = json_data.get('metadata', {}).get('data_id', 'unknown')
        anonymized_text = anonymizer.anonymize(json_data['text'])

        # Check if anonymization returned empty string (indicates error)
        if not anonymized_text:
            errors.append({
                'data_id': data_id,
                'error': 'Anonymization returned empty result'
            })
            continue

        # Create output format with length anomaly check
        length_info = check_length_anomaly(json_data['text'], anonymized_text)
        output_data = {
            "metadata": json_data['metadata'],
            "text": json_data['text'],
            "anonymized_text": anonymized_text,
            **length_info
        }
        results.append(output_data)

    # Save results
    if output_file is None:
        file_name = os.path.basename(input_file)
        base_name = os.path.splitext(file_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / with _)
        model_suffix = model_name.replace("/", "_") if model_name else "unknown"

        # Determine output directory based on input path
        if "panorama" in input_file.lower():
            output_dir = "data/spia/anonymized/panorama"
        elif "tab" in input_file.lower():
            output_dir = "data/spia/anonymized/tab"
        else:
            output_dir = os.path.dirname(input_file)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_dp_prompt_{model_suffix}_{timestamp}.jsonl")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nResults saved to '{output_file}'")
        print(f"Success: {len(results)}, Errors: {len(errors)}")
        if errors:
            error_file = os.path.splitext(output_file)[0] + "_errors.jsonl"
            with open(error_file, 'w', encoding='utf-8') as f:
                for err in errors:
                    f.write(json.dumps(err, ensure_ascii=False) + '\n')
            print(f"Errors saved to '{error_file}'")
        print(f"Processing completed!")
    except Exception as e:
        print(f"Error saving output file '{output_file}': {e}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anonymize text using DP-Prompt technique (zero-shot paraphrasing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/anonymize/anonymize_dp_prompt.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_dp_prompt.py --input data.jsonl --model gpt-4.1-mini
    python src/anonymize/anonymize_dp_prompt.py --input data.jsonl --temperature 1.5
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated)"
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "ollama", "local"],
        help="API provider (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4.1-mini",
        help="Model name (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, recommended: 1.0-2.0)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Run anonymization
    anonymize_dp_prompt(
        input_file=args.input,
        output_file=args.output,
        api_provider=args.api_provider,
        model_name=args.model,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()

