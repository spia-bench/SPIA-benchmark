"""
DeID-GPT Text Anonymization

Simple function to anonymize text using DeID-GPT technique.

Usage:
    python src/anonymize/anonymize_deid_gpt.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_deid_gpt.py --input data.jsonl --model gpt-4.1-mini --api_provider openai
"""

import argparse
from typing import Union, List
from src.baseline.DeID_GPT.anonymizer import DeIDGPTAnonymizer
from tqdm import tqdm
from datetime import datetime
import json
import sys
import os

from src.anonymize.utils import check_length_anomaly


def anonymize_deid_gpt(
    texts: Union[str, List[str]],
    api_provider: str = "openai",
    model: str = "gpt-4",
    temperature: float = 0.05,
    max_tokens: int = 4096,
    show_progress: bool = True
) -> Union[str, List[str]]:
    """
    Anonymize text(s) using DeID-GPT technique.

    Args:
        texts: Single text string or list of text strings
        api_provider: API provider ("openai", "anthropic", or "ollama") - API keys from config.settings
        model: Model name (e.g., "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo")
        temperature: Sampling temperature (default: 0.05 for consistency)
        max_tokens: Max tokens for response (default: 4096)
        show_progress: Show progress bar for batch processing (default: True)

    Returns:
        Anonymized text(s) - same type as input (str or List[str])

    Examples:
        # Single text
        >>> result = anonymize_deid_gpt(
        ...     "Record date: 2069-04-07\\nMr. Villegas is seen today.",
        ...     api_provider="openai"
        ... )

        # Batch processing (sequential with tqdm)
        >>> results = anonymize_deid_gpt(
        ...     ["Text 1", "Text 2", "Text 3"],
        ...     api_provider="openai",
        ...     model="gpt-4"
        ... )
    """
    # Initialize anonymizer (uses llm.client for API calls)
    anonymizer = DeIDGPTAnonymizer(
        api_provider=api_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Handle single text
    if isinstance(texts, str):
        return anonymizer.anonymize(texts)

    # Handle batch processing with progress bar
    results = []
    text_list = texts

    if show_progress:
        try:
            from tqdm import tqdm
            text_list = tqdm(texts, desc="Anonymizing")
        except ImportError:
            pass  # No progress bar if tqdm not installed

    for text in text_list:
        result = anonymizer.anonymize(text)
        results.append(result)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anonymize text using DeID-GPT technique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/anonymize/anonymize_deid_gpt.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_deid_gpt.py --input data.jsonl --model gpt-4.1-mini
    python src/anonymize/anonymize_deid_gpt.py --input data.jsonl --api_provider anthropic --model claude-sonnet-4
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
        choices=["openai", "anthropic", "ollama"],
        help="API provider (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4.1-mini",
        help="Model name (default: gpt-4.1-mini)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    from src.llm.client import set_api_provider

    args = parse_args()

    input_file = args.input
    output_file = args.output
    api_provider = args.api_provider
    model = args.model

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        json_list = [json.loads(line) for line in f]

    print(f"Loaded {len(json_list)} documents from {input_file}")

    # Run Anonymization
    set_api_provider(api_provider, model)

    print("=" * 60)
    print("DeID-GPT Batch Anonymization")
    print("=" * 60)
    print(f"\nProcessing {len(json_list)} texts with {model}...\n")

    results = []
    errors = []
    for json_data in tqdm(json_list):
        data_id = json_data.get('metadata', {}).get('data_id', 'unknown')
        # Uses baseline defaults: temperature=0.05, max_tokens=4096
        result = anonymize_deid_gpt(
            texts=json_data['text'],  # Single text
            api_provider=api_provider,
            model=model,
            show_progress=False  # Disable tqdm since we're showing custom progress
        )

        # Check if anonymization returned empty string (indicates error)
        if not result:
            errors.append({
                'data_id': data_id,
                'error': 'Anonymization returned empty result'
            })
            continue

        # Create output format with length anomaly check
        length_info = check_length_anomaly(json_data['text'], result)
        output_data = {
            "metadata": json_data['metadata'],
            "text": json_data['text'],
            "anonymized_text": result,
            **length_info
        }
        results.append(output_data)

    # Save results
    if output_file is None:
        file_name = os.path.basename(input_file)
        base_name = os.path.splitext(file_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / with _)
        model_suffix = model.replace("/", "_") if model else "unknown"

        # Determine output directory based on input path
        if "panorama" in input_file.lower():
            output_dir = "data/spia/anonymized/panorama"
        elif "tab" in input_file.lower():
            output_dir = "data/spia/anonymized/tab"
        else:
            output_dir = os.path.dirname(input_file)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_deid_gpt_{model_suffix}_{timestamp}.jsonl")

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


if __name__ == "__main__":
    main()
