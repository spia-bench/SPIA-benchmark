"""
Adversarial Anonymization

Multi-round adversarial anonymization technique for enhanced privacy protection.

Usage:
    python src/anonymize/anonymize_adversarial.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_adversarial.py --input data/spia/panorama_151.jsonl --model gpt-4.1-mini --rounds 3
"""

import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime
import json
import sys
import os

from src.anonymize.utils import check_length_anomaly


# ============================================================
# Dynamic Anonymizer Selection
# ============================================================

def detect_dataset_type(input_file: str) -> str:
    """Detect dataset type from input file path or filename."""
    input_lower = input_file.lower()
    if "panorama" in input_lower:
        return "PANORAMA"
    elif "tab" in input_lower:
        return "TAB"
    else:
        raise ValueError(f"Cannot detect dataset type from path: {input_file}. "
                        "Path or filename must contain 'tab' or 'panorama'.")


def load_anonymizer_module(dataset_type: str):
    """
    Dynamically load the appropriate anonymizer module based on dataset type.
    Returns (AdversarialAnonymizer class, DEFAULT_TARGET_ATTRIBUTES list)
    """
    if dataset_type == "TAB":
        from src.baseline.adversarial_anonymizer.anonymizer_tab import (
            AdversarialAnonymizer,
            DEFAULT_TARGET_ATTRIBUTES
        )
        print(f"Loaded TAB anonymizer: anonymizer_tab.py")
    elif dataset_type == "PANORAMA":
        from src.baseline.adversarial_anonymizer.anonymizer_panorama import (
            AdversarialAnonymizer,
            DEFAULT_TARGET_ATTRIBUTES
        )
        print(f"Loaded PANORAMA anonymizer: anonymizer_panorama.py")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return AdversarialAnonymizer, DEFAULT_TARGET_ATTRIBUTES


# ============================================================
# Core Anonymization Function
# ============================================================

def anonymize_adversarial_anonymization(
    text: str,
    anonymizer_class,
    default_attributes: List[str],
    target_attributes: Optional[List[str]] = None,
    num_rounds: int = 3,
    api_provider: str = "openai",
    anonymizer_model: str = "gpt-4",
    inference_model: str = "gpt-4",
    prompt_level: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform multi-round adversarial anonymization on text.

    This method iteratively anonymizes text by having an anonymizer model
    rewrite the text to protect attributes, while an inference model attempts
    to extract those attributes. The process repeats for multiple rounds to
    strengthen anonymization.

    Args:
        text: Text to anonymize
        anonymizer_class: AdversarialAnonymizer class (loaded dynamically based on dataset)
        default_attributes: Default target attributes from the loaded anonymizer module
        target_attributes: List of attributes to protect. If None, uses default_attributes
        num_rounds: Number of adversarial rounds to perform (default: 3)
        api_provider: API provider ("openai", "anthropic", or "ollama") - API keys from config.settings
        anonymizer_model: Model for anonymization (default: "gpt-4")
        inference_model: Model for attribute inference (default: "gpt-4")
        prompt_level: Prompting strategy level (1-3, default: 3 for Chain-of-Thought)
        verbose: Print progress information (default: False)

    Returns:
        Dictionary containing:
            - original_text: Original input text
            - anonymized_text: Final anonymized text after all rounds
            - rounds: List of round-by-round results
            - target_attributes: Attributes that were protected
            - num_rounds: Number of rounds performed

    Examples:
        >>> # Load anonymizer dynamically
        >>> AnonymizerClass, DEFAULT_ATTRS = load_anonymizer_module("TAB")
        >>> result = anonymize_adversarial_anonymization(
        ...     text="I'm a 28-year-old software engineer in San Francisco.",
        ...     anonymizer_class=AnonymizerClass,
        ...     default_attributes=DEFAULT_ATTRS,
        ...     target_attributes=["age", "occupation", "location"],
        ...     num_rounds=3,
        ...     api_provider="openai"
        ... )
        >>> print(result['anonymized_text'])
        "I'm a professional working in a major city."
    """
    # Use default target attributes if not provided
    if target_attributes is None:
        target_attributes = default_attributes

    # Initialize the anonymizer (uses llm.client for API calls)
    anonymizer = anonymizer_class(
        api_provider=api_provider,
        anonymizer_model=anonymizer_model,
        inference_model=inference_model,
        prompt_level=prompt_level
    )

    # Perform multi-round adversarial anonymization
    result = anonymizer.adversarial_anonymize(
        text=text,
        target_attributes=target_attributes,
        num_rounds=num_rounds,
        verbose=verbose
    )

    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-round adversarial anonymization for enhanced privacy protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/anonymize/anonymize_adversarial.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_adversarial.py --input data/spia/panorama_151.jsonl --model gpt-4.1-mini
    python src/anonymize/anonymize_adversarial.py --input data.jsonl --rounds 5 --api_provider anthropic
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
        help="Model for anonymization and inference (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Number of adversarial rounds (default: 3)"
    )
    parser.add_argument(
        "--prompt_level",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Prompting strategy level: 1=naive, 2=better, 3=Chain-of-Thought (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed round-by-round output"
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
    num_rounds = args.rounds
    prompt_level = args.prompt_level
    verbose = args.verbose

    print("Adversarial Anonymization - Batch Processing")
    print("=" * 60)

    # Detect dataset type and load appropriate anonymizer
    dataset_type = detect_dataset_type(input_file)
    print(f"Detected dataset type: {dataset_type}")
    AdversarialAnonymizer, DEFAULT_TARGET_ATTRIBUTES = load_anonymizer_module(dataset_type)

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        json_list = [json.loads(line) for line in f]

    print(f"Loaded {len(json_list)} documents from {input_file}")

    # Run Batch Anonymization
    set_api_provider(api_provider, model)

    print(f"\nProcessing {len(json_list)} texts with {model}...")
    print(f"Rounds: {num_rounds} | Protecting: {len(DEFAULT_TARGET_ATTRIBUTES)} attributes\n")

    results = []
    errors = []
    for json_data in tqdm(json_list):
        data_id = json_data.get('metadata', {}).get('data_id', 'unknown')
        result = anonymize_adversarial_anonymization(
            text=json_data['text'],
            anonymizer_class=AdversarialAnonymizer,
            default_attributes=DEFAULT_TARGET_ATTRIBUTES,
            num_rounds=num_rounds,
            api_provider=api_provider,
            anonymizer_model=model,
            inference_model=model,
            prompt_level=prompt_level,
            verbose=verbose
        )

        # Check if anonymization returned empty string (indicates error)
        anonymized_text = result.get('anonymized_text', '')
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
        model_suffix = model.replace("/", "_") if model else "unknown"

        # Determine output directory based on input path
        if "panorama" in input_file.lower():
            output_dir = "data/spia/anonymized/panorama"
        elif "tab" in input_file.lower():
            output_dir = "data/spia/anonymized/tab"
        else:
            output_dir = os.path.dirname(input_file)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_adversarial_{model_suffix}_{timestamp}.jsonl")

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

