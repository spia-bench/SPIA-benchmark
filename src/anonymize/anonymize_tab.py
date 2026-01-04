"""
TAB Anonymization

Text anonymization using Longformer model trained on TAB dataset.

Usage:
    python src/anonymize/anonymize_tab.py --input data/spia/tab_144.jsonl --model src/baseline/TAB/longformer_experiments/long_model_our.pt
"""

import argparse
import torch
from src.baseline.TAB.longformer_experiments.longformer_model import Model
from src.baseline.TAB.longformer_experiments.data_handling import LabelSet
from transformers import LongformerTokenizerFast
from datetime import datetime
import sys
import os
from src.baseline.TAB.byte_offset_utils import correct_offset_mapping, correct_offset_mapping_utf16, convert_utf16_offset_to_char_range
import json
from tqdm import tqdm

# spaCy and Annotator imports
try:
    import spacy
    from src.baseline.TAB.scripts.annotate import Annotator
except ImportError:
    spacy = None
    Annotator = None

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer.

    Args:
        model_path: Path to fine-tuned model weights file (.pt)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert = "allenai/longformer-base-4096"

    # Initialize tokenizer and label set
    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set = LabelSet(labels=["MASK"])

    # Create model object
    model = Model(model=bert, num_labels=len(label_set.ids_to_label.values()))
    model = model.to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Set evaluation mode
    model.eval()

    return model, tokenizer, device

def tokenize_and_predict(text, model, tokenizer, device):
    """Tokenize text and perform prediction."""
    # Tokenize text (same as training)
    tokens = tokenizer(
        text,
        max_length=4096,
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    # Convert to batch format
    batch = {
        "input_ids": torch.tensor([tokens["input_ids"]]).to(device),
        "attention_masks": torch.tensor([tokens["attention_mask"]]).to(device)
    }

    # Run inference
    with torch.no_grad():
        output = model(batch)
        # Adjust output dimensions (same as training)
        output = output.permute(0, 2, 1)
        predictions = output.argmax(dim=1).cpu().numpy()[0]

    # Calculate confidence scores
    confidence_scores = torch.softmax(output, dim=1).max(dim=1)[0].cpu().numpy()[0]

    return tokens, predictions, confidence_scores

def infer_entity_types(text, masked_spans, use_annotator=True):
    """Infer actual entity types for masked spans."""
    if not masked_spans:
        return masked_spans

    if use_annotator and Annotator is not None:
        # Use project's Annotator (recommended)
        annotator = Annotator(spacy_model="en_core_web_md")
        i = 1
        for span in masked_spans:
            span["entity_type"] = annotator.annotate_improved(span["span_text"])
            if span["entity_type"] == "PERSON" or span["entity_type"] == "CODE":
                span["identifier_type"] = "direct"
            else:
                span["identifier_type"] = "quasi"
            span["span_id"] = f"{i}"
            span["entity_id"] = f"{i}"
            i += 1
        return masked_spans
    elif spacy is not None:
        # Use default spaCy NER
        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        ent_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    else:
        raise RuntimeError(
            "Annotator is not available. Please install spaCy model:\n"
            "  python -m spacy download en_core_web_md"
        )

    # IoU-based matching to assign entity types
    for span in masked_spans:
        ss, se = span["start_offset"], span["end_offset"]
        best_label = "MISC"
        best_iou = 0.0

        for es, ee, lab in ent_spans:
            # Calculate overlap
            inter = max(0, min(se, ee) - max(ss, es))
            if inter == 0:
                continue
            union = (se - ss) + (ee - es) - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_label = lab if isinstance(lab, str) else str(lab)

        # Only change type when IoU >= 0.3
        if best_iou >= 0.0:
            span["entity_type"] = best_label
        else:
            span["entity_type"] = "MISC"

    return masked_spans

def find_masked_spans_tab_format(predictions, offset_mapping, confidence_scores, text, doc_id="DOC1", use_utf16_code_units=False):
    """Find masked spans in TAB dataset format."""
    # Correct offset_mapping to selected standard
    if use_utf16_code_units:
        # Windows Notepad standard (UTF-16 code units)
        offset_mapping = correct_offset_mapping_utf16(text, offset_mapping)
    else:
        # UTF-8 byte standard (default)
        offset_mapping = correct_offset_mapping(text, offset_mapping)

    masked_spans = []
    current_span = None
    entity_counter = 1
    mention_counter = 1

    for i, (pred, offset, conf) in enumerate(zip(predictions, offset_mapping, confidence_scores)):
        # If prediction is MASK and token corresponds to actual text
        if pred > 0 and offset[0] is not None and offset[1] is not None:
            if current_span is None and conf >= 0.55:
                # Start new masked span
                current_span = {
                    "entity_type": "MISC",
                    "entity_mention_id": f"{doc_id}_em{mention_counter}",
                    "start_offset": offset[0],
                    "end_offset": offset[1],
                    "span_text": "",
                    "edit_type": "insert",
                    "confidential_status": "NOT_CONFIDENTIAL",
                    "identifier_type": "DIRECT",
                    "entity_id": f"{doc_id}_e{entity_counter}",
                    "label": pred,
                    "confidence": float(conf),
                    "tokens": [i],
                    "start_token": i,
                    "end_token": i
                }
            elif current_span is not None:
                # Extend existing span
                current_span["end_offset"] = offset[1]
                current_span["end_token"] = i
                current_span["confidence"] = max(current_span["confidence"], float(conf))
                current_span["tokens"].append(i)
        else:
            # End of masked span
            if current_span is not None:
                if conf >= 0.55 and predictions[i-1] > 0 and predictions[i+1 if i < len(predictions) - 1 else i] > 0:
                    # Extend existing span
                    current_span["end_offset"] = offset[1]
                    current_span["end_token"] = i
                    current_span["confidence"] = max(current_span["confidence"], float(conf))
                    current_span["tokens"].append(i)
                else:
                    # Convert UTF-16 code unit offset to character offset to extract span_text
                    if use_utf16_code_units:
                        char_start, char_end = convert_utf16_offset_to_char_range(
                            text, current_span["start_offset"], current_span["end_offset"]
                        )
                        current_span["span_text"] = text[char_start:char_end]
                    else:
                        current_span["span_text"] = text[current_span["start_offset"]:current_span["end_offset"]]

                    spans = []
                    # Split by |
                    if "|" in current_span["span_text"]:
                        span1 = current_span.copy()
                        span2 = current_span.copy()

                        span1["span_text"] = span1["span_text"].split("|")[0]
                        span1["end_offset"] = span1["start_offset"] + len(span1["span_text"])
                        spans.append(span1)

                        span2["span_text"] = span2["span_text"].split("|")[1]
                        span2["start_offset"] = span2["end_offset"] - len(span2["span_text"])
                        spans.append(span2)

                    elif "/" in current_span["span_text"]:
                        span1 = current_span.copy()
                        span2 = current_span.copy()

                        span1["span_text"] = span1["span_text"].split("/")[0]
                        span1["end_offset"] = span1["start_offset"] + len(span1["span_text"])
                        spans.append(span1)

                        span2["span_text"] = span2["span_text"].split("/")[1]
                        span2["start_offset"] = span2["end_offset"] - len(span2["span_text"])
                        spans.append(span2)

                    else:
                        spans.append(current_span)

                    for span in spans:
                        if span["entity_type"] == "CODE" and ":" in span["span_text"]:
                            span["span_text"] = span["span_text"].split(":")[1]
                            span["start_offset"] = span["end_offset"] - len(span["span_text"])

                        # Trim and adjust span after ending masked region
                        tm_text = span["span_text"].lstrip()
                        tm_text = tm_text.lstrip("'@.,;:!?\"()[]{}<>")
                        span["start_offset"] = span["start_offset"] + (len(span["span_text"]) - len(tm_text))
                        span["span_text"] = tm_text

                        tm_text = span["span_text"].rstrip()
                        tm_text = tm_text.rstrip("'@.,;:!?\"()[]{}<>")
                        span["end_offset"] = span["end_offset"] - (len(span["span_text"]) - len(tm_text))
                        span["span_text"] = tm_text

                        masked_spans.append(span)
                    current_span = None
                    entity_counter += 1
                    mention_counter += 1


    # Handle last span
    if current_span is not None:
        # Convert UTF-16 code unit offset to character offset to extract span_text
        if use_utf16_code_units:
            char_start, char_end = convert_utf16_offset_to_char_range(
                text, current_span["start_offset"], current_span["end_offset"]
            )
            current_span["span_text"] = text[char_start:char_end]
        else:
            current_span["span_text"] = text[current_span["start_offset"]:current_span["end_offset"]]
        masked_spans.append(current_span)

    return masked_spans

def process_entities(text, predictions, offset_mapping, confidence_scores, doc_id="TEST_DOC", use_utf16_code_units=True):
    """Run full entity processing pipeline."""
    # Find masked spans
    masked_spans = find_masked_spans_tab_format(
        predictions, offset_mapping, confidence_scores, text, doc_id=doc_id, use_utf16_code_units=use_utf16_code_units
    )

    # Infer entity types
    try:
        masked_spans = infer_entity_types(text, masked_spans, use_annotator=True)
    except Exception as e:
        print(f"⚠️  spaCy/Annotator not available: {e}")
        print("Using default 'MASK' entity types")

    return masked_spans

def create_tab_format_json(json_data, masked_spans, masked_text):
    """Create TAB format JSON structure."""
    entities = []
    for span in masked_spans:
        if span["span_text"] == "" or span["end_offset"] == 0:
            continue
        entities.append({
            "span_text": span["span_text"],
            "entity_type": span["entity_type"],
            "start_offset": span["start_offset"],
            "end_offset": span["end_offset"],
            "span_id": span["span_id"],
            "entity_id": span["entity_id"],
            "annotator": "",
            "identifier_type": span["identifier_type"]
        })

    return {
        "metadata": json_data['metadata'],
        "text": json_data['text'],
        "anonymized_text": masked_text,
        "entities": entities
    }

def apply_masking(text, masked_spans):
    """Apply masking to text."""
    if not masked_spans:
        return text

    # Sort spans in reverse order to mask from end
    sorted_spans = sorted(masked_spans, key=lambda x: x["start_offset"], reverse=True)
    masked_text = text
    for span in sorted_spans:
        start, end = span["start_offset"], span["end_offset"]
        masked_text = masked_text[:start] + f"[{span['entity_type']}]" + masked_text[end:]

    return masked_text

def print_results(text, masked_spans, tab_json, masked_text):
    """Print results."""
    print(f"\nFound {len(masked_spans)} masked spans (TAB format):")

    print("TAB format JSON structure:")
    print(f"Document ID: {tab_json[0]['doc_id']}")
    print(f"Number of entity mentions: {len(tab_json[0]['annotations']['annotator_1']['entity_mentions'])}")

    print("\nText masking results:")
    print("Original text preview:", text[:200] + "..." if len(text) > 200 else text)
    print("Masked text preview:", masked_text[:200] + "..." if len(masked_text) > 200 else masked_text)

def save_results(input_file, output_file, tab_json_list):
    """Save results to file."""
    # Determine output file path
    if output_file is None:
        file_name = os.path.basename(input_file)
        base_name = os.path.splitext(file_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine output directory based on input path
        if "panorama" in input_file.lower():
            output_dir = "data/spia/anonymized/panorama"
        elif "tab" in input_file.lower():
            output_dir = "data/spia/anonymized/tab"
        else:
            output_dir = os.path.dirname(input_file)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_longformer_{timestamp}.jsonl")

    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tab_json_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nResults saved to '{output_file}'")
        print(f"Processing completed successfully!")

    except Exception as e:
        print(f"Error saving output file '{output_file}': {e}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anonymize text using Longformer model trained on TAB dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/anonymize/anonymize_tab.py --input data/spia/tab_144.jsonl
    python src/anonymize/anonymize_tab.py --input data/spia/panorama_151.jsonl --output results.jsonl
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="src/baseline/TAB/longformer_experiments/long_model_our.pt",
        help="Path to fine-tuned model weights (.pt file)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    input_file = args.input
    model_path = args.model
    output_file = args.output

    # Input file validation
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)

    # Model file validation
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)

    json_list = []
    # Read input file
    try:
        if input_file.endswith('.jsonl'):
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    json_list.append(json.loads(line))
        else:
            print(f"Error: Input file '{input_file}' is not a jsonl file!")
            sys.exit(1)
        print(f"Loaded {len(json_list)} documents from {input_file}")
    except Exception as e:
        print(f"Error reading file '{input_file}': {e}")
        sys.exit(1)

    # Load model and tokenizer
    print(f"Loading model from '{model_path}'...")
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    print(f"Model loaded on device: {device}")

    print("=" * 60)
    print("TAB Anonymization - Batch Processing")
    print("=" * 60)
    print(f"\nProcessing {len(json_list)} documents...")

    tab_json_list = []
    masked_text_list = []
    masked_spans_list = []

    for json_data in tqdm(json_list, desc="Processing documents", unit="doc"):
        # Tokenize and predict
        tokens, predictions, confidence_scores = tokenize_and_predict(json_data['text'], model, tokenizer, device)
        # Process entities
        masked_spans = process_entities(json_data['text'], predictions, tokens["offset_mapping"], confidence_scores)
        # Generate masked text
        masked_text = apply_masking(json_data['text'], masked_spans)
        tab_json_list.append(create_tab_format_json(json_data, masked_spans, masked_text))
        masked_text_list.append(masked_text)
        masked_spans_list.append(masked_spans)

    # Save results
    save_results(input_file, output_file, tab_json_list)


if __name__ == "__main__":
    main()
