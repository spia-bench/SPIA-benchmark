"""
Align Human and LLM Annotations
Combines ground truth (human) annotations with LLM annotations into a unified format
with annotation_A (ground truth) and annotation_B (LLM) for comparison.
"""

from pathlib import Path
import argparse

from src.common_utils import load_jsonl_data, save_jsonl_data


def parse_args():
    parser = argparse.ArgumentParser(description="Align human GT and LLM annotations")
    parser.add_argument("--gt_file", type=str, required=True,
                        help="Ground truth JSONL file path")
    parser.add_argument("--llm_file", type=str, required=True,
                        help="LLM annotated JSONL file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path for combined annotations")
    return parser.parse_args()

# ==================== Main Functions ====================

def align_annotations(args):
    ground_truth_file = args.gt_file
    llm_file = args.llm_file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ground_truth_data = load_jsonl_data(ground_truth_file)
    llm_data = load_jsonl_data(llm_file)

    # Index by data_id
    gt_by_id = {data["metadata"]["data_id"]: data for data in ground_truth_data}
    llm_by_id = {data["metadata"]["data_id"]: data for data in llm_data}

    # Check data_id alignment
    gt_ids = set(gt_by_id.keys())
    llm_ids = set(llm_by_id.keys())

    # Find common data_ids (intersection)
    common_ids = gt_ids & llm_ids
    missing_in_llm = gt_ids - llm_ids
    missing_in_gt = llm_ids - gt_ids

    # Count total subjects
    gt_total_subjects = sum(len(data.get("subjects", [])) for data in ground_truth_data)
    llm_total_subjects = sum(len(data.get("subjects", [])) for data in llm_data)

    print(f"\nData ID alignment:")
    print(f"  GT total: {len(gt_ids)} docs, {gt_total_subjects} subjects")
    print(f"  LLM total: {len(llm_ids)} docs, {llm_total_subjects} subjects")
    print(f"  Common (matched): {len(common_ids)}")
    if missing_in_llm:
        print(f"  Missing in LLM: {len(missing_in_llm)} ({sorted(list(missing_in_llm))[:5]}...)")
    if missing_in_gt:
        print(f"  Missing in GT: {len(missing_in_gt)} ({sorted(list(missing_in_gt))[:5]}...)")

    if not common_ids:
        raise ValueError("No common data IDs found between GT and LLM!")

    # Create aligned annotations for common data_ids only
    aligned_data = []
    mismatched_subject_counts = []

    for data_id in sorted(common_ids):
        gt_entry = gt_by_id[data_id]
        llm_entry = llm_by_id[data_id]

        # Check text consistency
        if gt_entry["text"] != llm_entry["text"]:
            print(f"[WARNING] Text mismatch for {data_id}")

        # Extract subjects
        annotation_a = gt_entry["subjects"]
        annotation_b = llm_entry["subjects"]

        # Track subject count mismatches
        if len(annotation_a) != len(annotation_b):
            mismatched_subject_counts.append({
                "data_id": data_id,
                "gt_count": len(annotation_a),
                "llm_count": len(annotation_b)
            })

        # Create aligned structure
        # annotation_A = Ground Truth (always, based on original text)
        # annotation_B = LLM prediction (always, may be based on anonymized text)
        aligned = {
            "metadata": {
                "data_id": data_id,
                "number_of_subjects": gt_entry["metadata"]["number_of_subjects"],
                "gt_file": ground_truth_file,
                "llm_file": llm_file
            },
            "text": gt_entry["text"],
        }

        # Include anonymized_text if present in LLM entry (for anon mode alignment)
        # Place it right after "text" field
        if "anonymized_text" in llm_entry:
            aligned["anonymized_text"] = llm_entry["anonymized_text"]

        # Add annotations after text fields
        aligned["annotation_A"] = annotation_a  # GT (human, from original text)
        aligned["annotation_B"] = annotation_b  # LLM (from original or anonymized text)

        aligned_data.append(aligned)

    # Save aligned data
    save_jsonl_data(aligned_data, output_path)

    print(f"✓ Aligned {len(aligned_data)} records to: {output_path}")
    if mismatched_subject_counts:
        print(f"  Subject count mismatches: {len(mismatched_subject_counts)}")


# ==================== Main ====================

if __name__ == "__main__":
    args = parse_args()
    align_annotations(args)
