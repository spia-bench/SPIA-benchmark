"""
SPIA Human Label Checker
Evaluates inter-annotator agreement with rule/model/human fallback
Outputs evaluation results as JSON artifacts for further analysis
"""

import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from src.common_utils import load_jsonl_data
from src.evaluate.eval_utils import compare_pii_keywords, get_human_evaluation
from src.llm.client import get_annotation_agreement_evaluation
from src.llm.parser.annotation_agreement_parser import parse_agreement_response
from src.evaluate.calculate_inference_accuracy import calculate_statistics, print_statistics_summary, build_case_logs, print_case_logs


# ==================== Subject Statistics ====================

def calculate_subject_statistics(documents):
    """
    Calculate subject-level agreement statistics from aligned documents.

    Uses 'matched' field to determine subject matching status:
    - matched=True: Subject exists in both A and B annotations
    - matched=False, unmatched_side='A': Subject only in A (B has dummy PIIs)
    - matched=False, unmatched_side='B': Subject only in B (A has dummy PIIs)

    Args:
        documents: List of aligned documents with subjects containing 'matched' field

    Returns:
        dict with subject-level statistics
    """
    total_documents = len(documents)
    total_subjects_A = 0
    total_subjects_B = 0
    total_matched = 0
    total_unmatched_A = 0
    total_unmatched_B = 0
    docs_with_mismatch = 0
    by_document = []

    for doc in documents:
        data_id = doc['metadata']['data_id']
        subjects = doc.get('subjects', [])

        # Count subjects per annotation side using 'matched' field
        doc_subjects_A = 0
        doc_subjects_B = 0
        doc_matched = 0
        doc_unmatched_A = 0
        doc_unmatched_B = 0

        for subject in subjects:
            is_matched = subject.get('matched', True)  # Default True for legacy data
            unmatched_side = subject.get('unmatched_side', None)

            if is_matched:
                # Matched subject: exists in both A and B
                doc_matched += 1
                doc_subjects_A += 1
                doc_subjects_B += 1
            elif unmatched_side == 'A':
                # A-only subject: exists only in A
                doc_unmatched_A += 1
                doc_subjects_A += 1
            elif unmatched_side == 'B':
                # B-only subject: exists only in B
                doc_unmatched_B += 1
                doc_subjects_B += 1

        # Check if document has subject count mismatch
        if doc_subjects_A != doc_subjects_B:
            docs_with_mismatch += 1

        # Accumulate totals
        total_subjects_A += doc_subjects_A
        total_subjects_B += doc_subjects_B
        total_matched += doc_matched
        total_unmatched_A += doc_unmatched_A
        total_unmatched_B += doc_unmatched_B

        by_document.append({
            'data_id': data_id,
            'subjects_A': doc_subjects_A,
            'subjects_B': doc_subjects_B,
            'matched': doc_matched,
            'unmatched_A': doc_unmatched_A,
            'unmatched_B': doc_unmatched_B
        })

    # Calculate rates
    total_unique_subjects = total_matched + total_unmatched_A + total_unmatched_B
    subject_match_rate = total_matched / total_unique_subjects if total_unique_subjects > 0 else 0

    return {
        'total_documents': total_documents,
        'total_subjects_A': total_subjects_A,
        'total_subjects_B': total_subjects_B,
        'matched_subjects': total_matched,
        'unmatched_A': total_unmatched_A,
        'unmatched_B': total_unmatched_B,
        'total_unique_subjects': total_unique_subjects,
        'subject_match_rate': round(subject_match_rate, 4),
        'docs_with_subject_mismatch': docs_with_mismatch,
        'by_document': by_document
    }


def print_subject_statistics(subject_stats):
    """
    Print formatted subject-level statistics summary.

    Args:
        subject_stats: Subject statistics dict from calculate_subject_statistics()
    """
    print(f"\n{'='*80}")
    print(f"SUBJECT-LEVEL STATISTICS")
    print(f"{'='*80}")
    print(f"Total documents: {subject_stats['total_documents']}")
    print(f"Total subjects (Annotator A): {subject_stats['total_subjects_A']}")
    print(f"Total subjects (Annotator B): {subject_stats['total_subjects_B']}")
    print(f"Total unique subjects: {subject_stats['total_unique_subjects']}")
    print(f"\nSubject Agreement:")
    print(f"  - Matched subjects:   {subject_stats['matched_subjects']:4d} ({subject_stats['subject_match_rate']:6.2%})")
    print(f"  - Unmatched (A only): {subject_stats['unmatched_A']:4d}")
    print(f"  - Unmatched (B only): {subject_stats['unmatched_B']:4d}")
    print(f"\nDocuments with subject count mismatch: {subject_stats['docs_with_subject_mismatch']}")

    # Print per-document details for mismatched cases
    mismatched_docs = [d for d in subject_stats['by_document']
                       if d['unmatched_A'] > 0 or d['unmatched_B'] > 0]
    if mismatched_docs:
        print(f"\n{'='*80}")
        print(f"DOCUMENTS WITH UNMATCHED SUBJECTS ({len(mismatched_docs)} docs)")
        print(f"{'='*80}")
        print(f"{'Data ID':<20} {'A':>4} {'B':>4} {'Match':>6} {'Only A':>7} {'Only B':>7}")
        print(f"{'-'*80}")
        for doc in mismatched_docs:
            print(f"{doc['data_id']:<20} {doc['subjects_A']:4d} {doc['subjects_B']:4d} "
                  f"{doc['matched']:6d} {doc['unmatched_A']:7d} {doc['unmatched_B']:7d}")


# ==================== LLM & Human Evaluation ====================

def get_model_evaluation(keyword_a: str, keyword_b: str, tag: str) -> str:
    """
    Use LLM to evaluate if two keywords match (inter-annotator mode)

    Args:
        keyword_a: First keyword (Annotator A)
        keyword_b: Second keyword (Annotator B)
        tag: PII tag type

    Returns:
        "yes", "no", or "less precise"
    """
    response = get_annotation_agreement_evaluation(keyword_a, keyword_b, "inter_annotator")
    return parse_agreement_response(response)


def evaluate_with_fallback(keyword_a: str, keyword_b: str, tag: str, auto_score: float,
                          decider: str = "rule") -> tuple[float, str]:
    """
    Evaluate with fallback mechanism based on decider mode (inter-annotator)

    Args:
        keyword_a: First keyword (Annotator A)
        keyword_b: Second keyword (Annotator B)
        tag: PII tag type
        auto_score: Score from automatic rule-based evaluation
        decider: Evaluation strategy ("rule", "rule_model", "rule_model_human")

    Returns:
        tuple: (final_score, evaluation_method)
            - final_score: 0.0, 0.5, or 1.0
            - evaluation_method: "RULE", "MODEL", "HUMAN"
    """
    # If automatic evaluation succeeded, use it
    if auto_score > 0:
        return auto_score, "RULE"

    # Automatic evaluation failed (score = 0)
    if decider == "rule":
        return 0.0, "RULE"

    # Try model evaluation
    if decider in ["rule_model", "rule_model_human"]:
        try:
            model_result = get_model_evaluation(keyword_a, keyword_b, tag)
            if model_result == "yes":
                return 1.0, "MODEL"
            elif model_result == "less precise":
                return 0.5, "MODEL"
            elif model_result == "no":
                return 0.0, "MODEL"

            # If model gave unclear answer and we have human fallback
            if decider == "rule_model_human":
                human_result = get_human_evaluation(keyword_a, keyword_b, tag)
                if human_result == "yes":
                    return 1.0, "HUMAN"
                elif human_result == "less precise":
                    return 0.5, "HUMAN"
                else:
                    return 0.0, "HUMAN"

            return 0.0, "MODEL"  # Default if model unclear
        except Exception as e:
            print(f"  [ERROR] Model evaluation failed: {e}")
            if decider == "rule_model_human":
                # Fall back to human
                human_result = get_human_evaluation(keyword_a, keyword_b, tag)
                if human_result == "yes":
                    return 1.0, "HUMAN"
                elif human_result == "less precise":
                    return 0.5, "HUMAN"
                else:
                    return 0.0, "HUMAN"
            return 0.0, "MODEL"

    return 0.0, "RULE"


# ==================== Evaluation Functions ====================

def evaluate_subject_annotations(subject, config):
    """
    Evaluate agreement between annotation_A and annotation_B for a subject

    Two modes:
    1. inter_annotator: Evaluate union of tags from both annotations (both-way)
    2. ground_truth: Evaluate only tags present in GT annotation (one-way)
    """
    results = {}

    # Determine which tags to evaluate based on mode
    all_tags = set()

    # Inter-annotator mode: Evaluate union of all tags from both annotations
    for ann in subject["annotation_A"]:
        if "tag" in ann:
            all_tags.add(ann["tag"])
    for ann in subject["annotation_B"]:
        if "tag" in ann:
            all_tags.add(ann["tag"])

    # Filter tags if specified
    if config["tags_to_evaluate"]:
        all_tags = all_tags.intersection(set(config["tags_to_evaluate"]))

    for tag in all_tags:
        # Get all annotations for this tag (there can be multiple with same tag)
        # Apply certainty filter based on evaluation mode
        # Inter-annotator mode: both sides need to meet min_certainty
        anns_a = [a for a in subject["annotation_A"]
                    if a.get("tag") == tag]
        anns_b = [b for b in subject["annotation_B"]
                    if b.get("tag") == tag]

        # Skip if no valid annotations on the both side
        if anns_a[0]["certainty"] < config["min_certainty"] and anns_b[0]["certainty"] < config["min_certainty"]:
            continue

        # Find best matching pair among all combinations
        best_result = None
        best_score = -1.0

        for ann_a in anns_a:
            if anns_b:
                # Compare with all B annotations and find best match
                for ann_b in anns_b:
                    # Skip if both have certainty 0
                    if ann_a["certainty"] == 0 and ann_b["certainty"] == 0:
                        continue

                    # Step 1: Automatic rule-based comparison
                    auto_score = compare_pii_keywords(
                        ann_a["keyword"],
                        ann_b["keyword"],
                        tag,
                        location_max_depth=config.get("location_max_depth", None)
                    )

                    # Step 2 & 3: Fallback evaluation (model/human) if needed
                    final_score, evaluation_method = evaluate_with_fallback(
                        ann_a["keyword"],
                        ann_b["keyword"],
                        tag,
                        auto_score,
                        config.get("decider", "rule")
                    )

                    # Update best result if this is better
                    if final_score > best_score:
                        best_score = final_score
                        best_result = {
                            "keyword_A": ann_a["keyword"],
                            "keyword_B": ann_b["keyword"],
                            "certainty_A": ann_a["certainty"],
                            "certainty_B": ann_b["certainty"],
                            "hardness_A": ann_a["hardness"],
                            "hardness_B": ann_b["hardness"],
                            "agreement_score": final_score,
                            "evaluation_method": evaluation_method,
                            "match": final_score >= 1.0,
                            "less_precise": 0.0 < final_score < 1.0,
                            "mismatch": final_score == 0.0
                        }
            else:
                # No B annotations - mismatch
                if best_score < 0:  # Only set if no better match found
                    best_score = 0.0
                    best_result = {
                        "keyword_A": ann_a["keyword"],
                        "keyword_B": "",
                        "certainty_A": ann_a["certainty"],
                        "certainty_B": 0,
                        "hardness_A": ann_a["hardness"],
                        "hardness_B": 0,
                        "agreement_score": 0.0,
                        "evaluation_method": "RULE",
                        "match": False,
                        "less_precise": False,
                        "mismatch": True
                    }

        # Save best result for this tag
        if best_result:
            results[tag] = best_result

    return results


def process_subject_wrapper(subject_data, config):
    """
    Wrapper function for parallel processing of subject annotations

    Args:
        subject_data: Dict with subject information and data_id
        config: Configuration dictionary

    Returns:
        Dict with subject_id, description, evaluations, and data_id for reassembly
    """
    try:
        evaluations = evaluate_subject_annotations({
            "id": subject_data["id"],
            "description": subject_data["description"],
            "annotation_A": subject_data["annotation_A"],
            "annotation_B": subject_data["annotation_B"]
        }, config)

        return {
            "data_id": subject_data["data_id"],
            "subject_id": subject_data["id"],
            "description": subject_data["description"],
            "evaluations": evaluations,
            "doc_index": subject_data["doc_index"],
            "subject_index": subject_data["subject_index"]
        }
    except Exception as e:
        print(f"\n[ERROR] Failed to process subject {subject_data['id']}: {e}")
        return {
            "data_id": subject_data["data_id"],
            "subject_id": subject_data["id"],
            "description": subject_data["description"],
            "evaluations": {},
            "doc_index": subject_data["doc_index"],
            "subject_index": subject_data["subject_index"],
            "error": str(e)
        }


# ==================== Main Evaluation ====================

def evaluate_annotations(input_file, output_file, config):
    """
    Main evaluation function with parallel processing for LLM evaluation
    Outputs evaluation results as JSON artifact
    """
    # Load JSONL data
    documents = load_jsonl_data(input_file)

    # Prepare all subjects for parallel processing
    all_subjects = []
    for doc_idx, data in enumerate(documents):
        for subj_idx, subject in enumerate(data["subjects"]):
            subject_data = {
                "id": subject["id"],
                "description": subject["description"],
                "annotation_A": subject["PIIs"]["annotation_A"],
                "annotation_B": subject["PIIs"]["annotation_B"],
                "data_id": data["metadata"]["data_id"],
                "doc_index": doc_idx,
                "subject_index": subj_idx
            }
            all_subjects.append(subject_data)

    print(f"Processing {len(all_subjects)} subjects across {len(documents)} documents with {config.get('num_workers', 20)} workers...")

    # Process subjects in parallel
    results_map = {}
    with ThreadPoolExecutor(max_workers=config.get('num_workers', 20)) as executor:
        # Submit all tasks
        future_to_subject = {
            executor.submit(process_subject_wrapper, subject_data, config): subject_data
            for subject_data in all_subjects
        }

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_subject), total=len(all_subjects), desc="Evaluating"):
            result = future.result()
            # Store result indexed by (doc_index, subject_index) for reassembly
            key = (result["doc_index"], result["subject_index"])
            results_map[key] = result

    # Reassemble results into original document structure
    all_results = []
    for doc_idx, data in enumerate(documents):
        doc_results = {
            "data_id": data["metadata"]["data_id"],
            "number_of_subjects": data["metadata"]["number_of_subjects"],
            "subjects": []
        }

        for subj_idx in range(len(data["subjects"])):
            key = (doc_idx, subj_idx)
            result = results_map.get(key)
            if result:
                doc_results["subjects"].append({
                    "subject_id": result["subject_id"],
                    "description": result["description"],
                    "evaluations": result["evaluations"]
                })
                if "error" in result:
                    print(f"[WARNING] Subject {result['subject_id']} had error: {result['error']}")

        all_results.append(doc_results)

    # Calculate statistics
    from src.evaluate.calculate_inference_accuracy import calculate_statistics, build_case_logs
    pii_statistics = calculate_statistics(all_results)

    # Calculate subject-level statistics
    subject_statistics = calculate_subject_statistics(documents)

    # Build case logs if requested
    case_logs = None
    if config.get('show_case_logs', True):
        case_logs = build_case_logs(all_results, filter_mode=config.get('case_log_filter', 'disagree'))

    # Save evaluation results as artifact with statistics and case logs
    evaluation_data = {
        "config": config,
        "statistics": {
            "pii_level": pii_statistics,
            "subject_level": subject_statistics
        },
        "case_logs": case_logs,
        "detailed_results": all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluation complete. Results saved to: {output_file}")
    print(f"  Total documents: {len(all_results)}")
    total_subjects = sum(len(doc["subjects"]) for doc in all_results)
    print(f"  Total subjects: {total_subjects}")

    # Return both evaluation data and original documents for GT update
    evaluation_data['input_documents'] = documents
    return evaluation_data


def build_case_logs_inter(all_results, input_documents, min_certainty=3):
    """
    Build case logs from evaluation results and update GT files

    Args:
        all_results: List of document evaluation results
        input_documents: Original input documents from aligned file
        min_certainty: Minimum certainty level to include (0-5)
    Returns:
        list of formatted log strings
    """
    logs = []
    mismatch = []
    less_precise = []
    perfect_match = []
    perfect_match_less_precise = []
    match = []

    matched_data = []
    unmatched_data = []
    need_to_check_data_ids = []

    for doc_results in all_results:
        data_id = doc_results["data_id"]

        for subject_results in doc_results["subjects"]:
            subject_id = subject_results["subject_id"]

            for tag, result in subject_results["evaluations"].items():
                if result["certainty_A"] < min_certainty and result["certainty_B"] < min_certainty:
                    continue
                score = result["agreement_score"]
                method = result.get("evaluation_method", "RULE")
                # Format log entry with evaluation method and score
                log_entry = (
                    f"  [{data_id}][{subject_id}] [{method}] [{tag}] "
                    f"'{result['keyword_A']}:{result['certainty_A']}' vs '{result['keyword_B']}:{result['certainty_B']}' | "
                    f"score={score}"
                )
                # Determine result text
                if score == 0.0:
                    need_to_check_data_ids.append(data_id)
                    result_text = "mismatch"
                    log_entry += f" ({result_text})"
                    mismatch.append(log_entry)
                    if result['certainty_A'] == 0 or result['certainty_B'] == 0:    
                        data = {
                            "data_id": data_id,
                            "subject_id": subject_id,
                            "tag": tag,
                            "keyword": result['keyword_B'] if result['certainty_A'] == 0 else result['keyword_A'],
                            "certainty": result['certainty_B'] if result['certainty_A'] == 0 else result['certainty_A'],
                            "hardness": result['hardness_B'] if result['certainty_A'] == 0 else result['hardness_A']
                        }
                        unmatched_data.append(data)

                elif score == 0.5:
                    need_to_check_data_ids.append(data_id)
                    result_text = "less precise"
                    log_entry += f" ({result_text})"
                    less_precise.append(log_entry)
                else:
                    if result['keyword_A'] == result['keyword_B']:
                        if result['certainty_A'] != result['certainty_B']:
                            need_to_check_data_ids.append(data_id)
                            result_text = "Perfect match less precise"
                            log_entry += f" ({result_text})"
                            data = {
                                "data_id": data_id,
                                "subject_id": subject_id,
                                "tag": tag,
                                "keyword": result['keyword_A'],
                                "certainty": result['certainty_B'] if result['certainty_B'] < result['certainty_A'] else result['certainty_A'],
                                "hardness": result['hardness_B'] if result['certainty_B'] < result['certainty_A'] else result['hardness_A']
                            }
                            matched_data.append(data)
                            perfect_match_less_precise.append(log_entry)
                        else:
                            result_text = "Perfect match"
                            log_entry += f" ({result_text})"
                            perfect_match.append(log_entry)
                    else:
                        result_text = "match"
                        log_entry += f" ({result_text})"
                        match.append(log_entry)
                        need_to_check_data_ids.append(data_id)
                logs.append(log_entry)
    # save logs to file
    need_to_check_file = f"data/spia/inter_annotator/{args.data}/{args.data}_inter_need_to_check.txt"
    with open(need_to_check_file, "w") as f:
        print(f"[0.0] Mismatch: {len(mismatch)}")
        f.write(f"[0.0] Mismatch: {len(mismatch)}\n")
        for log in mismatch:
            print(log)
            f.write(log + "\n")
        print(f"[0.5] Less precise: {len(less_precise)}")
        f.write(f"[0.5] Less precise: {len(less_precise)}\n")
        for log in less_precise:
            print(log)
            f.write(log + "\n")
        print(f"[1.0] Match (keyword differs): {len(match)}")
        f.write(f"[1.0] Match (keyword differs): {len(match)}\n")
        for log in match:
            print(log)
            f.write(log + "\n")
        print(f"[1.0] Match (certainty differs): {len(perfect_match_less_precise)}")
        f.write(f"[1.0] Match (certainty differs): {len(perfect_match_less_precise)}\n")
        for log in perfect_match_less_precise:
            print(log)
            f.write(log + "\n")
        print(f"[1.0] Perfect match: {len(perfect_match)}")
        f.write(f"[1.0] Perfect match: {len(perfect_match)}\n")
        # for log in perfect_match:
        #     print(log)
        #     f.write(log + "\n")
    # Convert matched_data list to dictionary for fast lookup
    # Key: (data_id, subject_id, tag), Value: {keyword, certainty, hardness}
    matched_dict = {}
    for match_info in matched_data:
        key = (match_info['data_id'], match_info['subject_id'], match_info['tag'])
        matched_dict[key] = {
            'keyword': match_info['keyword'],
            'certainty': match_info['certainty'],
            'hardness': match_info['hardness']
        }

    unmatched_dict = {}
    for unmatched_info in unmatched_data:
        key = (unmatched_info['data_id'], unmatched_info['subject_id'], unmatched_info['tag'])
        unmatched_dict[key] = {
            'keyword': unmatched_info['keyword'],
            'certainty': unmatched_info['certainty'],
            'hardness': unmatched_info['hardness']
        }

    # Process input documents and update PIIs with resolved conflicts
    all_data = []
    for data in input_documents:
        data_id = data['metadata']['data_id']

        # Create updated document structure
        updated_doc = {
            'metadata': data['metadata'],
            'text': data['text'],
            'subjects': []
        }

        # Iterate through subjects
        for subject in data['subjects']:
            subject_id = subject['id']

            # Convert annotation_A/B structure to single PIIs list with updates
            updated_piis = []

            # Collect all unique PIIs from both annotations
            if 'PIIs' in subject and 'annotation_A' in subject['PIIs']:
                # Process annotation_A PIIs
                for pii in subject['PIIs']['annotation_A']:
                    key = (data_id, subject_id, pii['tag'])
                    updated_pii = pii.copy()

                    # Apply updates if available
                    if key in matched_dict:
                        match_info = matched_dict[key]
                        updated_pii['keyword'] = match_info['keyword']
                        updated_pii['certainty'] = match_info['certainty']
                        updated_pii['hardness'] = match_info['hardness']
                    elif key in unmatched_dict:
                        unmatched_info = unmatched_dict[key]
                        updated_pii['keyword'] = unmatched_info['keyword']
                        updated_pii['certainty'] = unmatched_info['certainty']
                        updated_pii['hardness'] = unmatched_info['hardness']

                    updated_piis.append(updated_pii)

            # Create updated subject with single PIIs list
            updated_subject = {
                'id': subject_id,
                'description': subject.get('description', ''),
                'PIIs': updated_piis
            }
            updated_doc['subjects'].append(updated_subject)

        all_data.append(updated_doc)

    # Convert to set to remove duplicates and for efficient lookup
    need_to_check_data_ids_set = set(need_to_check_data_ids)

    # Save split annotations: agreed vs conflict
    inter_dir = f"data/spia/inter_annotator/{args.data}"
    os.makedirs(inter_dir, exist_ok=True)
    agreed_file = f"{inter_dir}/{args.data}_inter_agreed.jsonl"
    conflict_file = f"{inter_dir}/{args.data}_inter_conflict.jsonl"

    with open(agreed_file, "w", encoding='utf-8') as f1:
        with open(conflict_file, "w", encoding='utf-8') as f2:
            for data in all_data:
                data_id = data['metadata']['data_id']
                if data_id in need_to_check_data_ids_set:
                    f2.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    f1.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\n✓ Split annotations saved:")
    print(f"  Agreed (unchanged): {agreed_file}")
    print(f"  Conflict (needs review): {conflict_file}")
    print(f"  Total conflicts: {len(need_to_check_data_ids_set)}")

    
    return logs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inter-annotator agreement with configurable parameters"
    )
    parser.add_argument("-d", "--data", type=str, default="tab", choices=["tab", "panorama"],
                        help="Dataset to process")
    parser.add_argument("-p", "--path", type=str, default="none",
                        help="Input file path (default: auto-detect from data)")
    parser.add_argument("--min_certainty", type=int, default=3,
                        help="Minimum certainty level to include (0-5)")
    parser.add_argument("--location_max_depth", type=int, default=4,
                        help="Max depth for LOCATION comparison (None=all levels)")
    parser.add_argument("--decider", type=str, default="rule_model",
                        choices=["rule", "rule_model", "rule_model_human"],
                        help="Evaluation strategy")
    parser.add_argument("--num_workers", type=int, default=20,
                        help="Number of parallel workers")
    parser.add_argument("--tags", type=str, nargs='+', default=None,
                        help="Specific tags to evaluate (default: all tags)")

    # LLM configuration (for rule_model and rule_model_human deciders)
    parser.add_argument("--api_provider", type=str, default="openai",
                        choices=["openai", "anthropic", "ollama"],
                        help="API provider for LLM evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name for LLM evaluation")

    parser.add_argument("--show_case_logs", action="store_true", default=True,
                        help="Show individual case details")
    parser.add_argument("--no_case_logs", action="store_false", dest="show_case_logs",
                        help="Hide individual case details")
    parser.add_argument("--case_log_filter", type=str, default="disagree",
                        choices=["all", "disagree", "mismatch"],
                        help="Filter for case logs (disagree=score<1.0, mismatch=score=0.0)")
    return parser.parse_args()

# ==================== Main ====================

if __name__ == "__main__":
    args = parse_args()

    if args.path == "none":
        INPUT_FILE = f"data/spia/inter_annotator/{args.data}/{args.data}_inter_annotator_aligned.jsonl"
    else:
        INPUT_FILE = args.path

    # Save output to inter_annotator directory
    OUTPUT_FILE = f"data/spia/inter_annotator/{args.data}/{args.data}_inter_evaluation.json"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Configure LLM client if using model-based decider
    if args.decider in ["rule_model", "rule_model_human"]:
        from src.llm.client import set_api_provider
        set_api_provider(provider=args.api_provider, model=args.model)

    # Build config from command-line arguments
    config = {
        "min_certainty": args.min_certainty,
        "tags_to_evaluate": args.tags,
        "decider": args.decider,
        "location_max_depth": args.location_max_depth,
        "num_workers": args.num_workers,
        "show_case_logs": args.show_case_logs,
        "case_log_filter": args.case_log_filter,
        "api_provider": args.api_provider,
        "model": args.model
    }

    # Print configuration
    print(f"\n{'='*60}")
    print(f"Inter-Annotator Evaluation Configuration")
    print(f"{'='*60}")
    print(f"API Provider: {config['api_provider']}")
    print(f"Model: {config['model']}")
    print(f"Min certainty: {config['min_certainty']}")
    print(f"Location max depth: {config['location_max_depth']}")
    print(f"Decider: {config['decider']}")
    print(f"Num workers: {config['num_workers']}")
    print(f"Tags filter: {config['tags_to_evaluate'] or 'All tags'}")
    print(f"{'='*60}\n")

    # Run evaluation
    evaluation_data = evaluate_annotations(INPUT_FILE, OUTPUT_FILE, config)

    # Print subject-level statistics
    print_subject_statistics(evaluation_data["statistics"]["subject_level"])

    # Calculate and display PII-level statistics
    pii_statistics = calculate_statistics(evaluation_data["detailed_results"], min_certainty=args.min_certainty)
    print_statistics_summary(pii_statistics)

    # Print case logs if requested
    if args.show_case_logs:
        case_logs = build_case_logs(evaluation_data["detailed_results"], filter_mode=args.case_log_filter, min_certainty=args.min_certainty)
        print_case_logs(case_logs, filter_mode=args.case_log_filter)
        # Build inter-annotator specific logs and update ground truth
        build_case_logs_inter(
            evaluation_data["detailed_results"],
            evaluation_data["input_documents"],
            min_certainty=args.min_certainty
        )
