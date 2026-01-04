"""
NER Inter-Annotator Agreement Checker
Evaluates inter-annotator agreement for NER labeling
Compares entities from two annotators (a4 and a5) based on span_text, entity_type, and offsets
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def calculate_entity_agreement(entity_a: Dict, entity_b: Dict) -> float:
    """
    Calculate agreement score between two entities
    
    Args:
        entity_a: Entity from annotator A with keys: span_text, entity_type, start_offset, end_offset, identifier_type
        entity_b: Entity from annotator B with same keys
    
    Returns:
        float: 1.0 (perfect match), 0.5 (partial overlap), 0.0 (no match)
    
    Note:
        entity_id is not compared directly as it's just a number.
        Instead, we compare the actual entity information (span_text, offsets, etc.)
    """
    # Check if entity_type matches
    if entity_a.get("entity_type") != entity_b.get("entity_type"):
        return 0.0
    
    # Check if span_text matches
    if entity_a.get("span_text") != entity_b.get("span_text"):
        return 0.0
    
    # Check offsets
    start_a = entity_a.get("start_offset")
    end_a = entity_a.get("end_offset")
    start_b = entity_b.get("start_offset")
    end_b = entity_b.get("end_offset")
    
    # Check identifier_type
    identifier_type_a = entity_a.get("identifier_type")
    identifier_type_b = entity_b.get("identifier_type")
    
    # Perfect match: all offsets and identifier_type are the same
    if start_a == start_b and end_a == end_b and identifier_type_a == identifier_type_b:
        return 1.0
    
    # If identifier_type doesn't match, cannot be perfect match
    # But can still be partial match if offsets overlap
    if identifier_type_a != identifier_type_b:
        # Check if one contains the other (partial overlap) despite identifier_type mismatch
        if (start_a <= start_b and end_a >= end_b) or (start_b <= start_a and end_b >= end_a):
            return 0.5
        # Check if they overlap at all
        if not (end_a < start_b or end_b < start_a):
            return 0.5
        return 0.0
    
    # identifier_type matches, but offsets differ
    # Check if one contains the other (partial overlap)
    # A contains B: start_a <= start_b and end_a >= end_b
    # B contains A: start_b <= start_a and end_b >= end_a
    if (start_a <= start_b and end_a >= end_b) or (start_b <= start_a and end_b >= end_a):
        return 0.5
    
    # Check if they overlap at all
    # Overlap: not (end_a < start_b or end_b < start_a)
    if not (end_a < start_b or end_b < start_a):
        return 0.5
    
    # No match
    return 0.0


def find_best_match(entity_a: Dict, entities_b: List[Dict]) -> Tuple[Optional[Dict], float]:
    """
    Find the best matching entity from entities_b for entity_a
    
    Args:
        entity_a: Entity from annotator A
        entities_b: List of entities from annotator B
    
    Returns:
        tuple: (best_matching_entity, best_score)
    """
    best_match = None
    best_score = -1.0
    
    for entity_b in entities_b:
        score = calculate_entity_agreement(entity_a, entity_b)
        if score > best_score:
            best_score = score
            best_match = entity_b
    
    return best_match, best_score


def evaluate_document_entities(doc_a: Dict, doc_b: Dict) -> Dict:
    """
    Evaluate agreement between entities from two annotators for a single document
    
    Args:
        doc_a: Document from annotator A
        doc_b: Document from annotator B
    
    Returns:
        dict with evaluation results
    """
    data_id = doc_a.get("metadata", {}).get("data_id")
    
    entities_a = doc_a.get("entities", [])
    entities_b = doc_b.get("entities", [])
      
    evaluations = []
    
    # For each entity in A, find best match in B
    for idx_a, entity_a in enumerate(entities_a):
        for idx_b, entity_b in enumerate(entities_b):
            if entity_a.get("start_offset") == entity_b.get("start_offset"):
                evaluation = {
                    "entity_A": entity_a,
                    "entity_B": entity_b,
                    "match": True,
                    "Entity_Type": 0.0,
                    "Identifier_Type": 0.0
                }
                if entity_a.get("entity_type") == entity_b.get("entity_type"):
                    if entity_a.get("end_offset") == entity_b.get("end_offset"):
                        evaluation["Entity_Type"] = 1.0
                    else:
                        evaluation["Entity_Type"] = 0.5
                if entity_a.get("identifier_type") == entity_b.get("identifier_type"):
                    if entity_a.get("end_offset") == entity_b.get("end_offset"):
                        evaluation["Identifier_Type"] = 1.0
                    else:
                        evaluation["Identifier_Type"] = 0.5
                evaluations.append(evaluation)
                break
    
    # find does nor matched entities in A and B
    for entity_a in entities_a:
        if entity_a not in [evaluation["entity_A"] for evaluation in evaluations]:
            evaluations.append({
                "entity_A": entity_a,
                "entity_B": None,
                "match": False,
                "Entity_Type": 0.0,
                "Identifier_Type": 0.0
            })
            evaluations.append(evaluation)
            break
    for entity_b in entities_b:
        if entity_b not in [evaluation["entity_B"] for evaluation in evaluations]:
            evaluations.append({
                "entity_A": None,
                "entity_B": entity_b,
                "match": False,
                "Entity_Type": 0.0,
                "Identifier_Type": 0.0
            })
            evaluations.append(evaluation)
            break
    return {
        "data_id": data_id,
        "text": doc_a.get("text"),
        "evaluations": evaluations
    }


def calculate_statistics(all_results: List[Dict]) -> Dict:
    """
    Calculate overall statistics from evaluation results
    
    Args:
        all_results: List of document evaluation results
    
    Returns:
        dict with overall and by_tag statistics
    """
    total_entities = 0
    total_Entity_Type_Exact_Match = 0
    total_Entity_Type_Partial_Match = 0
    total_Entity_Type_Mismatch = 0
    total_Identifier_Type_Exact_Match = 0
    total_Identifier_Type_Partial_Match = 0
    total_Identifier_Type_Mismatch = 0
    
    tag_stats = {
        "Entity_Type_Exact_Match": 0,
        "Entity_Type_Partial_Match": 0,
        "Entity_Type_Mismatch": 0,
        "Identifier_Type_Exact_Match": 0,
        "Identifier_Type_Partial_Match": 0,
        "Identifier_Type_Mismatch": 0
    }
    
    for doc_result in all_results:
        for evaluation in doc_result["evaluations"]:
            total_entities += 1
            if evaluation["Entity_Type"] == 1.0:
                total_Entity_Type_Exact_Match += 1
                total_Entity_Type_Partial_Match += 1
            elif evaluation["Entity_Type"] == 0.5:
                total_Entity_Type_Partial_Match += 1
            else:
                total_Entity_Type_Mismatch += 1
            if evaluation["Identifier_Type"] == 1.0:
                total_Identifier_Type_Exact_Match += 1
                total_Identifier_Type_Partial_Match += 1
            elif evaluation["Identifier_Type"] == 0.5:
                total_Identifier_Type_Partial_Match += 1
            else:
                total_Identifier_Type_Mismatch += 1
    
    
    return {
        "overall": {
            "total_entities": total_entities,
            "total_Entity_Type_Exact_Match": total_Entity_Type_Exact_Match,
            "total_Entity_Type_Partial_Match": total_Entity_Type_Partial_Match,
            "total_Entity_Type_Mismatch": total_Entity_Type_Mismatch,
            "total_Identifier_Type_Exact_Match": total_Identifier_Type_Exact_Match,
            "total_Identifier_Type_Partial_Match": total_Identifier_Type_Partial_Match,
            "total_Identifier_Type_Mismatch": total_Identifier_Type_Mismatch,
            "Entity_Type_Exact_Match_Rate": total_Entity_Type_Exact_Match / total_entities,
            "Entity_Type_Partial_Match_Rate": total_Entity_Type_Partial_Match / total_entities,
            "Identifier_Type_Exact_Match_Rate": total_Identifier_Type_Exact_Match / total_entities,
            "Identifier_Type_Partial_Match_Rate": total_Identifier_Type_Partial_Match / total_entities,
        }
    }


def print_statistics(statistics: Dict):
    """
    Print statistics in formatted table
    """
    overall = statistics["overall"]
    
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total entities: {overall['total_entities']}")
    print(f"  - Entity Type Exact Match:     {overall['total_Entity_Type_Exact_Match']:>4} ({overall['Entity_Type_Exact_Match_Rate']:>5.2f}%)")
    print(f"  - Entity Type Partial Match:     {overall['total_Entity_Type_Partial_Match']:>4} ({overall['Entity_Type_Partial_Match_Rate']:>5.2f}%)")
    print(f"  - Identifier Type Exact Match:     {overall['total_Identifier_Type_Exact_Match']:>4} ({overall['Identifier_Type_Exact_Match_Rate']:>5.2f}%)")
    print(f"  - Identifier Type Partial Match:     {overall['total_Identifier_Type_Partial_Match']:>4} ({overall['Identifier_Type_Partial_Match_Rate']:>5.2f}%)")

def build_case_logs(all_results: List[Dict], filter_mode: str = "disagree") -> List[str]:
    """
    Build case logs from evaluation results
    
    Args:
        all_results: List of document evaluation results
        filter_mode: "all", "disagree" (score < 1.0), or "mismatch" (score = 0.0)
    
    Returns:
        list of formatted log strings
    """
    logs = []
    
    for doc_result in all_results:
        data_id = doc_result["data_id"]
        
        for evaluation in doc_result["evaluations"]:
            
            entity_a = evaluation["entity_A"]
            entity_b = evaluation["entity_B"]
            
            if entity_a and entity_b:
                id_type_a = entity_a.get('identifier_type', 'N/A')
                id_type_b = entity_b.get('identifier_type', 'N/A')
                if evaluation["Entity_Type"] == 1.0:
                    entity_type_match = "Entity_Exact"
                elif evaluation["Entity_Type"] == 0.5:
                    entity_type_match = "Entity_Partial"
                else:
                    entity_type_match = "Entity_Mismatch"
                if evaluation["Identifier_Type"] == 1.0:
                    identifier_type_match = "Identifier_Exact"
                elif evaluation["Identifier_Type"] == 0.5:
                    identifier_type_match = "Identifier_Partial"
                else:
                    identifier_type_match = "Identifier_Mismatch"
                log_entry = (
                    f"  [{data_id}] "
                    f"[{entity_a.get('entity_type', 'UNKNOWN')}] "
                    f"'{entity_a.get('span_text', '')}' "
                    f"({entity_a.get('start_offset')}-{entity_a.get('end_offset')}) "
                    f"[{id_type_a}] vs "
                    f"[{entity_b.get('entity_type', 'UNKNOWN')}] "
                    f"'{entity_b.get('span_text', '')}' "
                    f"({entity_b.get('start_offset')}-{entity_b.get('end_offset')}) "
                    f"[{id_type_b}] {entity_type_match} {identifier_type_match} | "
                )
            elif entity_a:
                id_type_a = entity_a.get('identifier_type', 'N/A')
                if evaluation["Entity_Type"] == 1.0:
                    entity_type_match = "Entity_Exact"
                elif evaluation["Entity_Type"] == 0.5:
                    entity_type_match = "Entity_Partial"
                else:
                    entity_type_match = "Entity_Mismatch"
                if evaluation["Identifier_Type"] == 1.0:
                    identifier_type_match = "Identifier_Exact"
                elif evaluation["Identifier_Type"] == 0.5:
                    identifier_type_match = "Identifier_Partial"
                else:
                    identifier_type_match = "Identifier_Mismatch"
                log_entry = (
                    f"  [{data_id}] "
                    f"[{entity_a.get('entity_type', 'UNKNOWN')}] "
                    f"'{entity_a.get('span_text', '')}' "
                    f"({entity_a.get('start_offset')}-{entity_a.get('end_offset')}) "
                    f"[{id_type_a}] vs [UNKNOWN] <NONE> {entity_type_match} {identifier_type_match} | "
                )
            else:  # entity_b
                id_type_b = entity_b.get('identifier_type', 'N/A')
                if evaluation["Entity_Type"] == 1.0:
                    entity_type_match = "Entity_Exact"
                elif evaluation["Entity_Type"] == 0.5:
                    entity_type_match = "Entity_Partial"
                else:
                    entity_type_match = "Entity_Mismatch"
                if evaluation["Identifier_Type"] == 1.0:
                    identifier_type_match = "Identifier_Exact"
                elif evaluation["Identifier_Type"] == 0.5:
                    identifier_type_match = "Identifier_Partial"
                else:
                    identifier_type_match = "Identifier_Mismatch"
                log_entry = (
                    f"  [{data_id}] [UNKNOWN] "
                    f"<NONE> vs "
                    f"[{entity_b.get('entity_type', 'UNKNOWN')}] "
                    f"'{entity_b.get('span_text', '')}' "
                    f"({entity_b.get('start_offset')}-{entity_b.get('end_offset')}) "
                    f"[{id_type_b}] {entity_type_match} {identifier_type_match} | "
                )
            
            logs.append(log_entry)
    
    return logs


def print_case_logs(logs: List[str], filter_mode: str = "disagree"):
    """
    Print case logs grouped by score
    """
    entity_exact_match = []
    entity_partial_match = []
    entity_mismatch = []
    identifier_exact_match = []
    identifier_partial_match = []
    identifier_mismatch = []
    
    for log in logs:
        if "Entity_Exact" in log:
            entity_exact_match.append(log)
        elif "Entity_Partial" in log:
            entity_partial_match.append(log)
        elif "Entity_Mismatch" in log:
            entity_mismatch.append(log)
        if "Identifier_Exact" in log:
            identifier_exact_match.append(log)
        elif "Identifier_Partial" in log:
            identifier_partial_match.append(log)
        elif "Identifier_Mismatch" in log:
            identifier_mismatch.append(log)
    
    print(f"\n{'='*80}")
    print(f"CASE LOGS (filter: {filter_mode})")
    print(f"{'='*80}")
    
    if filter_mode == "all" or filter_mode == "disagree":
        print(f"\n[1.0] Entity Exact Match: {len(entity_exact_match)}")
        if entity_exact_match and filter_mode == "all":
            for log in entity_exact_match[:10]:  # Show first 10
                print(log)
            if len(entity_exact_match) > 10:
                print(f"  ... and {len(entity_exact_match) - 10} more")

    if filter_mode == "all" or filter_mode == "disagree":
        print(f"\n[0.5] Entity Partial Match: {len(entity_partial_match)}")
        for log in entity_partial_match:
            print(log)

    if filter_mode == "all" or filter_mode == "disagree":
        print(f"\n[0.0] Entity Mismatch: {len(entity_mismatch)}")
        for log in entity_mismatch:
            print(log)
    
    if filter_mode == "all" or filter_mode == "disagree":
        print(f"\n[1.0] Identifier Exact Match: {len(identifier_exact_match)}")
        for log in identifier_exact_match:
            print(log)
    
    if filter_mode == "all" or filter_mode == "disagree":
        print(f"\n[0.5] Identifier Partial Match: {len(identifier_partial_match)}")
        for log in identifier_partial_match:
            print(log)
    
    if filter_mode == "all" or filter_mode == "disagree":
        print(f"\n[0.0] Identifier Mismatch: {len(identifier_mismatch)}")
        for log in identifier_mismatch:
            print(log)
    
def load_ner_file(file_path: str) -> Dict[str, Dict]:
    """
    Load NER file and return dict mapping data_id to document
    """
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                data_id = data.get("metadata", {}).get("data_id")
                if data_id:
                    documents[data_id] = data
    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inter-annotator agreement for NER labeling"
    )
    parser.add_argument(
        "--file_a",
        type=str,
        required=True,
        help="Path to first annotator's NER file (e.g., PANORAMA_NER_a4_removed.jsonl)"
    )
    parser.add_argument(
        "--file_b",
        type=str,
        required=True,
        help="Path to second annotator's NER file (e.g., PANORAMA_NER_a5_removed.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated)"
    )
    parser.add_argument(
        "--case_log_filter",
        type=str,
        default="disagree",
        choices=["all", "disagree", "mismatch"],
        help="Filter for case logs"
    )
    parser.add_argument(
        "--show_case_logs",
        action="store_true",
        default=True,
        help="Show case logs"
    )
    
    args = parser.parse_args()
    
    # Load documents
    print(f"Loading documents from {args.file_a}...")
    docs_a = load_ner_file(args.file_a)
    print(f"  Loaded {len(docs_a)} documents")
    
    print(f"Loading documents from {args.file_b}...")
    docs_b = load_ner_file(args.file_b)
    print(f"  Loaded {len(docs_b)} documents")
    
    # Find common data_ids
    common_ids = set(docs_a.keys()) & set(docs_b.keys())
    only_a = set(docs_a.keys()) - set(docs_b.keys())
    only_b = set(docs_b.keys()) - set(docs_a.keys())
    
    print(f"\nDocument overlap:")
    print(f"  Common documents: {len(common_ids)}")
    print(f"  Only in file A: {len(only_a)}")
    print(f"  Only in file B: {len(only_b)}")
    
    if only_a:
        print(f"\n  Documents only in file A: {sorted(list(only_a))[:10]}")
        if len(only_a) > 10:
            print(f"    ... and {len(only_a) - 10} more")
    
    if only_b:
        print(f"\n  Documents only in file B: {sorted(list(only_b))[:10]}")
        if len(only_b) > 10:
            print(f"    ... and {len(only_b) - 10} more")
    
    # Evaluate common documents
    print(f"\nEvaluating {len(common_ids)} common documents...")
    all_results = []
    
    for data_id in sorted(common_ids):
        doc_a = docs_a[data_id]
        doc_b = docs_b[data_id]
        result = evaluate_document_entities(doc_a, doc_b)
        all_results.append(result)
    
    # Calculate statistics
    statistics = calculate_statistics(all_results)
    
    # Print statistics
    print_statistics(statistics)
    
    # Build and print case logs
    if args.show_case_logs:
        case_logs = build_case_logs(all_results, filter_mode=args.case_log_filter)
        print_case_logs(case_logs, filter_mode=args.case_log_filter)
    
    # Save results
    if args.output is None:
        # Auto-generate output path
        base_name = os.path.splitext(os.path.basename(args.file_a))[0]
        output_dir = os.path.dirname(args.file_a)
        args.output = os.path.join(output_dir, f"{base_name}_inter_evaluation.json")
    
    output_data = {
        "config": {
            "file_a": args.file_a,
            "file_b": args.file_b,
            "case_log_filter": args.case_log_filter
        },
        "statistics": statistics,
        "detailed_results": all_results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

