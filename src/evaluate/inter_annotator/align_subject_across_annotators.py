"""
Align subjects across two different annotations using LLM analysis
"""

import json
import sys
import os
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import argparse

from src.common_utils import load_jsonl_data
from src.llm.client import get_alignment_analysis, set_api_provider
from src.llm.parser import parse_subject_alignment_response, validate_subject_matched_pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align subjects across two annotations using LLM analysis"
    )
    parser.add_argument("-d", "--data", type=str, required=True, choices=["tab", "panorama"],
                        help="Dataset to process")
    parser.add_argument("--api_provider", type=str, required=True,
                        choices=["anthropic", "openai", "ollama"],
                        help="API provider")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to use")
    parser.add_argument("--processing_mode", type=str, default="concurrent",
                        choices=["sequential", "concurrent"],
                        help="Processing mode")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of concurrent workers")
    parser.add_argument("--document_limit", type=int, default=None,
                        help="Limit number of documents to process (for testing)")
    return parser.parse_args()

# ============================================================
# Helper Functions
# ============================================================
def load_processed_ids(output_file):
    """Get IDs of already processed documents"""
    processed_ids = set()
    if os.path.exists(output_file):
        print(f"Found existing output: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    data_id = data.get('metadata', {}).get('data_id')
                    if data_id:
                        processed_ids.add(data_id)
                except json.JSONDecodeError:
                    continue
    return processed_ids


def save_processed_data(output_file, data):
    """Save successfully processed document"""
    with open(output_file, 'a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def print_error(data_id, error_type, error_msg):
    """Print error immediately"""
    print(f"\n[ERROR] {data_id} - {error_type}: {error_msg}")


# ============================================================
# Processing Functions
# ============================================================
def process_alignment(text, annotation_a, annotation_b, data_id):
    """
    Process alignment for a single document

    Returns:
        tuple: (status, matched_pairs, response, error, match_stats)
        match_stats contains: {
            'total_subjects_a': int,
            'total_subjects_b': int,
            'matched_count': int,
            'unmatched_a_count': int,
            'unmatched_b_count': int,
            'matched_pairs': [(a_idx, b_idx), ...],
            'unmatched_a_ids': [ids...],
            'unmatched_b_ids': [ids...]
        }
    """
    status, response = get_alignment_analysis(text, annotation_a, annotation_b)

    if status == "error":
        print_error(data_id, "ALIGNMENT_ERROR", response)
        return False, None, response, f"ALIGNMENT_ERROR: {response}", None

    status, matched_pairs = parse_subject_alignment_response(response)
    if status == "error":
        print_error(data_id, "PARSING_ERROR", matched_pairs)
        return False, None, response, f"PARSING_ERROR: {matched_pairs}", None

    status, validated_pairs = validate_subject_matched_pairs(matched_pairs, annotation_a, annotation_b)
    if status == "error":
        print_error(data_id, "VALIDATION_ERROR", validated_pairs)
        return False, None, response, f"VALIDATION_ERROR: {validated_pairs}", None

    # Calculate match statistics
    total_subjects_a = len(annotation_a)
    total_subjects_b = len(annotation_b)
    matched_count = len(validated_pairs)

    # Find unmatched subjects
    matched_a_indices = {pair[0] for pair in validated_pairs}
    matched_b_indices = {pair[1] for pair in validated_pairs}

    unmatched_a_ids = [annotation_a[i].get('id', i) for i in range(total_subjects_a) if i not in matched_a_indices]
    unmatched_b_ids = [annotation_b[i].get('id', i) for i in range(total_subjects_b) if i not in matched_b_indices]

    match_stats = {
        'total_subjects_a': total_subjects_a,
        'total_subjects_b': total_subjects_b,
        'matched_count': matched_count,
        'unmatched_a_count': len(unmatched_a_ids),
        'unmatched_b_count': len(unmatched_b_ids),
        'matched_pairs': validated_pairs,
        'unmatched_a_ids': unmatched_a_ids,
        'unmatched_b_ids': unmatched_b_ids
    }

    return True, validated_pairs, response, None, match_stats


def transform_to_aligned_format(annotation_a, annotation_b, matched_pairs):
    """
    Transform subjects into aligned format including unmatched subjects.

    Args:
        annotation_a: List of subjects from annotation A
        annotation_b: List of subjects from annotation B
        matched_pairs: List of (A_id, B_id) tuples

    Returns:
        List of aligned subjects with separate annotation_A and annotation_B PIIs.
        - Matched subjects: 'matched': True, both A and B have real PIIs
        - Unmatched A-only: 'matched': False, 'unmatched_side': 'A', B has dummy PIIs
        - Unmatched B-only: 'matched': False, 'unmatched_side': 'B', A has dummy PIIs
    """
    aligned_subjects = []

    # Find matched and unmatched indices
    matched_a_indices = {pair[0] for pair in matched_pairs}
    matched_b_indices = {pair[1] for pair in matched_pairs}

    # 1. Add matched subjects
    for a_id, b_id in matched_pairs:
        subject_a = annotation_a[a_id]
        subject_b = annotation_b[b_id]

        # Use description from annotation_A (or B if A is empty)
        description = subject_a.get('description', '') or subject_b.get('description', '')

        aligned_subject = {
            'id': len(aligned_subjects),
            'matched': True,
            'description': description,
            'PIIs': {
                'annotation_A': subject_a.get('PIIs', []),
                'annotation_B': subject_b.get('PIIs', [])
            }
        }
        aligned_subjects.append(aligned_subject)

    # 2. Add A-only subjects (unmatched in A)
    for a_idx in range(len(annotation_a)):
        if a_idx not in matched_a_indices:
            subject_a = annotation_a[a_idx]
            a_piis = subject_a.get('PIIs', [])

            # Create dummy PIIs for B side
            dummy_piis = [
                {
                    'keyword': '',
                    'tag': pii.get('tag', ''),
                    'hardness': 0,
                    'certainty': 0
                }
                for pii in a_piis
            ]

            aligned_subject = {
                'id': len(aligned_subjects),
                'matched': False,
                'unmatched_side': 'A',
                'description': subject_a.get('description', ''),
                'PIIs': {
                    'annotation_A': a_piis,
                    'annotation_B': dummy_piis
                }
            }
            aligned_subjects.append(aligned_subject)

    # 3. Add B-only subjects (unmatched in B)
    for b_idx in range(len(annotation_b)):
        if b_idx not in matched_b_indices:
            subject_b = annotation_b[b_idx]
            b_piis = subject_b.get('PIIs', [])

            # Create dummy PIIs for A side
            dummy_piis = [
                {
                    'keyword': '',
                    'tag': pii.get('tag', ''),
                    'hardness': 0,
                    'certainty': 0
                }
                for pii in b_piis
            ]

            aligned_subject = {
                'id': len(aligned_subjects),
                'matched': False,
                'unmatched_side': 'B',
                'description': subject_b.get('description', ''),
                'PIIs': {
                    'annotation_A': dummy_piis,
                    'annotation_B': b_piis
                }
            }
            aligned_subjects.append(aligned_subject)

    return aligned_subjects


# ============================================================
# Concurrent Processing
# ============================================================

# Thread-safe containers for concurrent processing
result_queue = queue.Queue()
stats_lock = threading.Lock()
concurrent_stats = {
    'processed': 0,
    'errors': 0,
    'matched_count': 0,
    'total_subjects': 0,
    'unmatched_count': 0,
    'unmatched_cases': []  # List of {data_id, unmatched_a, unmatched_b}
}


def process_document_concurrent(data_with_idx):
    """Process single document through alignment pipeline in concurrent mode"""
    data, idx = data_with_idx
    data_id = data.get('metadata', {}).get('data_id', f'line_{idx}')

    try:
        text = data['text']
        annotation_a = data.get('annotation_A', [])
        annotation_b = data.get('annotation_B', [])

        # Validate annotations exist
        if not annotation_a or not annotation_b:
            error_msg = f"Missing annotations: A={len(annotation_a)}, B={len(annotation_b)}"
            print_error(data_id, "missing_annotations", error_msg)
            return {
                'data_id': data_id,
                'data': data,
                'status': 'error',
                'error_type': 'missing_annotations',
                'error': error_msg,
                'response': None
            }

        # Process alignment
        success, matched_pairs, response, error, match_stats = process_alignment(
            text, annotation_a, annotation_b, data_id
        )

        if not success:
            return {
                'data_id': data_id,
                'data': data,
                'status': 'error',
                'error_type': 'alignment',
                'error': error,
                'response': response,
                'match_stats': None
            }

        # Transform to aligned format
        aligned_subjects = transform_to_aligned_format(annotation_a, annotation_b, matched_pairs)

        # Prepare result data with cross_labeled flag
        metadata = data.get('metadata', {})
        metadata['cross_labeled'] = True

        result_data = {
            'metadata': metadata,
            'text': text,
            'subjects': aligned_subjects
        }

        return {
            'data_id': data_id,
            'data': result_data,
            'status': 'success',
            'matched_count': len(matched_pairs),
            'response': response,
            'match_stats': match_stats
        }

    except Exception as e:
        return {
            'data_id': data_id,
            'data': data,
            'status': 'error',
            'error_type': 'exception',
            'error': str(e),
            'exception': True,
            'response': None
        }


def file_writer_worker(output_file, skipped_count, dataset_size):
    """Dedicated file writer thread for concurrent processing"""
    while True:
        try:
            result = result_queue.get(timeout=10)
            if result is None:
                break

            data_id = result['data_id']
            data = result['data']
            status = result['status']

            if status == 'error':
                with stats_lock:
                    concurrent_stats['errors'] += 1
            else:
                # Save successful result
                save_processed_data(output_file, data)
                match_stats = result.get('match_stats')

                with stats_lock:
                    concurrent_stats['processed'] += 1
                    concurrent_stats['matched_count'] += result.get('matched_count', 0)

                    if match_stats:
                        total_subj = match_stats['total_subjects_a'] + match_stats['total_subjects_b']
                        concurrent_stats['total_subjects'] += total_subj
                        unmatched = match_stats['unmatched_a_count'] + match_stats['unmatched_b_count']
                        concurrent_stats['unmatched_count'] += unmatched

                        # Record unmatched case if any
                        if unmatched > 0:
                            concurrent_stats['unmatched_cases'].append({
                                'data_id': data_id,
                                'unmatched_a_ids': match_stats['unmatched_a_ids'],
                                'unmatched_b_ids': match_stats['unmatched_b_ids'],
                                'unmatched_a_count': match_stats['unmatched_a_count'],
                                'unmatched_b_count': match_stats['unmatched_b_count']
                            })

            with stats_lock:
                total_done = concurrent_stats['processed'] + concurrent_stats['errors']
                if total_done % 10 == 0:
                    current_total = total_done + skipped_count
                    avg_matched = (concurrent_stats['matched_count'] / concurrent_stats['processed']
                                   if concurrent_stats['processed'] > 0 else 0)
                    print(f"Progress: {current_total}/{dataset_size} "
                          f"(Success: {concurrent_stats['processed']}, Error: {concurrent_stats['errors']}, "
                          f"Avg Matched: {avg_matched:.1f}, Skipped: {skipped_count})")

            result_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"File writer error: {e}")
            continue


def process_concurrent(unprocessed_docs, output_file, skipped_count, dataset_size, max_workers):
    """Process documents concurrently"""
    writer_thread = threading.Thread(
        target=file_writer_worker,
        args=(output_file, skipped_count, dataset_size)
    )
    writer_thread.daemon = True
    writer_thread.start()

    print(f"Starting concurrent processing of {len(unprocessed_docs)} documents with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(process_document_concurrent, doc_data): doc_data
            for doc_data in unprocessed_docs
        }

        for future in tqdm(as_completed(future_to_doc),
                          total=len(unprocessed_docs),
                          desc="Processing documents"):
            try:
                result = future.result()
                result_queue.put(result)
            except Exception as e:
                doc_data = future_to_doc[future]
                data_id = doc_data[0].get('metadata', {}).get('data_id', f'line_{doc_data[1]}')
                print(f"Error processing document {data_id}: {e}")

    result_queue.join()
    result_queue.put(None)
    writer_thread.join(timeout=5)

    return (concurrent_stats['processed'], concurrent_stats['errors'],
            concurrent_stats['matched_count'], concurrent_stats['unmatched_count'],
            concurrent_stats['total_subjects'], concurrent_stats['unmatched_cases'])


# ============================================================
# Sequential Processing
# ============================================================

def process_sequential(dataset, output_file, processed_ids):
    """Process documents sequentially"""
    stats = {
        'processed': 0,
        'errors': 0,
        'skipped': len(processed_ids),
        'matched_count': 0,
        'unmatched_count': 0,
        'total_subjects': 0,
        'unmatched_cases': []
    }

    for idx, data in enumerate(tqdm(dataset, desc="Processing")):
        data_id = data.get('metadata', {}).get('data_id', f'line_{idx}')

        # Skip if processed
        if data_id in processed_ids:
            continue

        text = data['text']
        annotation_a = data.get('annotation_A', [])
        annotation_b = data.get('annotation_B', [])

        # Validate annotations exist
        if not annotation_a or not annotation_b:
            error_msg = f"Missing annotations: A={len(annotation_a)}, B={len(annotation_b)}"
            print_error(data_id, "missing_annotations", error_msg)
            stats['errors'] += 1
            continue

        # Process alignment
        success, matched_pairs, response, error, match_stats = process_alignment(
            text, annotation_a, annotation_b, data_id
        )

        if not success:
            stats['errors'] += 1
            continue

        # Transform to aligned format
        aligned_subjects = transform_to_aligned_format(annotation_a, annotation_b, matched_pairs)

        # Save result with cross_labeled flag
        metadata = data.get('metadata', {})
        metadata['cross_labeled'] = True

        result_data = {
            'metadata': metadata,
            'text': text,
            'subjects': aligned_subjects
        }

        save_processed_data(output_file, result_data)
        stats['processed'] += 1
        stats['matched_count'] += len(matched_pairs)

        # Update match statistics
        if match_stats:
            total_subj = match_stats['total_subjects_a'] + match_stats['total_subjects_b']
            stats['total_subjects'] += total_subj
            unmatched = match_stats['unmatched_a_count'] + match_stats['unmatched_b_count']
            stats['unmatched_count'] += unmatched

            # Record unmatched case if any
            if unmatched > 0:
                stats['unmatched_cases'].append({
                    'data_id': data_id,
                    'unmatched_a_ids': match_stats['unmatched_a_ids'],
                    'unmatched_b_ids': match_stats['unmatched_b_ids'],
                    'unmatched_a_count': match_stats['unmatched_a_count'],
                    'unmatched_b_count': match_stats['unmatched_b_count']
                })

        # Progress update
        if (stats['processed'] + stats['errors']) % 10 == 0:
            total = stats['processed'] + stats['errors'] + stats['skipped']
            avg_matched = stats['matched_count'] / stats['processed'] if stats['processed'] > 0 else 0
            print(f"Progress: {total}/{len(dataset)}, Avg Matched: {avg_matched:.1f}")

    return (stats['processed'], stats['errors'], stats['matched_count'],
            stats['unmatched_count'], stats['total_subjects'], stats['unmatched_cases'])


# ============================================================
# Main Pipeline
# ============================================================
def main():
    args = parse_args()
    data = args.data
    model = args.model
    api_provider = args.api_provider
    processing_mode = args.processing_mode
    max_workers = args.num_workers
    document_limit = args.document_limit

    INPUT_FILE = f"./data/spia/inter_annotator/{data}/{data}_inter_cross_aligned.jsonl"
    BASE_NAME = f"{data}_inter"

    # Initialize API
    try:
        set_api_provider(api_provider, model)
        print(f"Using {api_provider} API with model: {model}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load data
    dataset = load_jsonl_data(INPUT_FILE)
    if document_limit:
        dataset = dataset[:document_limit]
    dataset_size = len(dataset)

    # Output file
    output_file = f'data/spia/inter_annotator/{data}/{BASE_NAME}_aligned_{api_provider}_{model}.jsonl'

    # Remove existing output file to reprocess all documents
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing output file: {output_file}")

    # Start with empty processed IDs (reprocess all)
    processed_ids = set()

    print(f"Processing {dataset_size} documents (fresh start)")
    print(f"Processing mode: {processing_mode.upper()}")

    if len(processed_ids) == dataset_size:
        print("All documents already processed!")
        return

    start_time = time.time()

    if processing_mode == 'sequential':
        (total_processed, total_errors, total_matched,
         total_unmatched, total_subjects, unmatched_cases) = process_sequential(
            dataset, output_file, processed_ids
        )
    else:  # concurrent
        unprocessed_docs = [
            (data_item, idx) for idx, data_item in enumerate(dataset)
            if data_item.get('metadata', {}).get('data_id', f'line_{idx}') not in processed_ids
        ]

        if not unprocessed_docs:
            print("All documents already processed!")
            return

        (total_processed, total_errors, total_matched,
         total_unmatched, total_subjects, unmatched_cases) = process_concurrent(
            unprocessed_docs, output_file,
            len(processed_ids), dataset_size, max_workers
        )

    end_time = time.time()
    processing_time = end_time - start_time

    # Calculate statistics
    # Denominator should be matched + unmatched (not A + B which double counts matched subjects)
    total_unique_subjects = total_matched + total_unmatched
    match_ratio = total_matched / total_unique_subjects if total_unique_subjects > 0 else 0
    unmatch_ratio = total_unmatched / total_unique_subjects if total_unique_subjects > 0 else 0

    # Print summary
    print(f"\n{'='*50}")
    print(f"Alignment Processing Complete!")
    print(f"{'='*50}")
    print(f"Mode: {processing_mode.upper()}")
    if processing_mode == 'concurrent':
        print(f"Workers: {max_workers}")
    print(f"Model: {model}")
    print(f"Output: {output_file}")
    print(f"{'='*50}")
    print(f"Total documents: {dataset_size}")
    print(f"Newly processed: {total_processed + total_errors}")
    print(f"Success: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"{'='*50}")
    print(f"Total subjects in annotations (A + B): {total_subjects}")
    print(f"Total unique subjects: {total_unique_subjects}")
    print(f"Matched subjects: {total_matched}")
    print(f"Unmatched subjects: {total_unmatched}")
    print(f"Match ratio: {match_ratio:.2%} (matched / unique)")
    print(f"Unmatch ratio: {unmatch_ratio:.2%} (unmatched / unique)")
    if total_processed > 0:
        print(f"Average matched per document: {total_matched/total_processed:.2f}")
    print(f"Documents with unmatched subjects: {len(unmatched_cases)}")
    print(f"{'='*50}")
    print(f"Processing time: {processing_time:.2f} seconds")
    if total_processed + total_errors > 0:
        print(f"Average time per doc: {processing_time/(total_processed + total_errors):.2f} seconds")


if __name__ == "__main__":
    main()
