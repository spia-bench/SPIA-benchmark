"""
TAB Dataset Entity Recall Evaluation Script
Calculates ERdi (Direct Identifier Entity Recall) and ERqi (Quasi Identifier Entity Recall)

Based on the evaluation methodology from evaluation.py

Usage:
    # Single file evaluation (command line)
    python src/evaluate/privacy/evaluate_recall.py --gt_file data/entity/tab_144_gt.jsonl --anonymized_file data/spia/anonymized/tab/tab_144_adversarial.jsonl

    # Batch evaluation via config
    python src/evaluate/privacy/run_evaluate_recall_experiments.py
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import spacy


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
        info["model"] = "claude-sonnet-4"
    elif "claude-3-5-haiku" in name or "claude_haiku" in name or "claude-haiku" in name:
        info["model"] = "claude-3-5-haiku"
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


# POS tags, tokens or characters that can be ignored from the recall scores
# (because they do not carry much semantic content, and there are discrepancies
# on whether to include them in the annotated spans or not)
# Same as evaluation.py
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"}
TOKENS_TO_IGNORE = {"mr", "mrs", "ms", "no", "nr", "about"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" '"""


# Global spacy model (loaded once)
_nlp = None


def get_spacy_model(model_name: str = "en_core_web_md"):
    """Load spacy model (singleton pattern)"""
    global _nlp
    if _nlp is None:
        print(f"Loading spaCy model: {model_name}")
        _nlp = spacy.load(model_name)
    return _nlp


@dataclass
class Entity:
    """Represents an entity with its mentions"""
    entity_id: str
    mentions: List[Tuple[int, int]]  # List of (start, end) spans
    mention_level_masking: List[bool]  # Per-mention masking requirement
    identifier_type: str  # DIRECT, QUASI, NO_MASK (entity-level, derived from mentions)
    entity_type: str

    @property
    def need_masking(self) -> bool:
        # Entity needs masking if ANY mention needs masking
        return any(self.mention_level_masking)

    @property
    def is_direct(self) -> bool:
        return self.identifier_type.upper() == "DIRECT"

    @property
    def is_quasi(self) -> bool:
        return self.identifier_type.upper() == "QUASI"

    @property
    def mentions_to_mask(self) -> List[Tuple[int, int]]:
        """Returns only mentions that need masking"""
        return [m for m, need in zip(self.mentions, self.mention_level_masking) if need]


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def get_entities_from_mentions(entity_mentions: List[Dict]) -> Dict[str, Entity]:
    """Group entity mentions by entity_id to create Entity objects"""
    entities = {}

    for mention in entity_mentions:
        entity_id = mention.get("entity_id")
        if not entity_id:
            continue

        start = mention.get("start_offset")
        end = mention.get("end_offset")
        identifier_type = mention.get("identifier_type", "NO_MASK")
        entity_type = mention.get("entity_type", "UNKNOWN")
        mention_need_masking = identifier_type.upper() in ["DIRECT", "QUASI"]

        if entity_id in entities:
            # Add new mention to existing entity
            entities[entity_id].mentions.append((start, end))
            entities[entity_id].mention_level_masking.append(mention_need_masking)
            # Update entity-level identifier_type if this mention needs masking
            if mention_need_masking and entities[entity_id].identifier_type.upper() == "NO_MASK":
                entities[entity_id].identifier_type = identifier_type
        else:
            # Create new entity
            entities[entity_id] = Entity(
                entity_id=entity_id,
                mentions=[(start, end)],
                mention_level_masking=[mention_need_masking],
                identifier_type=identifier_type,
                entity_type=entity_type
            )

    return entities


def get_masked_offsets_from_anonymized_text(original_text: str, anonymized_text: str, entities: List[Dict]) -> Set[int]:
    """
    Extract masked character offsets by finding where entity placeholders appear in anonymized text.
    Uses the entities list from anonymized file to get the masked spans.
    """
    masked_offsets = set()

    for entity in entities:
        start = entity.get("start_offset")
        end = entity.get("end_offset")
        if start is not None and end is not None:
            masked_offsets.update(range(start, end))

    return masked_offsets


def generate_entities_by_span_search(
    original_text: str,
    anonymized_text: str,
    gt_entities: List[Dict]
) -> List[Dict]:
    """
    Generate anonymized entities by searching GT entity span_text in anonymized_text.

    Method: For each GT entity, check if its span_text exists in anonymized_text.
    If NOT found, the entity is considered masked (add to result).
    If found, the entity is NOT masked (skip).

    Returns: List of entities that were masked (span_text not found in anonymized_text)
    """
    masked_entities = []

    for entity in gt_entities:
        span_text = entity.get("span_text", "")
        if not span_text:
            continue

        # If span_text is NOT found in anonymized_text, it was masked
        if span_text not in anonymized_text:
            masked_entities.append(entity)

    return masked_entities


def generate_entities_by_offset_comparison(
    original_text: str,
    anonymized_text: str,
    gt_entities: List[Dict]
) -> List[Dict]:
    """
    Generate anonymized entities by comparing text at same offsets.

    Method: For each GT entity, extract text at (start_offset, end_offset) from both
    original_text and anonymized_text. If they differ, the entity was masked.

    Note: This method works best when anonymized_text has similar structure to original.
    If anonymized_text has different length, offsets may not align properly.

    Returns: List of entities that were masked (text differs at offset positions)
    """
    masked_entities = []

    for entity in gt_entities:
        start = entity.get("start_offset")
        end = entity.get("end_offset")
        span_text = entity.get("span_text", "")

        if start is None or end is None:
            continue

        # Extract text at same offset from anonymized_text
        if end <= len(anonymized_text):
            anon_span = anonymized_text[start:end]
            # If different, entity was masked
            if anon_span != span_text:
                masked_entities.append(entity)
        else:
            # If anonymized_text is shorter, assume masked
            masked_entities.append(entity)

    return masked_entities


def get_masked_offsets_from_generated_entities(
    original_text: str,
    anonymized_text: str,
    gt_entities: List[Dict],
    method: str = "span_search"
) -> Set[int]:
    """
    Get masked offsets when anonymized file doesn't have entities.
    Uses GT entities and comparison method to determine which entities were masked.

    Args:
        original_text: Original text from GT file
        anonymized_text: Anonymized text from anonymized file
        gt_entities: Entities from GT file
        method: "span_search" or "offset_comparison"

    Returns:
        Set of character offsets that were masked
    """
    if method == "span_search":
        masked_entities = generate_entities_by_span_search(
            original_text, anonymized_text, gt_entities
        )
    elif method == "offset_comparison":
        masked_entities = generate_entities_by_offset_comparison(
            original_text, anonymized_text, gt_entities
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'span_search' or 'offset_comparison'")

    masked_offsets = set()
    for entity in masked_entities:
        start = entity.get("start_offset")
        end = entity.get("end_offset")
        if start is not None and end is not None:
            masked_offsets.update(range(start, end))

    return masked_offsets


def is_mention_masked(mention_start: int, mention_end: int,
                      masked_offsets: Set[int], text: str,
                      spacy_doc: spacy.tokens.Doc) -> bool:
    """
    Check if a mention is fully masked.
    A mention is considered masked if all its meaningful characters are covered.

    Same logic as evaluation.py:is_mention_masked (Lines 436-461)
    """
    # Computes the character offsets that must be masked
    offsets_to_mask = set(range(mention_start, mention_end))

    # Build the set of character offsets that are not covered
    non_covered_offsets = offsets_to_mask - masked_offsets

    # If we have not covered everything, we also make sure punctuations,
    # spaces, titles, etc. are ignored
    if len(non_covered_offsets) > 0:
        span = spacy_doc.char_span(mention_start, mention_end, alignment_mode="expand")
        if span is not None:
            for token in span:
                if token.pos_ in POS_TO_IGNORE or token.lower_ in TOKENS_TO_IGNORE:
                    non_covered_offsets -= set(range(token.idx, token.idx + len(token)))

    # Remove ignorable characters
    for i in list(non_covered_offsets):
        if i < len(text) and text[i] in CHARACTERS_TO_IGNORE:
            non_covered_offsets.discard(i)

    # If that set is empty, we consider the mention as properly masked
    return len(non_covered_offsets) == 0


def is_entity_masked(entity: Entity, masked_offsets: Set[int], text: str,
                     spacy_doc: spacy.tokens.Doc) -> bool:
    """
    Check if an entity is fully masked (all its mentions that need masking are masked).
    Only checks mentions where mention_level_masking is True.
    """
    for i, (mention_start, mention_end) in enumerate(entity.mentions):
        # Skip mentions that don't need masking
        if not entity.mention_level_masking[i]:
            continue
        if not is_mention_masked(mention_start, mention_end, masked_offsets, text, spacy_doc):
            return False
    return True


def split_by_tokens(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """
    Generates the (start, end) boundaries of each token included in this span.
    Same logic as evaluation.py:split_by_tokens (Lines 508-514)
    """
    tokens = []
    for match in re.finditer(r"\w+", text[start:end]):
        start_token = start + match.start(0)
        end_token = start + match.end(0)
        tokens.append((start_token, end_token))
    return tokens


def is_span_covered_by_gt(span_start: int, span_end: int,
                          gt_entities: Dict[str, Entity]) -> bool:
    """
    Check if a system-masked span is covered by any GT entity requiring masking.
    A span is considered a true positive if it's fully contained within a GT mention
    that needs masking (mention_level_masking is True).
    """
    for entity in gt_entities.values():
        if not entity.need_masking:
            continue
        for i, (mention_start, mention_end) in enumerate(entity.mentions):
            # Only consider mentions that need masking
            if not entity.mention_level_masking[i]:
                continue
            # Check if system span is fully covered by GT mention
            if mention_start <= span_start and mention_end >= span_end:
                return True
    return False


def calculate_recall(gt_docs: List[Dict], anon_docs: List[Dict],
                     spacy_model: str = "en_core_web_md",
                     entity_detection_method: str = "auto") -> Dict:
    """
    Calculate ERdi, ERqi, Token Recall, and Token Precision from ground truth and anonymized documents.

    Args:
        gt_docs: Ground truth documents with entities
        anon_docs: Anonymized documents (may or may not have entities)
        spacy_model: spaCy model name for text processing
        entity_detection_method: Method to detect masked entities when anonymized file has no entities
            - "auto": Use entities if available, otherwise use "span_search"
            - "entities": Use entities from anonymized file (original behavior)
            - "span_search": Search GT span_text in anonymized_text
            - "offset_comparison": Compare text at same offsets

    Returns:
        Dictionary with ERdi, ERqi, token_recall, token_precision and detailed counts
    """
    # Load spacy model
    nlp = get_spacy_model(spacy_model)

    # Create lookup for anonymized docs by data_id
    anon_lookup = {}
    for doc in anon_docs:
        data_id = doc.get("metadata", {}).get("data_id")
        if data_id:
            anon_lookup[data_id] = doc

    # Entity-level counters
    direct_total = 0
    direct_masked = 0
    quasi_total = 0
    quasi_masked = 0

    # Entity-level counters by entity type
    direct_masked_by_type = defaultdict(int)
    direct_total_by_type = defaultdict(int)
    quasi_masked_by_type = defaultdict(int)
    quasi_total_by_type = defaultdict(int)

    # Token-level counters (by entity type)
    token_masked_by_type = defaultdict(int)
    token_total_by_type = defaultdict(int)

    # Precision counters (token-level)
    precision_tp = 0
    precision_system_total = 0

    # Track unmatched docs
    unmatched = []

    for gt_doc in gt_docs:
        data_id = gt_doc.get("metadata", {}).get("data_id")
        if not data_id:
            continue

        anon_doc = anon_lookup.get(data_id)
        if not anon_doc:
            unmatched.append(data_id)
            continue

        # Get original text
        original_text = gt_doc.get("text", "")

        # Parse text with spacy
        spacy_doc = nlp(original_text)

        # Build entities from GT
        gt_entities = get_entities_from_mentions(gt_doc.get("entities", []))

        # Get masked offsets based on detection method
        anon_entities = anon_doc.get("entities", [])
        anonymized_text = anon_doc.get("anonymized_text", "")
        gt_entity_list = gt_doc.get("entities", [])

        # Determine which method to use
        if entity_detection_method == "auto":
            # Use entities if available, otherwise use span_search
            use_method = "entities" if anon_entities else "span_search"
        else:
            use_method = entity_detection_method

        if use_method == "entities":
            # Original behavior: use entities from anonymized file
            masked_offsets = get_masked_offsets_from_anonymized_text(
                original_text, anonymized_text, anon_entities
            )
        else:
            # Use generated entities based on comparison method
            masked_offsets = get_masked_offsets_from_generated_entities(
                original_text, anonymized_text, gt_entity_list, method=use_method
            )

        # Check each GT entity
        for _, entity in gt_entities.items():
            if not entity.need_masking:
                continue

            # Entity-level recall (ERdi, ERqi)
            is_masked = is_entity_masked(entity, masked_offsets, original_text, spacy_doc)

            if entity.is_direct:
                direct_total += 1
                direct_total_by_type[entity.entity_type] += 1
                if is_masked:
                    direct_masked += 1
                    direct_masked_by_type[entity.entity_type] += 1
            elif entity.is_quasi:
                quasi_total += 1
                quasi_total_by_type[entity.entity_type] += 1
                if is_masked:
                    quasi_masked += 1
                    quasi_masked_by_type[entity.entity_type] += 1

            # Token-level recall (only for mentions that need masking)
            for i, (mention_start, mention_end) in enumerate(entity.mentions):
                if not entity.mention_level_masking[i]:
                    continue
                tokens = split_by_tokens(original_text, mention_start, mention_end)
                for token_start, token_end in tokens:
                    token_total_by_type[entity.entity_type] += 1
                    if is_mention_masked(token_start, token_end, masked_offsets, original_text, spacy_doc):
                        token_masked_by_type[entity.entity_type] += 1

        # Token-level precision: check each system-masked span
        # Split system masks into tokens and check if they match GT
        for anon_entity in anon_entities:
            start = anon_entity.get("start_offset")
            end = anon_entity.get("end_offset")
            if start is not None and end is not None:
                system_tokens = split_by_tokens(original_text, start, end)
                for token_start, token_end in system_tokens:
                    precision_system_total += 1
                    if is_span_covered_by_gt(token_start, token_end, gt_entities):
                        precision_tp += 1

    # Calculate entity-level recalls
    ERdi = direct_masked / direct_total if direct_total > 0 else 0.0
    ERqi = quasi_masked / quasi_total if quasi_total > 0 else 0.0

    # Calculate token-level recall
    total_tokens = sum(token_total_by_type.values())
    masked_tokens = sum(token_masked_by_type.values())
    token_recall = masked_tokens / total_tokens if total_tokens > 0 else 0.0

    # Calculate token-level precision
    # Note: precision is only meaningful when using "entities" method
    # For span_search/offset_comparison, we can only check GT entities, so precision would always be 1.0
    if precision_system_total > 0:
        token_precision = precision_tp / precision_system_total
    else:
        token_precision = None  # Cannot calculate precision without anonymized entities

    # Token recall per entity type
    token_recall_by_type = {}
    for entity_type in token_total_by_type:
        if token_total_by_type[entity_type] > 0:
            token_recall_by_type[entity_type] = token_masked_by_type[entity_type] / token_total_by_type[entity_type]
        else:
            token_recall_by_type[entity_type] = 0.0

    # ERdi per entity type
    erdi_by_type = {}
    for entity_type in direct_total_by_type:
        if direct_total_by_type[entity_type] > 0:
            erdi_by_type[entity_type] = direct_masked_by_type[entity_type] / direct_total_by_type[entity_type]
        else:
            erdi_by_type[entity_type] = 0.0

    # ERqi per entity type
    erqi_by_type = {}
    for entity_type in quasi_total_by_type:
        if quasi_total_by_type[entity_type] > 0:
            erqi_by_type[entity_type] = quasi_masked_by_type[entity_type] / quasi_total_by_type[entity_type]
        else:
            erqi_by_type[entity_type] = 0.0

    return {
        "ERdi": ERdi,
        "ERqi": ERqi,
        "token_recall": token_recall,
        "token_precision": token_precision,
        "token_recall_by_type": token_recall_by_type,
        "erdi_by_type": erdi_by_type,
        "erqi_by_type": erqi_by_type,
        "DIRECT": {
            "tp": direct_masked,
            "total": direct_total,
            "by_type": {k: {"tp": direct_masked_by_type[k], "total": direct_total_by_type[k]} for k in direct_total_by_type}
        },
        "QUASI": {
            "tp": quasi_masked,
            "total": quasi_total,
            "by_type": {k: {"tp": quasi_masked_by_type[k], "total": quasi_total_by_type[k]} for k in quasi_total_by_type}
        },
        "TOKEN_RECALL": {
            "tp": masked_tokens,
            "total": total_tokens
        },
        "TOKEN_PRECISION": {
            "tp": precision_tp,
            "total": precision_system_total
        },
        "unmatched_docs": unmatched
    }


def print_detailed_results(gt_docs: List[Dict], anon_docs: List[Dict],
                           spacy_model: str = "en_core_web_md", max_examples: int = 5):
    """Print detailed analysis of missed entities"""
    nlp = get_spacy_model(spacy_model)
    anon_lookup = {doc.get("metadata", {}).get("data_id"): doc for doc in anon_docs}

    missed_direct = []
    missed_quasi = []

    for gt_doc in gt_docs:
        data_id = gt_doc.get("metadata", {}).get("data_id")
        anon_doc = anon_lookup.get(data_id)
        if not anon_doc:
            continue

        original_text = gt_doc.get("text", "")
        spacy_doc = nlp(original_text)
        gt_entities = get_entities_from_mentions(gt_doc.get("entities", []))
        anon_entities = anon_doc.get("entities", [])
        masked_offsets = get_masked_offsets_from_anonymized_text(
            original_text,
            anon_doc.get("anonymized_text", ""),
            anon_entities
        )

        for entity_id, entity in gt_entities.items():
            if not entity.need_masking:
                continue

            if not is_entity_masked(entity, masked_offsets, original_text, spacy_doc):
                # Find which mentions were missed
                missed_mentions = []
                for start, end in entity.mentions:
                    if not is_mention_masked(start, end, masked_offsets, original_text, spacy_doc):
                        mention_text = original_text[start:end] if end <= len(original_text) else "[OUT OF RANGE]"
                        missed_mentions.append(f"'{mention_text}' ({start}-{end})")

                info = {
                    "data_id": data_id,
                    "entity_id": entity_id,
                    "entity_type": entity.entity_type,
                    "missed_mentions": missed_mentions
                }

                if entity.is_direct:
                    missed_direct.append(info)
                elif entity.is_quasi:
                    missed_quasi.append(info)

    print(f"\n=== Missed DIRECT Entities (showing first {max_examples}) ===")
    for info in missed_direct[:max_examples]:
        print(f"  Doc: {info['data_id']}, Entity: {info['entity_id']} ({info['entity_type']})")
        for mention in info['missed_mentions']:
            print(f"    - {mention}")
    if len(missed_direct) > max_examples:
        print(f"  ... and {len(missed_direct) - max_examples} more")

    print(f"\n=== Missed QUASI Entities (showing first {max_examples}) ===")
    for info in missed_quasi[:max_examples]:
        print(f"  Doc: {info['data_id']}, Entity: {info['entity_id']} ({info['entity_type']})")
        for mention in info['missed_mentions']:
            print(f"    - {mention}")
    if len(missed_quasi) > max_examples:
        print(f"  ... and {len(missed_quasi) - max_examples} more")


def detect_dataset_type(input_file: str) -> str:
    """Detect dataset type from input file path."""
    if "/PANORAMA/" in input_file:
        return "PANORAMA"
    elif "/TAB/" in input_file:
        return "TAB"
    else:
        raise ValueError(f"Cannot detect dataset type from path: {input_file}. "
                        "Path must contain '/TAB/' or '/PANORAMA/'.")


def validate_dataset_consistency(gt_file: str, anonymized_file: str) -> str:
    """
    Validate that gt_file and anonymized_file are from the same dataset.
    Returns the dataset type.
    """
    gt_type = detect_dataset_type(gt_file)
    anon_type = detect_dataset_type(anonymized_file)

    if gt_type != anon_type:
        raise ValueError(f"Dataset mismatch: gt_file is {gt_type} but anonymized_file is {anon_type}")

    return gt_type


def save_results_to_json(result: Dict, gt_file: str, anonymized_file: str,
                         entity_detection_method: str, dataset_type: str) -> str:
    """
    Save evaluation results to JSON file.
    Returns the output file path.

    Filename format: recall_{dataset}_{size}_{baseline}_{model}_{timestamp}.json
    """
    # Parse anonymized filename for components
    file_info = parse_anonymized_filename(anonymized_file)

    # Get output directory
    output_dir = get_output_directory(dataset_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate standardized filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = generate_output_filename(
        eval_type="recall",
        dataset=file_info["dataset"],
        dataset_size=file_info["dataset_size"],
        baseline=file_info["baseline"],
        model=file_info["model"],
        timestamp=timestamp
    )
    output_path = output_dir / output_filename

    # Build output data
    output_data = {
        "metadata": {
            "gt_file": gt_file,
            "anonymized_file": anonymized_file,
            "entity_detection_method": entity_detection_method,
            "dataset": dataset_type,
            "dataset_size": file_info["dataset_size"],
            "baseline": file_info["baseline"],
            "model": file_info["model"],
            "evaluation_timestamp": timestamp
        },
        "results": result
    }

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return str(output_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate privacy recall metrics (ERdi, ERqi, Token Recall) for anonymized files"
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        help="Path to ground truth JSONL file",
    )
    parser.add_argument(
        "--anonymized_file",
        type=str,
        help="Path to anonymized JSONL file",
    )
    parser.add_argument(
        "--entity_detection_method",
        type=str,
        default="auto",
        choices=["auto", "entities", "span_search", "offset_comparison"],
        help="Method to detect masked entities (default: auto)",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_md",
        help="spaCy model to use (default: en_core_web_md)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed missed entities",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # =========================================================================
    # Configuration for direct script execution (modify these variables)
    # =========================================================================
    DEFAULT_GT_FILE = "data/entity/panorama_151_gt.jsonl"
    DEFAULT_ANONYMIZED_FILE = "data/spia/anonymized/panorama/panorama_151_deid_gpt.jsonl"
    # Entity detection method:
    # - "auto": Use entities if available, otherwise use "span_search" (default)
    # - "entities": Use entities from anonymized file (original behavior)
    # - "span_search": Search GT span_text in anonymized_text (for files without entities)
    # - "offset_comparison": Compare text at same offsets (for files without entities)
    DEFAULT_ENTITY_DETECTION_METHOD = "auto"
    DEFAULT_SPACY_MODEL = "en_core_web_md"
    DEFAULT_DETAILED = False  # Show detailed missed entities
    # =========================================================================

    # Use command line args if provided, otherwise use defaults above
    gt_file = args.gt_file if args.gt_file else DEFAULT_GT_FILE
    anonymized_file = args.anonymized_file if args.anonymized_file else DEFAULT_ANONYMIZED_FILE
    entity_detection_method = args.entity_detection_method if args.entity_detection_method != "auto" else DEFAULT_ENTITY_DETECTION_METHOD
    spacy_model = args.spacy_model if args.spacy_model != "en_core_web_md" else DEFAULT_SPACY_MODEL
    show_detailed = args.detailed or DEFAULT_DETAILED

    # Validate dataset consistency
    dataset_type = validate_dataset_consistency(gt_file, anonymized_file)

    print(f"GT: {gt_file}")
    print(f"Anonymized: {anonymized_file}")
    print(f"Dataset Type: {dataset_type}")
    print(f"Entity Detection Method: {entity_detection_method}")

    gt_docs = load_jsonl(gt_file)
    anon_docs = load_jsonl(anonymized_file)
    print(f"Loaded {len(gt_docs)} GT docs, {len(anon_docs)} anonymized docs")

    result = calculate_recall(
        gt_docs, anon_docs,
        spacy_model=spacy_model,
        entity_detection_method=entity_detection_method
    )

    print(f"\n=== Entity-Level Recall ===")
    print(f"ERdi: {result['ERdi']:.4f} ({result['DIRECT']['tp']}/{result['DIRECT']['total']})")
    print(f"ERqi: {result['ERqi']:.4f} ({result['QUASI']['tp']}/{result['QUASI']['total']})")

    if result['erdi_by_type']:
        print(f"\nERdi by Entity Type:")
        for entity_type, recall in sorted(result['erdi_by_type'].items()):
            stats = result['DIRECT']['by_type'].get(entity_type, {})
            tp, total = stats.get('tp', 0), stats.get('total', 0)
            print(f"  {entity_type}: {recall:.4f} ({tp}/{total})")

    if result['erqi_by_type']:
        print(f"\nERqi by Entity Type:")
        for entity_type, recall in sorted(result['erqi_by_type'].items()):
            stats = result['QUASI']['by_type'].get(entity_type, {})
            tp, total = stats.get('tp', 0), stats.get('total', 0)
            print(f"  {entity_type}: {recall:.4f} ({tp}/{total})")

    print(f"\n=== Token-Level Metrics ===")
    print(f"Token Recall: {result['token_recall']:.4f} ({result['TOKEN_RECALL']['tp']}/{result['TOKEN_RECALL']['total']})")
    if result['token_precision'] is not None:
        print(f"Token Precision: {result['token_precision']:.4f} ({result['TOKEN_PRECISION']['tp']}/{result['TOKEN_PRECISION']['total']})")
    else:
        print(f"Token Precision: - (not available for {entity_detection_method} method)")
    print(f"\nToken Recall by Entity Type:")
    for entity_type, recall in sorted(result['token_recall_by_type'].items()):
        print(f"  {entity_type}: {recall:.4f}")

    if result['unmatched_docs']:
        print(f"\nWarning: {len(result['unmatched_docs'])} GT docs not found in anonymized file")

    # Save results to JSON
    output_path = save_results_to_json(
        result, gt_file, anonymized_file, entity_detection_method, dataset_type
    )
    print(f"\nResults saved to: {output_path}")

    # Show detailed missed entities if requested
    if show_detailed:
        print_detailed_results(gt_docs, anon_docs, spacy_model=spacy_model)


if __name__ == '__main__':
    main()

