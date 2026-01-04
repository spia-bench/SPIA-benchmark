import json
import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators_dir", type=str, required=True,
                        help="Path to annotators directory (e.g., data/spia/annotators/tab)")
    return parser.parse_args()

args = parse_args()
annotated_dir = Path(args.annotators_dir)

# Extract data name from parent directory (e.g., TAB -> tab)
data = annotated_dir.parent.name.lower()

output_dir = annotated_dir.parent / "inter_annotator"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{data}_inter_cross_aligned.jsonl"

# Load all annotator data
annotator_files = sorted(annotated_dir.glob("*.jsonl"))
all_data = defaultdict(list)

for file_path in annotator_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_id = data['metadata']['data_id']
            all_data[data_id].append(data)

# Separate duplicated and unique data
duplicated = {k: v for k, v in all_data.items() if len(v) > 1}
unique = {k: v[0] for k, v in all_data.items() if len(v) == 1}

print(f"Total unique data_ids: {len(all_data)}")
print(f"Duplicated data_ids: {len(duplicated)}")
print(f"Unique data_ids: {len(unique)}")

# Verify only 2 annotators per duplicated data
for data_id, entries in duplicated.items():
    assert len(entries) == 2, f"data_id {data_id} has {len(entries)} annotators (expected 2)"

print("Verified: All duplicated data has exactly 2 annotators")

# Verify text consistency for duplicated data
for data_id, entries in duplicated.items():
    assert entries[0]['text'] == entries[1]['text'], f"Text mismatch for data_id {data_id}"

print("Verified: All duplicated data has identical text")

# Create cross-labeled dataset
cross_labeled = []
mismatched_subject_count = []

for data_id, entries in duplicated.items():
    annotation_a = entries[0]['subjects']
    annotation_b = entries[1]['subjects']

    if len(annotation_a) != len(annotation_b):
        mismatched_subject_count.append(data_id)

    merged = {
        "metadata": {
            "data_id": data_id,
            "number_of_subjects": entries[0]['metadata']['number_of_subjects']
        },
        "text": entries[0]['text'],
        "annotation_A": annotation_a,
        "annotation_B": annotation_b
    }
    cross_labeled.append(merged)

print(f"\nMismatched subject counts: {len(mismatched_subject_count)}")
if mismatched_subject_count:
    print("Data IDs with mismatched subject counts:")
    for data_id in mismatched_subject_count:
        print(f"  {data_id}")

# Save cross-labeled data
with open(output_path, 'w', encoding='utf-8') as f:
    for data in cross_labeled:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"\nSaved {len(cross_labeled)} cross-labeled samples to {output_path}")
