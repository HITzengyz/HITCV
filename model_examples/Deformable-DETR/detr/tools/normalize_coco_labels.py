import argparse
import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def collect_category_ids(annotations):
    return sorted({ann["category_id"] for ann in annotations if "category_id" in ann})


def build_contiguous_mapping(category_ids):
    return {old_id: new_id for new_id, old_id in enumerate(category_ids)}


def remap_annotations(annotations, mapping):
    for ann in annotations:
        if "category_id" in ann:
            ann["category_id"] = mapping[ann["category_id"]]


def remap_categories(categories, mapping):
    if not categories:
        return [{"id": new_id, "name": f"class_{new_id}"} for new_id in mapping.values()]
    remapped = []
    for cat in categories:
        cat_id = cat.get("id")
        if cat_id in mapping:
            new_cat = dict(cat)
            new_cat["id"] = mapping[cat_id]
            remapped.append(new_cat)
    remapped.sort(key=lambda c: c["id"])
    return remapped


def validate_annotations(annotations):
    category_ids = collect_category_ids(annotations)
    if not category_ids:
        raise ValueError("No category_id values found in annotations.")
    min_id = min(category_ids)
    max_id = max(category_ids)
    contiguous = category_ids == list(range(min_id, max_id + 1))
    return {
        "min_id": min_id,
        "max_id": max_id,
        "unique_count": len(category_ids),
        "contiguous": contiguous,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize COCO category_id to 0-based contiguous IDs without overwriting the original file."
    )
    parser.add_argument("--input", required=True, help="Path to input COCO JSON (instances_*.json)")
    parser.add_argument("--output", required=True, help="Path to output COCO JSON (new file)")
    parser.add_argument("--force", action="store_true", help="Allow overwriting output if it exists")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}")

    data = load_json(args.input)
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    before_stats = validate_annotations(annotations)
    category_ids = collect_category_ids(annotations)
    mapping = build_contiguous_mapping(category_ids)

    remap_annotations(annotations, mapping)
    data["annotations"] = annotations
    data["categories"] = remap_categories(categories, mapping)

    after_stats = validate_annotations(annotations)
    if after_stats["min_id"] != 0:
        raise ValueError("Normalization failed: min category_id is not 0.")
    if after_stats["max_id"] != after_stats["unique_count"] - 1:
        raise ValueError("Normalization failed: category_id not contiguous after remap.")

    save_json(args.output, data)

    print("Normalization complete.")
    print("Input stats:", before_stats)
    print("Output stats:", after_stats)
    print("Mapping size:", len(mapping))
    print("Wrote:", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
