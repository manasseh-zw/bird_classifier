from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict


APP_DIR = Path(__file__).resolve().parent
SOURCE_CSV = APP_DIR / "birds.csv"
OUTPUT_CSV = APP_DIR / "bird_labels.csv"


class LabelRow(TypedDict):
    model_index: int
    class_id: int
    label: str


def load_unique_labels() -> list[LabelRow]:
    """Load labels in the same alphabetical order used by ImageFolder."""

    with SOURCE_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        deduped: dict[int, str] = {}

        for row in reader:
            class_id = int(float(row["class id"]))
            deduped.setdefault(class_id, row["labels"].strip())

    sorted_pairs = sorted(deduped.items(), key=lambda item: item[1])
    rows: list[LabelRow] = [
        {"model_index": model_index, "class_id": class_id, "label": label}
        for model_index, (class_id, label) in enumerate(sorted_pairs)
    ]
    return rows


def validate_rows(rows: list[LabelRow]) -> None:
    """Validate the exported mapping for model inference."""

    model_indices = [row["model_index"] for row in rows]
    class_ids = sorted(row["class_id"] for row in rows)

    if len(rows) != 525:
        raise ValueError(f"Expected 525 unique classes, found {len(rows)}.")

    expected_range = list(range(525))
    if model_indices != expected_range:
        raise ValueError("Model indices do not align with the expected 0-524 range.")

    if class_ids != expected_range:
        raise ValueError("Class IDs do not align with the expected 0-524 range.")


def write_rows(rows: list[LabelRow]) -> None:
    """Write the cleaned label rows to the output CSV."""

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["model_index", "class_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate the lightweight label mapping file for inference."""

    rows = load_unique_labels()
    validate_rows(rows)
    write_rows(rows)
    print(f"Wrote {len(rows)} labels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
