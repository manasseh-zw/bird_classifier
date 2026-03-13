from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict


APP_DIR = Path(__file__).resolve().parent
SOURCE_CSV = APP_DIR / "birds.csv"
OUTPUT_CSV = APP_DIR / "bird_labels.csv"


class LabelRow(TypedDict):
    class_id: int
    label: str


def load_unique_labels() -> list[LabelRow]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        deduped: dict[int, str] = {}

        for row in reader:
            class_id = int(float(row["class id"]))
            deduped.setdefault(class_id, row["labels"].strip())

    rows: list[LabelRow] = [{"class_id": class_id, "label": deduped[class_id]} for class_id in sorted(deduped)]
    return rows


def validate_rows(rows: list[LabelRow]) -> None:
    class_ids = [row["class_id"] for row in rows]

    if len(rows) != 525:
        raise ValueError(f"Expected 525 unique classes, found {len(rows)}.")

    expected_ids = list(range(525))
    if class_ids != expected_ids:
        raise ValueError("Class IDs do not align with the expected 0-524 range.")


def write_rows(rows: list[LabelRow]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["class_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_unique_labels()
    validate_rows(rows)
    write_rows(rows)
    print(f"Wrote {len(rows)} labels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
