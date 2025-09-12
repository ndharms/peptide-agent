import json
import pathlib
from schema import SchemaError
from schemas import example_metadata

DATA_DIR = pathlib.Path("data")
PAPERS_DIR = DATA_DIR / "relevant_papers"
OUT_PATH = DATA_DIR / "context_examples.json"

def collect_examples():
    all_examples = []

    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        metadata_path = paper_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        # Support two styles:
        # 1. metadata.json with a top-level "experiments" list
        # 2. metadata.json as a list of dicts (old style)
        if isinstance(meta, dict) and "experiments" in meta:
            records = meta["experiments"]
        elif isinstance(meta, list):
            records = meta
        else:
            records = []

        for rec in records:
            # Validate against schema
            try:
                example_metadata.validate(rec)
                rec["SOURCE"] = paper_dir.name
                all_examples.append(rec)
            except SchemaError as e:
                print(f"[SKIP] {paper_dir.name} record failed schema: {e}")

    return all_examples

def main():
    examples = collect_examples()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")

if __name__ == "__main__":
    main()
