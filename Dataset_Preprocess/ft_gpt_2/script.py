import csv, json, ast, random
from pathlib import Path

IN_CSV      = Path("../../Dataset/RAW_recipes.csv")
OUT_TRAIN   = Path("../../Dataset/recipes_train.jsonl")
OUT_VAL     = Path("../../Dataset/recipes_val.jsonl")
OUT_TEST    = Path("../../Dataset/recipes_test.jsonl")

def _as_list(cell):
    """CSV stores lists as strings like "['salt','pepper']"."""
    return ast.literal_eval(cell) if isinstance(cell, str) else []

def row_to_record(row):
    return {
        "id":          int(row["id"]),
        "name":        row["name"].strip(),
        "tags":        _as_list(row["tags"]),
        "ingredients": _as_list(row["ingredients"])
    }

# read & clean
with IN_CSV.open(newline="", encoding="utf-8") as f:
    reader  = csv.DictReader(f)
    records = [row_to_record(r) for r in reader if r["name"]]

# shuffle & split
random.seed(42)
random.shuffle(records)

n_total  = len(records)
n_train  = int(n_total * 0.80)
n_val    = int(n_total * 0.10)
n_test   = n_total - n_train - n_val

train_set = records[:n_train]
val_set   = records[n_train:n_train + n_val]
test_set  = records[n_train + n_val:]

assert len(train_set) + len(val_set) + len(test_set) == n_total

# three sets
for path, subset in [
    (OUT_TRAIN, train_set),
    (OUT_VAL,   val_set),
    (OUT_TEST,  test_set)
]:
    with path.open("w", encoding="utf-8") as f:
        for rec in subset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"wrote ➜  train: {len(train_set):,} | val: {len(val_set):,} | test: {len(test_set):,}")
