import json, itertools
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR   = Path("../Dataset")
IN_FILES   = {
    "train": DATA_DIR / "recipes_train.jsonl",
    "val":   DATA_DIR / "recipes_val.jsonl",
    "test":  DATA_DIR / "recipes_test.jsonl",
}

OUT_FILES  = {
    split: DATA_DIR / f"{f.stem}_multi.jsonl" for split, f in IN_FILES.items()
}

K_NEIGHBOURS = 10

# ---------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def as_doc(rec: Dict) -> str:
    """
    Turn a recipe into a single whitespace-separated string
    that mixes ingredients + tags.
    """
    return " ".join(
        itertools.chain(
            (t.lower().replace(" ", "_") for t in rec["tags"]),
            (ing.lower().replace(" ", "_") for ing in rec["ingredients"])
        )
    )

print("Loading datasets …")
datasets = {split: load_jsonl(path) for split, path in IN_FILES.items()}
all_records = list(itertools.chain.from_iterable(datasets.values()))

print(f"Vectorising {len(all_records):,} recipes with TF-IDF …")
docs = [as_doc(rec) for rec in all_records]
vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(docs)

print("Building nearest-neighbour index …")
nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, metric="cosine", algorithm="brute", n_jobs=-1).fit(X)

print("Querying neighbours …")
distances, indices = nn.kneighbors(X, return_distance=True)

print("Attaching ranked name lists …")
for rec, neigh_idxs, neigh_dists in zip(all_records, indices, distances):
    # neighbours are already sorted by distance (0 == self)
    neighbour_names = [ all_records[i]["name"] for i in neigh_idxs[1:] ]
    rec["names"] = [rec["name"], *neighbour_names]

print("Writing new *_multi.jsonl files …")
cursor = 0
for split, original in datasets.items():
    n = len(original)
    save_jsonl(Path(OUT_FILES[split]), all_records[cursor : cursor + n])
    cursor += n
    print(f"  {OUT_FILES[split]}  ← {n:,} records")
