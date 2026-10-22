import csv, json, os, random, struct, collections


def repo_root():
    """Walk up from this file until a directory containing 'Dataset' is found."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "Dataset")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.getcwd()
        d = parent


ROOT = repo_root()
SRC = os.path.join(ROOT, "Dataset", "Occupancy_Estimation.csv")
OUTDIR = os.path.join(ROOT, "Dataset")
os.makedirs(OUTDIR, exist_ok=True)

FEATURES = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
            "S1_Light", "S2_Light", "S3_Light", "S4_Light",
            "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
            "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]
TARGET = "Room_Occupancy_Count"
ALLCOLS = ["Date", "Time"] + FEATURES + [TARGET]
SEED = 42
TEST_FRAC = 0.2

if not os.path.exists(SRC):
    raise SystemExit("source dataset not found at " + SRC)

with open(SRC, newline="") as f:
    rows = list(csv.DictReader(f))
n = len(rows)

groups = collections.defaultdict(list)
for i, row in enumerate(rows):
    groups[row[TARGET]].append(i)

rng = random.Random(SEED)
test_idx = set()
for cls, idxs in groups.items():
    idxs = idxs[:]
    rng.shuffle(idxs)
    k = int(round(len(idxs) * TEST_FRAC))
    test_idx.update(idxs[:k])

test_rows  = [rows[i] for i in range(n) if i in test_idx]      # time order kept
train_rows = [rows[i] for i in range(n) if i not in test_idx]


def write_csv(path, rs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALLCOLS)
        w.writeheader()
        for row in rs:
            w.writerow({c: row[c] for c in ALLCOLS})


write_csv(os.path.join(OUTDIR, "train.csv"), train_rows)
write_csv(os.path.join(OUTDIR, "test.csv"), test_rows)

with open(os.path.join(OUTDIR, "replay.bin"), "wb") as f:
    for row in test_rows:
        vals = [float(row[c]) for c in FEATURES]
        f.write(struct.pack("<%df" % len(FEATURES), *vals))

with open(os.path.join(OUTDIR, "replay_labels.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "label", "Date", "Time"])
    for sid, row in enumerate(test_rows):
        w.writerow([sid, row[TARGET], row["Date"], row["Time"]])

manifest = {
    "source": SRC, "n_total": n,
    "features": FEATURES, "target": TARGET,
    "split": {"method": "stratified random", "seed": SEED, "test_fraction": TEST_FRAC},
    "counts": {"train": len(train_rows), "test": len(test_rows)},
    "class_balance": {
        "train": dict(sorted(collections.Counter(r[TARGET] for r in train_rows).items())),
        "test":  dict(sorted(collections.Counter(r[TARGET] for r in test_rows).items())),
    },
    "replay_bin": {
        "record_bytes": len(FEATURES) * 4, "rows": len(test_rows),
        "layout": f"per row: {len(FEATURES)} x float32 little-endian, in FEATURES order",
    },
}
with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
print("replay.bin bytes:", os.path.getsize(os.path.join(OUTDIR, "replay.bin")))