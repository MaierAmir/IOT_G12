import csv
import os
import struct
import sys

FEATURES = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
            "S1_Light", "S2_Light", "S3_Light", "S4_Light",
            "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
            "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]
REC = len(FEATURES) * 4
TOL = 1e-3


def repo_root(): #find repo root/ dataset dir
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "Dataset")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.getcwd()
        d = parent


def load_sent(path):
    rows = []
    with open(path, "rb") as f:
        while True:
            b = f.read(REC)
            if len(b) < REC:
                break
            rows.append(struct.unpack("<%df" % len(FEATURES), b))
    return rows


def main():
    root = repo_root()
    recv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "SimRuns", "received.csv")
    bin_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "Dataset", "replay.bin")

    if not os.path.exists(recv_path):
        print("received.csv not found at", recv_path, "- scp it over first.")
        return 1
    if not os.path.exists(bin_path):
        print("replay.bin not found at", bin_path)
        return 1

    sent = load_sent(bin_path)
    if not sent:
        print("replay.bin is empty")
        return 1

    n = 0
    mismatch = 0
    maxdiff = 0.0
    ids = []
    per_channel_max = [0.0] * len(FEATURES)

    with open(recv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2 + len(FEATURES):
                continue
            sid = int(row[1])
            vals = [float(x) for x in row[2:2 + len(FEATURES)]]
            ids.append(sid)
            n += 1
            # The node wraps replay.bin but sample_id keeps counting, so index % the number of rows that were sent
            exp = sent[sid % len(sent)]
            for i, (a, b) in enumerate(zip(vals, exp)):
                d = abs(a - b)
                if d > per_channel_max[i]:
                    per_channel_max[i] = d
                if d > maxdiff:
                    maxdiff = d
                if d > TOL:
                    mismatch += 1

    if n == 0:
        print("received.csv contains no data rows")
        return 1

    lo, hi = min(ids), max(ids)
    present = set(ids)
    missing = [i for i in range(lo, hi + 1) if i not in present]
    dupes = n - len(present)

    print("===== SENT vs RECEIVED VERIFICATION =====")
    print(f"rows compared        : {n}")
    print(f"unique sample ids    : {len(present)} (min {lo}, max {hi})")
    if missing:
        shown = missing[:20]
        print(f"ids missing in range : {len(missing)}  -> {shown}"
              + (" ..." if len(missing) > len(shown) else ""))
    else:
        print("ids missing in range : 0")
    print(f"duplicate ids        : {dupes}")
    print(f"max abs difference   : {maxdiff:.6g}")
    print(f"mismatched values    : {mismatch} (tolerance {TOL:g})")
    if mismatch == 0:
        worst = max(range(len(FEATURES)), key=lambda i: per_channel_max[i])
        print(f"worst channel        : {FEATURES[worst]} ({per_channel_max[worst]:.6g})")
    print("note                 : only rows that arrived are compared; ids absent")
    print("                       from the log are reported above as missing.")
    print("RESULT:", "PASS - values received match the values sent"
          if mismatch == 0 else "FAIL - values differ")
    return 0 if mismatch == 0 else 2


if __name__ == "__main__":
    sys.exit(main())