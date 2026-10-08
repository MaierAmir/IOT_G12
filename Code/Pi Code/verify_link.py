import csv, struct, os
BIN  = r"C:\Users\HP\Desktop\iot\processed\replay.bin"
RECV = r"C:\Users\HP\Desktop\iot\processed\received.csv"
FEATURES = ["S1_Temp","S2_Temp","S3_Temp","S4_Temp","S1_Light","S2_Light","S3_Light","S4_Light",
            "S1_Sound","S2_Sound","S3_Sound","S4_Sound","S5_CO2","S5_CO2_Slope","S6_PIR","S7_PIR"]
REC = len(FEATURES)*4

if not os.path.exists(RECV):
    print("received.csv not found at", RECV, "- scp it over first.")
    raise SystemExit(1)

sent = []
with open(BIN, "rb") as f:
    while True:
        b = f.read(REC)
        if len(b) < REC: break
        sent.append(struct.unpack("<%df" % len(FEATURES), b))

n = 0; mismatch = 0; maxdiff = 0.0; ids = set()
with open(RECV, newline="") as f:
    r = csv.reader(f); next(r)
    for row in r:
        sid = int(row[1]); vals = [float(x) for x in row[2:]]
        ids.add(sid); n += 1
        exp = sent[sid]
        for a, b in zip(vals, exp):
            d = abs(a-b); maxdiff = max(maxdiff, d)
            if d > 1e-3: mismatch += 1

print("===== SENT vs RECEIVED VERIFICATION =====")
print(f"rows compared      : {n}")
print(f"unique sample ids  : {len(ids)} (min {min(ids)}, max {max(ids)})")
print(f"max abs difference : {maxdiff:.6g}")
print(f"mismatched values  : {mismatch}")
print("RESULT:", "PASS - link is lossless" if mismatch == 0 else "FAIL - values differ")