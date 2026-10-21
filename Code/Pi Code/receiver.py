import asyncio, csv, statistics, struct, time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bleak import BleakScanner, BleakClient

DEVICE_NAME = "FeatherSense"
TX_UUID     = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # notify: node -> gateway
RX_UUID     = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   # write:  gateway -> node
N_FEATURES  = 16
RECORD_LEN  = 4 + N_FEATURES * 4          # 68 bytes: 4B sample_id + 64B payload

FEATURES = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
            "S1_Light", "S2_Light", "S3_Light", "S4_Light",
            "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
            "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]

# One representative channel per quantity, for the report figure.
PLOT = [(0,  "S1_Temp (degC)"),
        (4,  "S1_Light (lux)"),
        (8,  "S1_Sound"),
        (12, "S5_CO2 (ppm)"),
        (14, "S6_PIR")]

WINDOW = 10                               # edge aggregation window (samples)

buf = bytearray()
expected_idx = 0
records = []
edge_rows = []
window = deque(maxlen=WINDOW)             # rolling window

lost = 0
discarded_chunks = 0
last_id = None
first_t = None


def handle(_s, data):
    """Called by bleak for every BLE notification (one chunk, <= 20 bytes)."""
    global expected_idx, lost, discarded_chunks, last_id, first_t

    idx = data[0]
    if idx == 0:                          # first chunk of a new record
        buf.clear()
        expected_idx = 1
    elif idx == expected_idx:             # expected next chunk
        expected_idx += 1
    else:                                 # out of order -> drop and resync
        discarded_chunks += 1
        buf.clear()
        expected_idx = 0
        return

    buf.extend(data[1:])                  # add the received bytes to the buffer

    if len(buf) < RECORD_LEN:
        return

    rec = bytes(buf[:RECORD_LEN])         # completed record
    del buf[:RECORD_LEN]
    expected_idx = 0

    # decode
    sid = struct.unpack_from("<I", rec, 0)[0]              # sample id (4 bytes)
    vals = struct.unpack_from("<%df" % N_FEATURES, rec, 4)  # 16 floats (64 bytes)

    t = time.time()
    if first_t is None:
        first_t = t

    if last_id is not None and sid != last_id + 1:          # loss detection
        lost += sid - last_id - 1

    last_id = sid
    records.append((t, sid, vals))

    window.append(vals)                   # edge aggregation
    if len(window) == WINDOW:
        means = tuple(sum(w[i] for w in window) / WINDOW for i in range(N_FEATURES))
        edge_rows.append((t, sid, means))

    print(f"id={sid:5d}  " + " ".join(f"{v:.2f}" for v in vals))


def report():
    """Transmission summary, CSV evidence and the per-quantity figure."""
    if not records:
        print("No data received.")
        return

    arrivals = [r[0] for r in records]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    span = arrivals[-1] - arrivals[0]
    rate = (len(records) - 1) / span if span > 0 else 0.0
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    sd_gap = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0

    print("\n===== LINK REPORT =====")
    print(f"samples received     : {len(records)}")
    print(f"samples lost         : {lost}")
    print(f"discarded chunks     : {discarded_chunks}")
    print(f"first/last sample id : {records[0][1]} / {records[-1][1]}")
    print(f"observation time     : {span:.1f} s")
    print(f"achieved rate        : {rate:.4f} samples/s")
    print(f"inter-arrival mean   : {mean_gap:.3f} s")
    print(f"inter-arrival st.dev : {sd_gap:.3f} s")
    print(f"edge windows (mean)  : {len(edge_rows)}  (window={WINDOW})")

    # raw stream
    with open("received.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "id"] + FEATURES)
        for t, sid, vals in records:
            w.writerow([t, sid] + list(vals))

    # edge-aggregated stream
    with open("edge_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "id"] + [n + "_mean" for n in FEATURES])
        for t, sid, means in edge_rows:
            w.writerow([t, sid] + list(means))

    # per-channel summary: simple feature extraction performed at the edge
    with open("summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "min", "max", "mean", "std", "last"])
        for i, name in enumerate(FEATURES):
            col = [r[2][i] for r in records]
            w.writerow([name,
                        f"{min(col):.4g}", f"{max(col):.4g}",
                        f"{statistics.mean(col):.4g}",
                        f"{statistics.pstdev(col):.4g}",
                        f"{col[-1]:.4g}"])

    print("saved received.csv, edge_features.csv, summary.csv")

    # figure: one panel per quantity, raw vs rolling mean
    t0 = arrivals[0]
    fig, axes = plt.subplots(len(PLOT), 1, figsize=(10, 12), sharex=True)
    for ax, (i, label) in zip(axes, PLOT):
        ax.plot([r[0] - t0 for r in records], [r[2][i] for r in records],
                marker=".", markersize=2, linewidth=0.8, label=f"{label} (raw)")
        if edge_rows:
            ax.plot([r[0] - t0 for r in edge_rows], [r[2][i] for r in edge_rows],
                    color="tab:red", linewidth=1.3,
                    label=f"{label} ({WINDOW}-sample mean)")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("time since first sample (s)")
    fig.suptitle("Received sensor streams at the gateway "
                 "(one channel per quantity, raw vs edge-aggregated)")
    fig.tight_layout()
    fig.savefig("received_plot.png", dpi=120)
    print("saved received_plot.png")


async def main():
    print("Scanning for", DEVICE_NAME, "...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name == DEVICE_NAME, timeout=60)
    if dev is None:
        print("Device not found - is the Feather advertising?")
        return
    print("Found", dev.address)

    async with BleakClient(dev) as client:
        await client.start_notify(TX_UUID, handle)
        print("Subscribed. Ctrl-C to stop.")

        # Handshake: tell the node we are listening so it does not send the
        # first record before notifications are enabled.
        try:
            await client.write_gatt_char(RX_UUID, b"\x01", response=True)
            print("Sent ready signal to the node.")
        except Exception:
            try:
                await client.write_gatt_char(RX_UUID, b"\x01", response=False)
                print("Sent ready signal to the node (write without response).")
            except Exception as exc:
                print("Could not send ready signal:", exc)

        try:
            while True:
                await asyncio.sleep(1)
        finally:
            # Stop notifications before writing the evidence files so every
            # sample that arrived is included in the CSVs.
            try:
                await client.stop_notify(TX_UUID)
            except Exception:
                pass
            report()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass