import asyncio, struct, csv, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "FeatherSense"
TX_UUID     = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # notify: node -> gateway
NUM_FLOATS  = 1
REC_LEN     = 4 + 4 * NUM_FLOATS                       # 8 bytes, one notification

rows = []
expected = 0
lost = 0
last_id = None
first_t = None


def handle(_sender, data):
    global last_id, lost, expected, first_t
    if len(data) != REC_LEN:
        print(f"  !! bad length {len(data)} (expected {REC_LEN})")
        return
    sid = struct.unpack_from("<I", data, 0)[0]
    vals = struct.unpack_from(f"<{NUM_FLOATS}f", data, 4)
    t = time.time()
    if first_t is None:
        first_t = t
    if last_id is not None:
        expected += 1
        if sid != last_id + 1:
            gap = sid - last_id - 1
            lost += gap
            print(f"  ** lost {gap} sample(s); jumped {last_id} -> {sid}")
    last_id = sid
    rows.append((t, sid, *vals))
    print(f"t={t - first_t:7.3f}s  id={sid:6d}  vals={tuple(round(v, 4) for v in vals)}")


def report():
    if not rows:
        print("No data received.")
        return
    span = rows[-1][0] - rows[0][0]
    rate = (len(rows) - 1) / span if span > 0 else 0
    print("\n===== LINK REPORT =====")
    print(f"samples received : {len(rows)}")
    print(f"samples sent(est): {rows[-1][1] - rows[0][1] + 1}")
    print(f"samples lost     : {lost}")
    print(f"achieved rate    : {rate:.3f} samples/s")

    with open("received.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "id"] + [f"v{i}" for i in range(NUM_FLOATS)])
        w.writerows(rows)
    print("saved received.csv")

    t0 = rows[0][0]
    plt.figure()
    plt.plot([r[0] - t0 for r in rows], [r[2] for r in rows], marker=".")
    plt.xlabel("time (s)")
    plt.ylabel("value")
    plt.title("Data stream at gateway")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("received_plot.png", dpi=120)
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
        print("Subscribed. Receiving... (Ctrl-C to stop)")
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            report()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass