import asyncio, struct, csv, time
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "FeatherSense"
TX_UUID     = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
N_FEATURES  = 16
RECORD_LEN  = 4 + N_FEATURES * 4 #68 bytes (4B id + 64B payload)
FEATURES = ["S1_Temp","S2_Temp","S3_Temp","S4_Temp","S1_Light","S2_Light","S3_Light","S4_Light",
            "S1_Sound","S2_Sound","S3_Sound","S4_Sound","S5_CO2","S5_CO2_Slope","S6_PIR","S7_PIR"]

WINDOW = 10                               # edge aggregation window (samples)

buf = bytearray()
expected_idx = 0
records = [] 
edge_rows = []
window = deque(maxlen=WINDOW) #for rolling window

lost = 0
bad_records = 0
last_id = None
first_t = None

def handle(_s, data):#run when a ble notif arrives
    global expected_idx, lost, bad_records, last_id, first_t
    idx = data[0]

    if idx == 0:#new record
        buf.clear()
        expected_idx = 1

    elif idx == expected_idx:
        expected_idx += 1

    else:#corrupted so skip 
        bad_records += 1
        buf.clear()
        expected_idx = 0
        return

    buf.extend(data[1:])#add recieved to buffer

    if len(buf) >= RECORD_LEN: 
        rec = bytes(buf[:RECORD_LEN])#completed rec
        del buf[:RECORD_LEN]
        expected_idx = 0

        #decode
        sid = struct.unpack_from("<I", rec, 0)[0] #sample id first 4 bytes)
        vals = struct.unpack_from("<%df" % N_FEATURES, rec, 4) #16 sensor vals (64B) after 4B header

        t = time.time()

        if first_t is None:
            first_t = t

        if last_id is not None and sid != last_id + 1: #loss detect
            lost += sid - last_id - 1

        last_id = sid
        records.append((t, sid, vals))#store rec
        window.append(vals)# agg

        if len(window) == WINDOW:    #rolling window
            means = tuple(sum(w[i] for w in window) / WINDOW for i in range(N_FEATURES))
            edge_rows.append((t, sid, means))

        print(f"id={sid:5d}  " + " ".join(f"{v:.2f}" for v in vals))

def report(): #transmittion summary and plots
    if not records:
        print("No data received.")
        return

    span = records[-1][0] - records[0][0]
    rate = (len(records) - 1) / span if span > 0 else 0

    print("\n===== LINK REPORT =====")
    print(f"samples received   : {len(records)}")
    print(f"samples lost       : {lost}")
    print(f"corrupt records    : {bad_records}")
    print(f"achieved rate      : {rate:.3f} samples/s")
    print(f"edge windows (mean): {len(edge_rows)}  (window={WINDOW})")

    with open("received.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","id"] + FEATURES)
        for row in records: 
            w.writerow([row[0], row[1]] + list(row[2]))

    with open("edge_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","id"] + [n+"_mean" for n in FEATURES])
        for row in edge_rows: 
            w.writerow([row[0], row[1]] + list(row[2]))

    print("saved received.csv, edge_features.csv")

    t0 = records[0][0]
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot([r[0]-t0 for r in records], [r[2][0] for r in records], marker=".", label="S1_Temp (raw)")
    plt.legend(); plt.grid(True)
    plt.title("Gateway: raw vs edge-aggregated")
    plt.subplot(2,1,2)

    if edge_rows:
        plt.plot([r[0]-t0 for r in edge_rows], [r[2][0] for r in edge_rows], color="tab:red", label="S1_Temp (10-sample mean)")
        plt.legend()
        plt.grid(True)

    plt.xlabel("time (s)"); plt.tight_layout()
    plt.savefig("received_plot.png", dpi=120)
    print("saved received_plot.png")

async def main():
    print("Scanning for", DEVICE_NAME, "...")

    dev = await BleakScanner.find_device_by_filter( #find feather
        lambda d, ad: d.name == DEVICE_NAME, timeout=60)

    if dev is None:
        print("Device not found.")
        return
    print("Found", dev.address)

    async with BleakClient(dev) as client:
        await client.start_notify(TX_UUID, handle) #call handle() when data arrives
        print("Subscribed. Ctrl-C to stop.")

        try:
            while True: await asyncio.sleep(1)
        finally:
            report()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass