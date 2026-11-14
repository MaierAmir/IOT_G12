#python receiver_d3.py full-2s-01            (stops after sample_id 2026 = one pass)
#python receiver_d3.py short-30s-01 59       (stops after sample_id 59)

import asyncio, csv, json, math, os, statistics, struct, sys, time
from collections import deque

DEVICE_NAME = "FeatherSense"
TX_UUID     = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # notify: node -> gateway
RX_UUID     = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   # write:  gateway -> node
N_FEATURES  = 16
RECORD_LEN  = 4 + N_FEATURES * 4          # 68 bytes: 4B sample_id + 64B payload

FEATURES = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
            "S1_Light", "S2_Light", "S3_Light", "S4_Light",
            "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
            "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]

#D3: model input/ cloudlink settings ///////////////////////////////////////////////////////////////////////////

# model takes 16 features - 4 temp (deploy_config.json).
MODEL_FEATURES = FEATURES[4:]
# model signature declares as integers,node sends every val as f32
INT_FEATURES = {"S1_Light", "S2_Light", "S3_Light", "S4_Light", "S5_CO2", "S6_PIR", "S7_PIR"}
# realitic limits for input valid at edge (dataset max: light 280, CO2 1270).
LIMITS = {"light": (0, 20000), "sound": (0, 100), "co2": (300, 5000), "slope": (-100, 100), "pir": (0, 1)}
DRAIN_RATE = 10.0                         # max msg/s while clearing backlog (IoT Hub throttling)
DRAIN_TIMEOUT = 120.0                     # after run ends, give up on undel msgs after this

buf = bytearray()
expected_idx = 0
records = []

lost = 0
discarded_chunks = 0
last_id = None
first_t = None

# D3: run state, set by setup() ///////////////////////////////////////////////////////////////////////////////////////////
t_chunk0 = None                           # arvl time of chunk 0 of rec bieng assembled
RUN_ID = "run"
MAX_SID = 2026
SRC = "ble"                               # replay_sim.py set to sim
queue = deque()                           # messgs waiting for IoT Hub (in memory)
stop = None
accepted = 0
rejected = 0
delivered = 0


def handle(_s, data):
    """Called by bleak for every BLE notification (one chunk, <= 20 bytes)."""
    global expected_idx, lost, discarded_chunks, last_id, first_t, t_chunk0

    idx = data[0]
    if idx == 0:                          # first chunk of a new record
        buf.clear()
        expected_idx = 1
        t_chunk0 = time.time()            # D3: for the BLE chunk-spread delay
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

    print(f"id={sid:5d}  " + " ".join(f"{v:.2f}" for v in vals))

    forward(sid, vals, t_chunk0, t)       # D3: hand the record to the cloud uplink


# D3///////////////////////////////////////////////////////////////////////////////
def limit_for(name):
    return LIMITS["light" if "Light" in name else "sound" if "Sound" in name else "slope" if "Slope" in name else "co2" if "CO2" in name else "pir"]


def build_payload(sid, vals, t_first_chunk, t_rx):
#Prep rec for model keep 12 model feats, validate, cast ints Rets JSON dict thats sent to IoT Hub

    d = {"run": RUN_ID, "sid": int(sid), "src": SRC, "t_rx": round(t_rx, 3), "ble_ms": round((t_rx - t_first_chunk) * 1000.0, 1)}

    for name in MODEL_FEATURES:
        x = float(vals[FEATURES.index(name)])
        lo, hi = limit_for(name)
        if not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f"{name}={x} outside [{lo},{hi}]")
        d[name] = int(round(x)) if name in INT_FEATURES else round(x, 6)
    return d


def forward(sid, vals, t_first_chunk, t):
#Val record n queue it sender() delivers it, BLE reception never waits for Azure
    global accepted, rejected
    try:
        queue.append(build_payload(sid, vals, t_first_chunk, t))
        accepted += 1
    except ValueError as e:
        rejected += 1
        print(f"      [reject] sid={sid}: {e}")
    if sid >= MAX_SID:
        stop.set()


async def sender():
    """send queued mesgs to IoT, oldest 1st, ff Azure unreachable messges stay in q and retried w growing delay, whenlink returns backlog is sent in order"""

    global delivered
    from azure.iot.device import Message
    from azure.iot.device.aio import IoTHubDeviceClient
    conn = os.environ["IOTHUB_DEVICE_CONNECTION_STRING"]
    ws = os.environ.get("IOTHUB_WEBSOCKETS") == "1"
    client, backoff, stop_at = None, 1.0, None

    while True:
        if not queue:
            if stop.is_set():
                return
            await asyncio.sleep(0.05)
            continue
        
        if stop.is_set():
            stop_at = stop_at or time.time()
            if time.time() - stop_at > DRAIN_TIMEOUT:
                print(f"giving up: {len(queue)} message(s) were not delivered")
                return
            
        body = queue[0]

        try:
            if client is None:
                client = IoTHubDeviceClient.create_from_connection_string(conn, websockets=ws)
                await asyncio.wait_for(client.connect(), 20)

            body["t_tx"] = round(time.time(), 3)             # real send time, for the delay breakdown
            msg = Message(json.dumps(body, separators=(",", ":")), content_encoding="utf-8", content_type="application/json")
            await asyncio.wait_for(client.send_message(msg), 20)   # returns when IoT Hub acknowledges
            queue.popleft()
            delivered += 1
            backoff = 1.0

            if len(queue) > 1:                               # clearing a backlog: dont burst
                await asyncio.sleep(1.0 / DRAIN_RATE)

        except Exception as e:                               # outage, throttling, quota 
            print(f"[send] sid={body['sid']} failed ({type(e).__name__}: {e}); "
                  f"retry in {backoff:.0f}s, {len(queue)} queued")
            
            try:
                if client is not None:
                    await client.shutdown()

            except Exception:
                pass
            
            client = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def setup(run_id, max_sid):
    global RUN_ID, MAX_SID, stop
    RUN_ID, MAX_SID = run_id, max_sid
    os.environ["IOTHUB_DEVICE_CONNECTION_STRING"]            # fail early if not set
    stop = asyncio.Event()


def summary():
    print(f"\n== gateway summary: received={len(records)} lost={lost} discarded_chunks={discarded_chunks} "
          f"rejected={rejected} delivered={delivered} undelivered={len(queue)}")


def report():
    """Transmission summary and the raw CSV."""
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

    # raw stream (D3: 1 file per run so runs dont overwrite each other)
    with open(f"{RUN_ID}_received.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "id"] + FEATURES)
        for t, sid, vals in records:
            w.writerow([t, sid] + list(vals))

    print(f"saved {RUN_ID}_received.csv")


# //////////////////////////////////////////////////////////BLE (D1 main + reconnect)
async def main():
    from bleak import BleakScanner, BleakClient      # D3: imported here so replay_sim.py needs no BLE stack
    send_task = asyncio.create_task(sender())        # D3
    try:
        while not stop.is_set():                      # D3: loop so a dropped BLE link is rescanned
            print("Scanning for", DEVICE_NAME, "...")
            dev = await BleakScanner.find_device_by_filter(
                lambda d, ad: d.name == DEVICE_NAME, timeout=60)
            if dev is None:
                print("Device not found - is the Feather advertising?")
                continue
            print("Found", dev.address)

            gone = asyncio.Event()                    # D3: set if the BLe link drops
            try:
                async with BleakClient(dev, disconnected_callback=lambda c: gone.set()) as client:
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

                    waiters = [asyncio.create_task(gone.wait()), asyncio.create_task(stop.wait())]
                    await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    for w in waiters:
                        w.cancel()
                    if stop.is_set():
                        # Stop notifs before writing the evidence file so every
                        # sample that arrived is included in the CSV.
                        try:
                            await client.stop_notify(TX_UUID)
                        except Exception:
                            pass
            except Exception as e:
                print("BLE error:", e)
            if not stop.is_set():
                print("BLE link lost - re-scanning (the node keeps its sample_id)")
                await asyncio.sleep(2)
    finally:
        stop.set()
        await send_task                               # D3: wait for the queue to empty
        report()
        summary()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python receiver_d3.py <run-id> [max_sample_id]")

    async def run():
        setup(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2026)
        await main()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
