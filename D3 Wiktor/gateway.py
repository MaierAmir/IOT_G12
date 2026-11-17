#!/usr/bin/env python3
"""gateway.py - Deliverable III gateway (Raspberry Pi 4).

Receives the replayed records from the Feather Sense over BLE (same protocol as
receiver.py), keeps the 12 features the deployed model needs (temperature is
dropped), and forwards each record to Azure IoT Hub as a JSON message.

* Sending runs in a worker thread with a queue, so BLE reception never waits for
  the network. If the link to the cloud drops, records stay in the queue and are
  re-sent in order (retry with back-off).
* Every record is written to disk as it arrives (received_16.csv) and every send
  is timed (send_log.csv) for the delay measurement.

Usage:   ~/gateway-venv/bin/python ~/Wiktor/gateway.py [--dry-run] [--limit N]
The device connection string is read from ~/Wiktor/.iot_env (never from the code).
"""
import argparse, asyncio, csv, json, os, queue, struct, threading, time

DEVICE_NAME = "FeatherSense"
TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # notify: node -> gateway
RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   # write:  gateway -> node
N_FEATURES = 16
RECORD_LEN = 4 + N_FEATURES * 4                    # 4 B sample_id + 16 float32

FEATURES = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
            "S1_Light", "S2_Light", "S3_Light", "S4_Light",
            "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
            "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]
# Model input order (deploy_config.json): light, sound, CO2, CO2 slope, PIR.
MODEL_FEATURES = FEATURES[4:]

ENV_FILE = os.path.expanduser("~/Wiktor/.iot_env")


def load_connection_string():
    cs = os.environ.get("IOTHUB_DEVICE_CONNECTION_STRING")
    if cs:
        return cs
    with open(ENV_FILE) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            if k == "IOTHUB_DEVICE_CONNECTION_STRING":
                return v.strip().strip('"').strip("'")
    raise SystemExit("IOTHUB_DEVICE_CONNECTION_STRING not found in " + ENV_FILE)


class Sender(threading.Thread):
    """Sends queued messages to IoT Hub, in order, retrying until each succeeds."""

    def __init__(self, run_id, run_dir, dry_run):
        super().__init__(daemon=True)
        self.run_id, self.dry_run = run_id, dry_run
        self.q = queue.Queue()
        self.client = None
        self.sent = 0
        self.retries = 0
        self.conn_str = None if dry_run else load_connection_string()
        self.log = open(os.path.join(run_dir, "send_log.csv"), "w", newline="")
        self.w = csv.writer(self.log)
        self.w.writerow(["sample_id", "gateway_ts", "send_start_ts", "ack_ts",
                         "attempts", "bytes", "queue_len_after"])

    def _connect(self):
        from azure.iot.device import IoTHubDeviceClient
        self.client = IoTHubDeviceClient.create_from_connection_string(self.conn_str)
        self.client.connect()

    def _send(self, sid, payload):
        if self.dry_run:
            return
        from azure.iot.device import Message
        if self.client is None:
            self._connect()
        m = Message(payload)
        m.content_type, m.content_encoding = "application/json", "utf-8"
        m.message_id = f"{self.run_id}-{sid}"      # lets the cloud spot duplicates
        self.client.send_message(m)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            sid, gw_ts, payload = item
            attempts = 0
            while True:
                attempts += 1
                t_send = time.time()
                try:
                    self._send(sid, payload)
                    t_ack = time.time()
                    break
                except Exception as exc:
                    self.retries += 1
                    self.client = None             # force a fresh connection
                    wait = min(30, 2 ** min(attempts, 5))
                    print(f"[send] id={sid} failed ({type(exc).__name__}); "
                          f"retry in {wait}s, {self.q.qsize()} queued")
                    time.sleep(wait)
            self.sent += 1
            self.w.writerow([sid, f"{gw_ts:.3f}", f"{t_send:.3f}", f"{t_ack:.3f}",
                             attempts, len(payload), self.q.qsize()])
            self.log.flush()

    def close(self, timeout=120):
        self.q.put(None)
        self.join(timeout)
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:
                pass
        self.log.close()


class Gateway:
    def __init__(self, sender, run_id, run_dir, limit=0):
        self.sender, self.run_id, self.limit = sender, run_id, limit
        self.buf = bytearray()
        self.expected_idx = 0
        self.received = 0
        self.lost = 0
        self.discarded = 0
        self.last_id = None
        self.done = threading.Event()
        self.raw = open(os.path.join(run_dir, "received_16.csv"), "w", newline="")
        self.rw = csv.writer(self.raw)
        self.rw.writerow(["t", "id"] + FEATURES)

    def reset_link(self):
        self.buf.clear()
        self.expected_idx = 0

    def handle(self, _sender, data):
        """bleak callback: one BLE notification (<= 20 bytes) per call."""
        idx = data[0]
        if idx == 0:                               # first chunk of a new record
            self.buf.clear()
            self.expected_idx = 1
        elif idx == self.expected_idx:
            self.expected_idx += 1
        else:                                      # out of order -> resync
            self.discarded += 1
            self.reset_link()
            return
        self.buf.extend(data[1:])
        if len(self.buf) < RECORD_LEN:
            return

        rec = bytes(self.buf[:RECORD_LEN])
        del self.buf[:RECORD_LEN]
        self.expected_idx = 0

        t = time.time()                            # arrival time at the gateway
        sid = struct.unpack_from("<I", rec, 0)[0]
        vals = struct.unpack_from("<%df" % N_FEATURES, rec, 4)

        if self.last_id is not None and sid != self.last_id + 1:
            self.lost += max(0, sid - self.last_id - 1)
        self.last_id = sid
        self.received += 1

        self.rw.writerow([f"{t:.3f}", sid] + list(vals))
        self.raw.flush()

        msg = {"run_id": self.run_id, "sample_id": sid, "gateway_ts": round(t, 3)}
        msg.update({n: round(v, 4) for n, v in zip(MODEL_FEATURES, vals[4:])})
        payload = json.dumps(msg, separators=(",", ":"))
        self.sender.q.put((sid, t, payload))
        print(f"id={sid:5d} queued ({len(payload)} B, {self.sender.q.qsize()} waiting)")

        if self.limit and self.received >= self.limit:
            self.done.set()

    def close(self):
        self.raw.close()


async def ble_loop(gw):
    from bleak import BleakScanner, BleakClient
    while not gw.done.is_set():
        print("Scanning for", DEVICE_NAME, "...")
        dev = await BleakScanner.find_device_by_filter(
            lambda d, ad: d.name == DEVICE_NAME, timeout=60)
        if dev is None:
            print("Device not found - is the Feather advertising?")
            continue
        try:
            async with BleakClient(dev) as client:
                gw.reset_link()
                await client.start_notify(TX_UUID, gw.handle)
                print("Subscribed. Ctrl-C to stop.")
                for resp in (True, False):         # ready byte, as in receiver.py
                    try:
                        await client.write_gatt_char(RX_UUID, b"\x01", response=resp)
                        print("Sent ready signal to the node.")
                        break
                    except Exception as exc:
                        print("Ready signal failed:", exc)
                while client.is_connected and not gw.done.is_set():
                    await asyncio.sleep(0.5)
                try:
                    await client.stop_notify(TX_UUID)
                except Exception:
                    pass
        except Exception as exc:
            print("BLE error:", exc)
        if not gw.done.is_set():
            print("BLE link lost - reconnecting")
            await asyncio.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="do not contact Azure")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records")
    args = ap.parse_args()

    run_id = int(time.time())
    run_dir = os.path.expanduser(f"~/Wiktor/runs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"run_id={run_id}  logs in {run_dir}  dry_run={args.dry_run}")

    sender = Sender(run_id, run_dir, args.dry_run)
    gw = Gateway(sender, run_id, run_dir, args.limit)
    sender.start()
    try:
        asyncio.run(ble_loop(gw))
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Stopping: waiting for {sender.q.qsize()} queued messages ...")
        sender.close()
        gw.close()
        print("\n===== GATEWAY REPORT =====")
        print(f"records received : {gw.received}")
        print(f"records lost     : {gw.lost} (gaps in sample_id)")
        print(f"discarded chunks : {gw.discarded}")
        print(f"messages sent    : {sender.sent}")
        print(f"send retries     : {sender.retries}")


if __name__ == "__main__":
    main()
