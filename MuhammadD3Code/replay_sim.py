#python replay_sim.py <run-id> <seconds_between_records> [max_sample_id] [replay.bin]
#(python replay_sim.py smoke-02 0.5 20)
#receiver_d3.py in same folder, IOTHUB_DEVICE_CONNECTION_STRING set
#pip install azure-iot-device"

import asyncio, sys
import receiver_d3 as g

async def main(run_id, interval, max_sid, path):
    g.setup(run_id, max_sid)
    g.SRC = "sim"
    send_task = asyncio.create_task(g.sender())
    try:
        raw = open(path, "rb").read()
        n = len(raw) // (4 * g.N_FEATURES)
        for sid in range(n):
            if g.stop.is_set():
                break
            rec = sid.to_bytes(4, "little") + raw[sid * 64:(sid + 1) * 64]       # 68-byte record
            for i, k in enumerate(range(0, g.RECORD_LEN, 19)):
                g.handle(None, bytearray(bytes([i]) + rec[k:k + 19]))           # one notification
            await asyncio.sleep(interval)
    finally:
        g.stop.set()
        await send_task
        g.report()
        g.summary()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    try:
        asyncio.run(main(sys.argv[1], float(sys.argv[2]),
                         int(sys.argv[3]) if len(sys.argv) > 3 else 2026,
                         sys.argv[4] if len(sys.argv) > 4 else "replay.bin"))
    except KeyboardInterrupt:
        pass
