import time, struct
import adafruit_ble
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

DEVICE_NAME   = "FeatherSense"
DATASET       = "/replay.bin"
N_FEATURES    = 16
FEATURE_BYTES = N_FEATURES * 4           # 64
CHUNK         = 19                       # record bytes per notification (+1 = 20)
SEND_INTERVAL = 30.0                     # sec/row (dataset reporting interval)
READY_TIMEOUT = 5.0                      # max wait for the gateway's ready byte

ble  = adafruit_ble.BLERadio()
ble.name = DEVICE_NAME
uart = UARTService()
adv  = ProvideServicesAdvertisement(uart)

def send_record(record):
    idx, off = 0, 0
    while off < len(record):
        uart.write(bytes([idx]) + record[off:off + CHUNK])
        off += CHUNK
        idx += 1
        time.sleep(0.005)                # small gap between notifications

def wait_for_ready(timeout):#wait to send id=0 
    start = time.monotonic()
    while ble.connected:
        if uart.in_waiting:
            uart.read(uart.in_waiting)   # discard the ready byte
            return True
        if time.monotonic() - start > timeout:
            return False
        time.sleep(0.02)
    return False

sample_id = 0
f = open(DATASET, "rb")

while True:
    ble.start_advertising(adv)
    while not ble.connected:
        time.sleep(0.1)
    ble.stop_advertising()
    print("Central connected")
    #dont send until the gateway enables notifs, else id=0 is lost
    if wait_for_ready(READY_TIMEOUT):
        print("Gateway ready")
    else:
        print("No ready signal - starting anyway")

    while ble.connected:
        feats = f.read(FEATURE_BYTES)
        if len(feats) < FEATURE_BYTES:   # end of file -> loop back
            f.seek(0)
            feats = f.read(FEATURE_BYTES)
        send_record(struct.pack("<I", sample_id) + feats)
        print(sample_id)
        sample_id += 1
        time.sleep(SEND_INTERVAL)

    print("Central disconnected - re-advertising")