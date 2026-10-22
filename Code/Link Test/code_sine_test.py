import time
import struct
import math
import adafruit_ble
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

DEVICE_NAME   = "FeatherSense"
SEND_INTERVAL = 1.0                      # 1 Hz test rate

ble = adafruit_ble.BLERadio()
ble.name = DEVICE_NAME
uart = UARTService()
adv = ProvideServicesAdvertisement(uart)

def encode(sample_id, values):
    # little-endian: uint32 sample id, then N float32 values
    return struct.pack("<I", sample_id) + struct.pack("<%df" % len(values), *values)

sample_id = 0
while True:
    ble.start_advertising(adv)
    while not ble.connected:
        time.sleep(0.1)
    ble.stop_advertising()
    print("Central connected")

    while ble.connected:
        value = math.sin(sample_id * 0.1)     # known test waveform
        uart.write(encode(sample_id, [value]))
        print(sample_id, round(value, 3))
        sample_id += 1
        time.sleep(SEND_INTERVAL)

    print("Central disconnected - re-advertising")