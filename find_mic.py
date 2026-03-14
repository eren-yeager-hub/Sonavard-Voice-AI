import sounddevice as sd

devices = sd.query_devices()

print("\nAvailable recording devices:\n")

for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        print(i, dev['name'], "| channels:", dev['max_input_channels'])