import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(15, GPIO.IN)

try:
    while True:
        print(GPIO.input(15))
        time.sleep(0.01)
except KeyboardInterrupt:
    GPIO.cleanup()
