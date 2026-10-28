import sys
sys.path.insert(0, "/home/debo/packages")

import cv2
import numpy as np
import time
import os
from collections import defaultdict
from picamera2 import Picamera2
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

# -----------------------
# CONFIG
# -----------------------

CONFIDENCE_THRESHOLD = 0.7
VIDEO_FPS = 10
FRAME_SIZE = (640, 480)
YOLO_INTERVAL = 3
LEADERBOARD_WIDTH = 200
LEADERBOARD_HEIGHT_PER_ITEM = 25
LEADERBOARD_MARGIN = 10
CHUNK_DURATION_MIN = 20  # 20-minute chunks
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# -----------------------
# COCO Classes (Russian) – 80 total
# -----------------------

TRACK_CLASSES = {
    0: "человек",          # person
    1: "велосипед",        # bicycle
    2: "машина",           # car
    3: "мотоцикл",         # motorcycle
    4: "самолет",          # airplane
    5: "автобус",          # bus
    6: "поезд",            # train
    7: "грузовик",         # truck
    8: "лодка",            # boat
    9: "светофор",         # traffic light
    10: "пожарный гидрант", # fire hydrant
    11: "знак остановки",  # stop sign
    12: "парковочный счетчик", # parking meter
    13: "скамейка",        # bench
    14: "птица",           # bird
    15: "кот",             # cat
    16: "собака",          # dog
    17: "лошадь",          # horse
    18: "овца",            # sheep
    19: "корова",          # cow
    20: "слон",            # elephant
    21: "медведь",         # bear
    22: "зебра",           # zebra
    23: "жираф",           # giraffe
    24: "рюкзак",          # backpack
    25: "зонт",            # umbrella
    26: "сумка",           # handbag
    27: "кейс",            # tie case
    28: "чемодан",         # suitcase
    29: "фрисби",          # frisbee
    30: "лыжи",            # skis
    31: "сноуборд",        # snowboard
    32: "спортивный мяч",  # sports ball
    33: "кайт",            # kite
    34: "бейсбольная бита",# baseball bat
    35: "бейсбольный мяч", # baseball glove
    36: "скейтборд",       # skateboard
    37: "серфборд",        # surfboard
    38: "теннисная ракетка", # tennis racket
    39: "бутылка",         # bottle
    40: "бокал",           # wine glass
    41: "чашка",           # cup
    42: "вилка",           # fork
    43: "нож",             # knife
    44: "ложка",           # spoon
    45: "тарелка",         # bowl
    46: "банан",           # banana
    47: "яблоко",          # apple
    48: "сэндвич",         # sandwich
    49: "апельсин",        # orange
    50: "брокколи",        # broccoli
    51: "морковь",         # carrot
    52: "хот-дог",         # hot dog
    53: "пицца",           # pizza
    54: "пончик",          # donut
    55: "кекс",            # cake
    56: "стул",            # chair
    57: "диван",           # couch
    58: "горшок",          # potted plant
    59: "кровать",         # bed
    60: "обеденный стол",  # dining table
    61: "туалет",          # toilet?
    62: "телевизор",       # tv
    63: "ноутбук",         # laptop?
    64: "мышь",            # mouse
    65: "пульт",           # remote?
    66: "клавиатура",      # keyboa?
    67: "мобильный телефон", # cell?
    68: "микроволновка",   # microw?
    69: "духовка",         # oven
    70: "тостер",          # toaster
    71: "раковина",        # sink
    72: "холодильник",     # refrigerator
    73: "книга",           # book
    74: "часы",            # clock
    75: "ваза",            # vase
    76: "ножницы",         # scissors
    77: "мишень",          # teddy bear
    78: "фен",             # hair drier
    79: "зубная щетка"     # toothbrush
        }

HIST_THRESHOLD = 0.2

# -----------------------
# YOLO MODEL
# -----------------------

custom_model_path = "/home/debo/runs/train/your_model/weights/best.pt"
if os.path.isfile(custom_model_path):
    print(f"[INFO] Loading custom YOLO model from {custom_model_path}")
    model = YOLO(custom_model_path)
else:
    print(f"[WARNING] Custom model not found. Falling back to pretrained YOLOv8n model.")
    model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 nano

# -----------------------
# INITIALIZATION
# -----------------------

picam2 = Picamera2()
video_config = picam2.create_preview_configuration(main={"size": FRAME_SIZE, "format": "RGB888"})
picam2.configure(video_config)
picam2.start()

frame_count = 0
home = os.environ["HOME"]
save_path = f"{home}/Videos/DriveCounts"
os.makedirs(save_path, exist_ok=True)

object_histograms = defaultdict(list)
leaderboard_count = defaultdict(int)
leaderboard_confidence = defaultdict(float)
last_boxes = []

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def is_new_object(obj_roi, class_id, score, bbox, expiration_sec=5):
    """Return True if this is a new object, else False. Objects older than expiration_sec are removed."""
    hist = cv2.calcHist([obj_roi], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    x1, y1, x2, y2 = bbox
    now = time.time()

    # Remove expired objects
    object_histograms[class_id] = [(h, b, t) for h, b, t in object_histograms[class_id] if now - t <= expiration_sec]

    for existing_hist, existing_bbox, existing_time in object_histograms[class_id]:
        # Histogram similarity
        diff = cv2.compareHist(hist, existing_hist, cv2.HISTCMP_CORREL)
        # Bounding box overlap
        ex1, ey1, ex2, ey2 = existing_bbox
        dx = min(x2, ex2) - max(x1, ex1)
        dy = min(y2, ey2) - max(y1, ey1)
        overlap_area = max(dx, 0) * max(dy, 0)
        bbox_area = (x2 - x1) * (y2 - y1)
        overlap_ratio = overlap_area / max(bbox_area, 1)

        if diff > 1 - HIST_THRESHOLD and overlap_ratio > 0.2:
            leaderboard_confidence[class_id] = max(leaderboard_confidence[class_id], score)
            return False

    # Add new object with timestamp
    object_histograms[class_id].append((hist, bbox, now))
    leaderboard_count[class_id] += 1
    leaderboard_confidence[class_id] = max(leaderboard_confidence[class_id], score)
    return True

def put_text_russian(frame, text, position, font_size=20, color=(232, 62, 140)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_leaderboard(frame):
    x0 = FRAME_SIZE[0] - LEADERBOARD_WIDTH - LEADERBOARD_MARGIN
    y0 = LEADERBOARD_MARGIN
    max_count = max(leaderboard_count.values(), default=1)
    font_size = 16
    font = ImageFont.truetype(FONT_PATH, font_size)

    for i, cls in enumerate(leaderboard_count.keys()):
        y = y0 + i * LEADERBOARD_HEIGHT_PER_ITEM
        max_bar_width = LEADERBOARD_WIDTH - 10  # margin

        # Draw black outline first (full width)
        cv2.rectangle(frame, (x0, y), (x0 + max_bar_width, y + 20), (0, 0, 0), 2)

        # Draw purple bar proportional to count
        count_width = int((leaderboard_count[cls] / max_count) * max_bar_width)
        cv2.rectangle(frame, (x0, y), (x0 + count_width, y + 20), (111, 66, 193), -1)

        # Draw lighter inner bar for confidence
        conf_width = int(leaderboard_confidence[cls] * max_bar_width)
        cv2.rectangle(frame, (x0, y), (x0 + conf_width, y + 20), (200, 180, 250), -1)

        # Compose label text: class name + count
        label_text = f"{TRACK_CLASSES[cls]}: {leaderboard_count[cls]}"
        frame = put_text_russian(frame, label_text, (x0 + 5, y + 2), font_size=font_size, color=(255, 255, 255))

    return frame

def create_new_writer():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{save_path}/drive_{timestamp}.mp4"
    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        FRAME_SIZE
    )
    return writer, timestamp, filename

def save_chunk_summary(timestamp):
    try:
        summary_file = f"{save_path}/counts_{timestamp}.txt"
        with open(summary_file, "w") as f:
            for cls, count in leaderboard_count.items():
                f.write(f"{TRACK_CLASSES[cls]}: {count}\n")
        print(f"[INFO] Summary saved: {summary_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary for chunk {timestamp}: {e}")

# -----------------------
# MAIN LOOP
# -----------------------

video_writer, chunk_timestamp, chunk_filename = create_new_writer()
chunk_start_time = time.time()

try:
    print(f"[INFO] Начало записи: {chunk_filename}")
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        # YOLO detection
        if frame_count % YOLO_INTERVAL == 0:
            results = model.predict(frame, verbose=False, imgsz=480, device="cpu")[0]
            current_boxes = []
            for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                cls = int(cls)
                if score >= CONFIDENCE_THRESHOLD and cls in TRACK_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        is_new_object(roi, cls, score, (x1, y1, x2, y2))
                    current_boxes.append((x1, y1, x2, y2, score, cls))
            last_boxes = current_boxes
        else:
            current_boxes = last_boxes

        # Draw bounding boxes and leaderboard
        for x1, y1, x2, y2, score, cls in current_boxes:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (23, 162, 184), 2)
            label = f"{TRACK_CLASSES[cls]}: {score:.2f}"
            frame = put_text_russian(frame, label, (x1, y1 - 25), font_size=18)
        frame = draw_leaderboard(frame)

        # Show and record
        cv2.imshow("Drive Recorder", frame)
        video_writer.write(cv2.resize(frame, FRAME_SIZE))

        # Check for chunk rotation
        if time.time() - chunk_start_time >= CHUNK_DURATION_MIN * 60:
            try:
                video_writer.release()
                print(f"[INFO] Chunk saved: {chunk_filename}")
                save_chunk_summary(chunk_timestamp)
            except Exception as e:
                print(f"[ERROR] Failed to save chunk {chunk_filename}: {e}")
            # Start new chunk
            video_writer, chunk_timestamp, chunk_filename = create_new_writer()
            chunk_start_time = time.time()
            print(f"[INFO] Начало новой записи: {chunk_filename}")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    try:
        video_writer.release()
    except Exception as e:
        print(f"[ERROR] Failed to release final video writer: {e}")
    cv2.destroyAllWindows()
    print("[INFO] Остановка записи.")
    print("[INFO] Итого подсчитано:")
    for cls, count in leaderboard_count.items():
        print(f"  {TRACK_CLASSES[cls]}: {count}")
    save_chunk_summary(chunk_timestamp)












