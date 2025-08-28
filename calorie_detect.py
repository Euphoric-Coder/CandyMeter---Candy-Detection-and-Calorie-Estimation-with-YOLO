# detect_and_count_with_calories.py
import os
import sys
import argparse
import glob
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- Config: calories per detected item --------------------
CALORIES_PER_ITEM = {
    "MMs_peanut": 90,
    "MMs_regular": 73,
    "airheads": 60,
    "gummy_worms": 50,
    "milky_way": 80,
    "nerds": 50,
    "skittles": 60,
    "snickers": 80,
    "starbust": 40,  # matches your class name spelling
    "three_musketeers": 70,
    "twizzlers": 45,
}
# ---------------------------------------------------------------------------

# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    help='Path to YOLO model file (e.g., "runs/detect/train/weights/best.pt")',
)
parser.add_argument("--source", required=True, help="Image/video/folder/webcam source")
parser.add_argument(
    "--thresh", type=float, default=0.5, help="Min confidence threshold (default=0.5)"
)
parser.add_argument(
    "--resolution", default=None, help="WxH resolution for display/recording"
)
parser.add_argument(
    "--record",
    action="store_true",
    help="Record video/webcam output to demo1.avi (requires --resolution)",
)
args = parser.parse_args()

# Parse args
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Validate model
if not os.path.exists(model_path):
    print("ERROR: Invalid model path.")
    sys.exit(0)

# Load model + labels
model = YOLO(model_path, task="detect")
labels = model.names

# Detect source type
img_ext_list = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"}
vid_ext_list = {".avi", ".mov", ".mp4", ".mkv", ".wmv"}

if os.path.isdir(img_source):
    source_type = "folder"
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = "image"
    elif ext in vid_ext_list:
        source_type = "video"
    else:
        print(f"Unsupported file extension: {ext}")
        sys.exit(0)
elif "usb" in img_source:
    source_type = "usb"
    usb_idx = int(img_source[3:])
elif "picamera" in img_source:
    source_type = "picamera"
    picam_idx = int(img_source[8:])
else:
    print(f"Invalid --source: {img_source}")
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split("x"))
        resize = True
    except Exception:
        print('Invalid --resolution, use format "640x480".')
        sys.exit(0)

# Recording setup
if record:
    if source_type not in ["video", "usb"]:
        print("Recording works only for video/webcam.")
        sys.exit(0)
    if not user_res:
        print("Please specify --resolution to record.")
        sys.exit(0)
    record_name = "demo1.avi"
    record_fps = 30
    recorder = cv2.VideoWriter(
        record_name, cv2.VideoWriter_fourcc(*"MJPG"), record_fps, (resW, resH)
    )

# Init source
if source_type == "image":
    imgs_list = [img_source]
elif source_type == "folder":
    imgs_list = [
        f
        for f in glob.glob(os.path.join(img_source, "*"))
        if os.path.splitext(f)[1] in img_ext_list
    ]
    imgs_list.sort()
elif source_type in ("video", "usb"):
    cap_arg = img_source if source_type == "video" else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == "picamera":
    from picamera2 import Picamera2

    cap = Picamera2()
    if not user_res:
        resW, resH = 640, 480
        resize = True
    cap.configure(
        cap.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)})
    )
    cap.start()

# Colors
bbox_colors = [
    (164, 120, 87),
    (68, 148, 228),
    (93, 97, 209),
    (178, 182, 133),
    (88, 159, 106),
    (96, 202, 231),
    (159, 124, 168),
    (169, 162, 241),
    (98, 118, 150),
    (172, 176, 184),
]

# FPS tracking
avg_frame_rate = 0.0
frame_rate_buffer = deque(maxlen=200)
img_count = 0


def put_row(img, text, y, scale=0.7, color=(0, 255, 255), thickness=2):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ("image", "folder"):
        if img_count >= len(imgs_list):
            print("All images processed. Exiting.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == "video":
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
    elif source_type == "usb":
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print("Camera read failed.")
            break
    elif source_type == "picamera":
        frame = cap.capture_array()
        if frame is None:
            print("Picamera read failed.")
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    per_frame_counts = defaultdict(int)
    per_frame_calories = 0
    object_count = 0

    # Draw detections
    for i in range(len(detections)):
        conf = float(detections[i].conf.item())
        if conf < min_thresh:
            continue

        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels.get(classidx, str(classidx))

        color = bbox_colors[classidx % len(bbox_colors)]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        object_count += 1
        per_frame_counts[classname] += 1

        kcal = CALORIES_PER_ITEM.get(classname, 0)
        per_frame_calories += kcal

        label = f"{classname}: {int(conf*100)}%"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(ymin, th + 10)
        cv2.rectangle(
            frame,
            (xmin, y_text - th - 10),
            (xmin + tw, y_text + base - 10),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (xmin, y_text - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    # FPS
    t_stop = time.perf_counter()
    fps = 1.0 / max(1e-6, (t_stop - t_start))
    frame_rate_buffer.append(fps)
    avg_frame_rate = float(np.mean(frame_rate_buffer)) if frame_rate_buffer else fps

    # HUD
    row_y = 22
    put_row(frame, f"Objects: {object_count} | Frame kcal: {per_frame_calories}", row_y)
    row_y += 22
    if source_type in ("video", "usb", "picamera"):
        put_row(frame, f"FPS: {avg_frame_rate:0.2f}", row_y)
        row_y += 22
    if per_frame_counts:
        per_frame_line = "This frame: " + ", ".join(
            f"{k}:{v}" for k, v in sorted(per_frame_counts.items())
        )
        put_row(frame, per_frame_line, row_y, scale=0.55, thickness=1)
        row_y += 20

    # Display
    cv2.imshow("YOLO detection results", frame)
    if record and source_type in ("video", "usb"):
        recorder.write(frame)

    # Key handling
    if source_type in ("image", "folder"):
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key in (ord("q"), ord("Q")):
        break
    elif key in (ord("s"), ord("S")):
        cv2.waitKey()
    elif key in (ord("p"), ord("P")):
        cv2.imwrite("capture.png", frame)

# Cleanup
print(f"Average pipeline FPS: {avg_frame_rate:.2f}")
if source_type in ("video", "usb"):
    cap.release()
elif source_type == "picamera":
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
