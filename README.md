# Candy Detection & Calorie Estimation with YOLO

This project is a **computer vision application** that uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) to detect various candy types in **images, videos, and live camera feeds**, while also estimating the **calorie count per frame**.

The system can:
- Detect 11 different candy classes.
- Count the number of detected candies per frame.
- Estimate total calories in the current frame based on pre-defined calorie values.
- Work with single images, image folders, video files, or live webcam streams.
- (Optional) Record processed video with bounding boxes and calorie counts.

---

## Dataset & Classes

The dataset is organized with YOLO format:

```plaintext
dataset/
├── train
├── labels
└── classes.txt
```

The following candy classes are supported:

- `MMs_peanut`
- `MMs_regular`
- `airheads`
- `gummy_worms`
- `milky_way`
- `nerds`
- `skittles`
- `snickers`
- `starbust`
- `three_musketeers`
- `twizzlers`

Each class is associated with an **average calorie value** (per fun-size pack or serving):

| Candy              | Calories (approx) |
|--------------------|--------------------|
| MMs_peanut         | 90 kcal           |
| MMs_regular        | 73 kcal           |
| airheads           | 60 kcal           |
| gummy_worms        | 50 kcal           |
| milky_way          | 80 kcal           |
| nerds              | 50 kcal           |
| skittles           | 60 kcal           |
| snickers           | 80 kcal           |
| starbust           | 40 kcal           |
| three_musketeers   | 70 kcal           |
| twizzlers          | 45 kcal           |

---

## Features

- **Object Detection:** Uses YOLO11 model for real-time detection.  
- **Per-Frame Analysis:** Displays number of objects and estimated calories for each frame.  
- **Supports Multiple Sources:** Can work with:
  - Single image
  - Image folder
  - Video file
  - USB webcam (`usb0`)
  - Raspberry Pi Camera (`picamera0`)
- **Overlay HUD:** Shows bounding boxes, confidence %, object count, FPS, and per-frame calorie estimate.  
- **Recording Option:** Have the option to save processed video when running on webcam or video source.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Euphoric-Coder/CandyMeter---Candy-Detection-and-Calorie-Estimation-with-YOLO.git
```

```bash
cd CandyMeter---Candy-Detection-and-Calorie-Estimation-with-YOLO
```

```bash
python -m venv VENV
```
### (on macOS/Linux)
```bash
source VENV/bin/activate   
```
### (on Windows)
```bash
VENV\Scripts\activate      
```
Then install the dependencies for the project from the requirements.txt file

```bash
pip install --upgrade -r requirements.txt
```

## Usage
### 1. Split Dataset
Before training YOLO, split your dataset into **train** and **validation** sets using the provided script.  
This ensures that the model is trained on one subset and evaluated on another.

1. **--datapath**: path to your dataset folder containing images/ and labels/.   

2. **--train_pct**: fraction of data to use for training (for this, we will use 80% for training / 20% validation).
```bash
python split_dataset.py --datapath ./dataset --train_pct 0.8
```
The script will create the following structure:
```plaintext
dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  └── validation/
      ├── images/
      └── labels/
```
### 2. Create YOLO Config File (YAML)
YOLO requires a configuration file (data.yaml) that specifies dataset paths, number of classes, and class names.

Generate this automatically with the given `generate_yaml.py` file using the following command:
```bash
python generate_yaml.py
```
This will create a data.yaml file (inside dataset/) similar to:
```plaintext
path: ./dataset
train: train/images
val: validation/images
nc: 11
names:
- MMs_peanut
- MMs_regular
- airheads
- gummy_worms
- milky_way
- nerds
- skittles
- snickers
- starbust
- three_musketeers
- twizzlers
```
### 3. Train Model
Train the YOLO11 model on the dataset:
```bash
yolo detect train data=./dataset/data.yaml model=yolo11s.pt epochs=60 imgsz=640
```
1. **data**: path to your generated YAML file.
2. **model**: base model checkpoint (e.g., yolo11s.pt or yolo11l.pt).
3. **epochs** → number of training iterations.
4. **imgsz** → input image size (default 640x640).

After training, the best model weights will be saved at:
```bash
runs/detect/train/weights/best.pt
```
### 4. Run Detection & Calorie Estimation
Once training is complete, run inference with the detection + calorie counting script.
```bash
python detect_candy_with_calories.py --model runs/detect/train/weights/best.pt --source <SOURCE> --thresh 0.5
```
Where `<SOURCE>` can be:
1. Single Image
```bash
python detect_candy_with_calories.py --model runs/detect/train/weights/best.pt --source test.jpg --thresh 0.4
```
Detect candies and estimate calories in a single image.

2. Folder of Images
```bash
python detect_candy_with_calories.py --model runs/detect/train/weights/best.pt --source ./test_images --thresh 0.5
```
Process all images in a folder one by one.

3. Video File
```bash
python detect_candy_with_calories.py --model runs/detect/train/weights/best.pt --source demo.mp4 --resolution 1280x720
```
Run detection on a video and display results frame by frame.

4. Webcam (USB Camera)
```bash
python detect_candy_with_calories.py --model runs/detect/train/weights/best.pt --source usb0 --resolution 1280x720 --record
```
Run live detection on webcam feed.
**--record** saves the processed video as demo1.avi.
**--resolution** sets display & recording size.

5. Raspberry Pi Camera
```bash
python detect_candy_with_calories.py --model runs/detect/train/weights/best.pt --source picamera0 --resolution 640x480
```
Run detection on a PiCamera source.
