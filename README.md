# Football Match AI Analysis System

An end-to-end computer vision and machine learning pipeline built to extract tracking, movement, and tactical analytics from football match broadcast footage. 

This project follows the architecture popularized by [Abdullah Tarek's Football Analysis System](https://github.com/abdullahtarek/football_analysis), combining object detection, feature tracking, optical flow camera estimation, perspective transformation, and unsupervised team clustering into a modular pipeline.

---

## Key Features

* **Object Detection & Tracking**: Leverages YOLO-based object detection paired with custom tracking wrappers (`trackers/`) to maintain persistent IDs for players, referees, and the ball across frames.
* **Missing Data Interpolation**: Fills tracking gaps using linear interpolation (`pandas`) to accurately recover ball coordinates when occluded or out of frame.
* **Unsupervised Team Assignment**: Segments player jerseys using $K$-Means (`scikit-learn`) on top-half bounding box pixels to automatically assign team colors and IDs.
* **Camera Movement Estimation**: Calculates camera pan and tilt using Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) on static background features to isolate camera motion from player movement.
* **View Transformation**: Maps 2D pixel coordinates to real-world pitch meters using perspective transformation matrices (`view_transformer/`).
* **Speed & Distance Estimation**: Measures individual player movement metrics (speed in km/h and distance covered in meters) adjusted for camera panning.
* **Ball Possession Tracking**: Determines real-time ball possession per frame and calculates team possession percentages (`player_ball_assigner/`).
* **Performance Caching (Stubs)**: Uses `.pkl` stub files to cache intermediate detections and optical flow calculations, avoiding redundant model inference during visual testing.

---

## Project Architecture

```text
football/
├── camera_movement_estimator/
│   ├── __init__.py
│   └── camera_movement_estimator.py    # Optical flow-based camera motion estimation
├── development_and_analysis/           # Research notebooks, training, and color experiments
├── perspective_transform/ / view_transformer/
│   ├── __init__.py
│   └── view_transformer.py             # Perspective matrix mapping pixels to meters
├── player_ball_assigner/
│   ├── __init__.py
│   └── player_ball_assigner.py         # Ball possession logic based on proximity
├── speed_and_distance_estimator/
│   ├── __init__.py
│   └── speed_and_distance_estimator.py # Speed (km/h) and total distance calculation
├── stubs/                               # Cached detection and optical flow pickle files
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py                # K-Means clustering for kit color assignment
├── trackers/
│   ├── __init__.py
│   └── tracker.py                      # YOLO tracking, annotation, and interpolation
├── utils/
│   ├── __init__.py
│   ├── bbox_utils.py                   # Distance, spatial, and centroid utilities
│   └── video_utils.py                  # Frame reading and output video writing
├── input/                               # Input match video footage (.mp4)
├── output/                              # Rendered annotated output videos
├── main.py                             # Core pipeline orchestration script
├── requirements.txt                    # Python package dependencies
└── .gitignore                          # Ignored videos, weights, and binary stubs
```

---

## Prerequisites & Installation

### 1. Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and configure a virtual environment:

```bash
git clone https://github.com/your-username/football.git
cd football

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries, including OpenCV, Supervision, Scikit-Learn, PyTorch, and Pandas:

```bash
pip install -r requirements.txt
```

*(Optional)* If running inside Jupyter Notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name=football-env --display-name "Python (Football AI)"
```

---

## Usage Guide

### 1. Running the Main Pipeline
Place your target video inside `input/` (e.g., `input/sample.mp4`) and run the main entry point:

```bash
python main.py
```
The processed video with annotated team circles, camera tracking overlays, ball arrows, and player stats will be saved to `output/`.

### 2. Stub Caching System
To accelerate development iterations without re-running heavy YOLO inference every time:
* Set `read_from_stub=True` in `main.py` to load pre-calculated `.pkl` files from `stubs/`.
* Set `read_from_stub=False` (or delete files in `stubs/`) to re-trigger full AI detection and optical flow estimation.

---

## Technical Pipeline Overview

1. **Inference & Tracking**: Object bounding boxes are extracted and assigned persistent track IDs frame-by-frame.
2. **Color Clustering**: Top-half crops of detected players are passed to $K$-Means to differentiate team 1 from team 2.
3. **Camera Compensation**: Optical flow measures global camera shift, allowing local player displacements to reflect actual pitch movement.
4. **Metric Mapping**: Perspective transformation translates pixel coordinates into meters to calculate speed (km/h) and distance.

---

## Acknowledgments & Credits

* Architecture inspired by [Abdullah Tarek's Football Analysis System](https://github.com/abdullahtarek/football_analysis) [cite: 1.1.5, 1.1.8].

---

## License

This project is open-source and available under the [MIT License](LICENSE).