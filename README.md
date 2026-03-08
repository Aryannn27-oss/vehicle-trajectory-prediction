Vision-Based Vehicle Trajectory Prediction 🚗📈
Built a vehicle detection + tracking + prediction system using YOLOv8 + DeepSORT. Tracks cars in videos, smooths their paths, and predicts where they'll go next using Kalman Filters (working now) and LSTM (still training).

Quick note: LSTM part is WIP – not trained properly yet. Kalman is doing the heavy lifting for now.

What it does
Detects vehicles with YOLOv8

Tracks them across frames with DeepSORT

Extracts trajectories from track positions

Smooths paths with spline interpolation

Predicts motion – Kalman (main) + LSTM (experimental)

Output video shows:

Yellow bounding boxes around cars

Blue curved lines = actual trajectory

Red dot = Kalman prediction

Green dot = LSTM prediction

Saved as outputs/output.mp4

Pipeline
text
Video → YOLO Detection → DeepSORT Tracking → 
Trajectory → Smoothing → Kalman/LSTM Prediction → 
Output Video
Project Structure
text
vision_trajectory_prediction/
├── data/video.mp4              # your input video
├── models/yolov8n.pt           # download from ultralytics
├── outputs/output.mp4          # processed video
├── src/
│   ├── tracker.py
│   ├── kalman_predictor.py
│   └── trajectory_lstm.py
├── main.py
└── requirements.txt
Setup & Run
bash
git clone https://github.com/Aryannn27-oss/vision-trajectory-prediction.git
cd vision-trajectory-prediction

# Download YOLOv8 model
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/

pip install -r requirements.txt
Drop your video in data/ and run:

bash
python main.py
Tech Stack
Python, OpenCV, PyTorch

YOLOv8 (Ultralytics)

DeepSORT tracking

Kalman Filter + LSTM

NumPy, SciPy

What you see in output
Real-world uses
Traffic analysis

Self-driving car research

Surveillance

Smart city stuff

Next steps (TODO)
Train LSTM properly on trajectory datasets

Add vehicle speed estimation

Real-time processing

Traffic flow analysis

Heatmaps

About me
Aryan Verma
B.Tech Chemical Engineering
IIT (BHU) Varanasi
