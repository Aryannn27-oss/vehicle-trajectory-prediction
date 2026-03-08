🚗📈 Vision-Based Vehicle Trajectory Prediction
<small>Built vehicle detection + tracking + prediction using YOLOv8 + DeepSORT. Tracks cars, smooths paths, predicts motion with Kalman (working) + LSTM (WIP).</small>

⚠️ LSTM under training – Kalman is main prediction now.

What It Does
<small>1. YOLOv8 detects vehicles
2. DeepSORT tracks across frames
3. Extracts + smooths trajectories
4. Kalman/LSTM motion prediction</small>

Output shows: Yellow boxes, blue paths, red/green prediction dots → outputs/output.mp4

Pipeline
text
<small>Video → YOLO → DeepSORT → Trajectory → 
Smoothing → Kalman/LSTM → Output Video</small>
Project Structure
text
<small>
vision_trajectory_prediction/
├── data/video.mp4
├── models/yolov8n.pt
├── outputs/output.mp4
├── src/tracker.py
├── src/kalman_predictor.py
├── src/trajectory_lstm.py
├── main.py
└── requirements.txt
</small>
Setup & Run
bash
<small>git clone https://github.com/Aryannn27-oss/vision-trajectory-prediction.git<br>
cd vision_trajectory-prediction<br>
pip install -r requirements.txt</small>
<small>Download yolov8n.pt → models/
Drop video in data/ → python main.py</small>

Tech Stack
<small>Python - OpenCV - PyTorch - YOLOv8 - DeepSORT - Kalman - LSTM - NumPy - SciPy</small>

Output Demo
Applications
<small>Traffic analysis - Autonomous driving - Surveillance - Smart cities</small>

TODO
<small>- Train LSTM properly
- Vehicle speed estimation
- Real-time processing
- Traffic flow analysis
- Heatmaps</small>

About
<small>Aryan Verma
B.Tech Chemical Engineering
IIT (BHU) Varanasi
