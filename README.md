# Vision-Based Vehicle Trajectory Prediction 🚗📈

This project implements a **vision-based multi-object tracking and trajectory prediction system** using modern computer vision techniques.

The system detects vehicles in a video, tracks them across frames, and predicts their motion using **Kalman Filters and an experimental LSTM-based trajectory predictor**.

⚠️ **Note:** The LSTM-based prediction module is currently **under development and not fully trained yet**. At the moment, Kalman-based motion prediction is the primary prediction mechanism.


## Project Overview

The system processes a video and performs the following tasks:

1. **Vehicle Detection** using YOLOv8
2. **Multi-object Tracking** using DeepSORT
3. **Trajectory Extraction** from tracked vehicle positions
4. **Trajectory Smoothing** using spline interpolation
5. **Motion Prediction**
   - Kalman Filter (physics-based prediction)
   - LSTM Neural Network (learning-based prediction – under development)

The output video visualizes:

- vehicle bounding boxes
- trajectory paths
- predicted motion positions

---

## System Pipeline

Video Input  
↓  
YOLOv8 Vehicle Detection  
↓  
DeepSORT Multi-object Tracking  
↓  
Trajectory Extraction  
↓  
Trajectory Smoothing  
↓  
Kalman Motion Prediction  
↓  
Experimental LSTM Motion Prediction  
↓  
Visualization Output  

---

## Example Output

The output visualization includes:

- **Bounding Boxes** → detected vehicles  
- **Blue Curved Lines** → trajectory history  
- **Red Dot** → Kalman predicted position  
- **Green Dot** → LSTM predicted position (experimental)

The processed video is saved in:

outputs/output.mp4

---

## Project Structure

vision_trajectory_prediction/

data/
- video.mp4

models/
- yolov8n.pt

outputs/
- output.mp4
![Demo](outputs/demo.gif)

src/
- tracker.py
- kalman_predictor.py
- trajectory_lstm.py

main.py  
requirements.txt  
README.md  

---

## Installation

Clone the repository:

git clone https://github.com/Aryannn27-oss/vision-trajectory-prediction.git  
cd vision-trajectory-prediction  

Install dependencies:

pip install -r requirements.txt

---

## Running the Project

Place a video inside the **data/** folder and run:

python main.py

The processed output video will be saved to:

outputs/output.mp4

---

## Technologies Used

Python  
OpenCV  
PyTorch  
YOLOv8 (Ultralytics)  
DeepSORT Tracking  
Kalman Filter  
LSTM Neural Networks  
NumPy  
SciPy  

---

## Applications

Traffic monitoring  
Autonomous driving research  
Smart city analytics  
Vehicle behavior analysis  
Surveillance systems  

---

## Future Improvements

- Training the LSTM model on real trajectory datasets
- Vehicle speed estimation from trajectories
- Trajectory clustering and traffic flow analysis
- Real-time video processing
- Trajectory heatmap visualization

---

## Author

Aryan Verma  
B.Tech Chemical Engineering  
IIT (BHU) Varanasi
