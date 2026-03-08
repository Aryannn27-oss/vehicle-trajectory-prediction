import cv2
import os
from src.tracker import ObjectTracker

video_path = "data/video.mp4"
output_path = "outputs/output.mp4"

os.makedirs("outputs", exist_ok=True)

tracker = ObjectTracker()

cap = cv2.VideoCapture(video_path)


frame_width = 1280
frame_height = 720

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

  
    frame = cv2.resize(frame, (frame_width, frame_height))

    tracks = tracker.process_frame(frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, r, b = track.to_ltrb()

        l, t, r, b = int(l), int(t), int(r), int(b)

        cv2.rectangle(frame, (l, t), (r, b), (0,255,0), 2)

       
        cv2.putText(
            frame,
            f"ID {track_id}",
            (l, t-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    out.write(frame)

cap.release()
out.release()


print("Video processing complete.")
print(f"Output saved to: {output_path}")
