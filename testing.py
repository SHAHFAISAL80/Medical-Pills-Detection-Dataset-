import cv2
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO(r'medical_pills_model.pt')

# Path to the video file
video_path = r'6191838-uhd_4096_2160_25fps.mp4'

# Start video capture
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Desired processing size
target_width = 1280  # Set for HD resolution
target_height = 720

# Output video properties
fps = cap.get(cv2.CAP_PROP_FPS)
output_filename = 'output_video.mp4'

# Define the video writer for .mp4 format with H.264 codec
output_video = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Change to 'H264' if needed
    fps,
    (target_width, target_height)
)

# For FPS calculation
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Resize the frame to HD (1280x720) resolution
    frame = cv2.resize(frame, (target_width, target_height))

    # Perform detection
    try:
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
    except Exception as e:
        print(f"Error during model prediction: {e}")
        break

    # Calculate FPS
    current_time = time.time()
    fps_display = int(1 / (current_time - prev_time))
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(annotated_frame, f'FPS: {fps_display}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 180), 2)

    # Display the frame
    cv2.imshow('medical-pills detection', annotated_frame)

    # Write the frame to the output video
    output_video.write(annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Video processing completed. Output saved as {output_filename}")
