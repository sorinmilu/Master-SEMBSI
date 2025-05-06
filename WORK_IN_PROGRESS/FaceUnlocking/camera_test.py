import cv2

# Try opening the first 5 cameras (you can adjust the range)
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available.")
        cap.release()
    else:
        print(f"Camera {i} is not available.")

        