import cv2
import mediapipe as mp
import numpy as np
import signal
import sys

# Initialize mediapipe pose and drawing
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate joint angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Curl & Row counter variables
curls_counter, rows_counter = 0, 0
curls_stage, rows_stage = None, None
current_exercise = "None"

# Graceful exit handler (to print final counts)
def signal_handler(sig, frame):
    print("\n\n========== Workout Summary ==========")
    print(f"Bicep Curls: {curls_counter}")
    print(f"Bent Over Rows: {rows_counter}")
    print("=====================================")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)   # for curls
            back_angle = calculate_angle(shoulder, hip, knee)       # for rows
            arm_angle = calculate_angle(hip, shoulder, elbow)       # for rows

            # ---------------- Detect Exercise ----------------
            # If person is upright -> likely curls
            if back_angle > 150:
                current_exercise = "Bicep Curl"
                # Curl logic
                if elbow_angle > 160:
                    curls_stage = "down"
                if elbow_angle < 30 and curls_stage == "down":
                    curls_stage = "up"
                    curls_counter += 1
                    print(f"Curls Reps: {curls_counter}")

            # If person is bent forward -> likely rows
            elif 40 < back_angle < 120:
                current_exercise = "Bent Over Row"
                # Row logic (elbow pulling backwards)
                if arm_angle > 70:
                    rows_stage = "down"
                if arm_angle < 40 and rows_stage == "down":
                    rows_stage = "up"
                    rows_counter += 1
                    print(f"Rows Reps: {rows_counter}")

            # --------------------------------------------------

            # Visualize elbow angle
            cv2.putText(image, str(int(elbow_angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        # Setup status box
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)

        # Exercise Name
        cv2.putText(image, current_exercise, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Curl count
        cv2.putText(image, f"Curls: {curls_counter}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Row count
        cv2.putText(image, f"Rows: {rows_counter}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66),
                                   thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230),
                                   thickness=2, circle_radius=2)
        )

        # Show feed
        cv2.imshow('Workout Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
