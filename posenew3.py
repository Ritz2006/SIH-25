import cv2
import mediapipe as mp
import numpy as np
import signal
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Counters & stages
curls_counter, rows_counter, squats_counter = 0, 0, 0
curls_stage, rows_stage, squats_stage = None, None, None
current_exercise = "None"

# Graceful exit summary
def signal_handler(sig, frame):
    print("\n\n========== Workout Summary ==========")
    print(f"Bicep Curls: {curls_counter}")
    print(f"Bent Over Rows: {rows_counter}")
    print(f"Squats: {squats_counter}")
    print("=====================================")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Key joints
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
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)   # curls
            back_angle = calculate_angle(shoulder, hip, knee)       # rows
            arm_angle = calculate_angle(hip, shoulder, elbow)       # rows
            knee_angle = calculate_angle(hip, knee, ankle)          # squats

            # ---------------- Detect Exercise ----------------
            if 40 < back_angle < 120:   # Bent over posture
                current_exercise = "Bent Over Row"
                if arm_angle > 70:
                    rows_stage = "down"
                if arm_angle < 40 and rows_stage == "down":
                    rows_stage = "up"
                    rows_counter += 1
                    print(f"Rows Reps: {rows_counter}")

            elif knee_angle < 140:      # Legs bending = squat
                current_exercise = "Squat"
                if knee_angle > 160:
                    squats_stage = "up"
                if knee_angle < 90 and squats_stage == "up":
                    squats_stage = "down"
                    squats_counter += 1
                    print(f"Squat Reps: {squats_counter}")

            else:                       # Default: curls
                current_exercise = "Bicep Curl"
                if elbow_angle > 160:
                    curls_stage = "down"
                if elbow_angle < 30 and curls_stage == "down":
                    curls_stage = "up"
                    curls_counter += 1
                    print(f"Curls Reps: {curls_counter}")

            # --------------------------------------------------

        except:
            pass

        # Status box
        cv2.rectangle(image, (0, 0), (320, 120), (245, 117, 16), -1)

        cv2.putText(image, current_exercise, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Curls: {curls_counter}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Rows: {rows_counter}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Squats: {squats_counter}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        cv2.imshow('Workout Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
