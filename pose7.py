"""
Full Workout Tracker (menu-driven)
- Numeric menu for muscle groups + exercises
- Rep & set counting with angle heuristics for many exercises
- Miscellaneous mode counts generic reps using wrist/hip vertical movement
- Rest timer, calories estimate, session summary & CSV logging

Requirements:
pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import defaultdict

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- Utility functions ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2))

def safe_landmark(landmarks, idx):
    try:
        lm = landmarks[idx.value]
        return [lm.x, lm.y, lm.z]
    except:
        return None

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ---------- Session logging ----------
def log_session(session_summary, filename="workout_log.csv"):
    header = ["timestamp", "muscle_group", "exercise", "sets_completed", "target_sets", "total_reps", "calories"]
    write_header = False
    try:
        with open(filename, "r", newline="") as f:
            pass
    except FileNotFoundError:
        write_header = True

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([now_str(),
                         session_summary["group"],
                         session_summary["exercise"],
                         session_summary["sets_completed"],
                         session_summary["target_sets"],
                         session_summary["total_reps"],
                         session_summary["calories"]])

# ---------- Exercise detectors ----------
# Each detector returns: (stage, counter, sets, feedback, total_reps_increment_flag)
# total_reps_increment_flag is used when we want to add to total reps for calories & summary

def detector_bicep_curl(landmarks, stage, counter):
    # uses elbow angle (shoulder-elbow-wrist)
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    e = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    w = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    if not (s and e and w):
        return stage, counter, 0, "", False
    angle = calculate_angle(s[:2], e[:2], w[:2])
    feedback = ""
    done = False
    if angle > 160:
        stage = "down"
    if angle < 40 and stage == "down":
        stage = "up"
        counter += 1
        feedback = "Nice curl!"
        done = True
    return stage, counter, 0, feedback, done

def detector_hammer_curl(landmarks, stage, counter):
    # same elbow logic but we can also check wrist orientation by x difference.
    return detector_bicep_curl(landmarks, stage, counter)

def detector_squat(landmarks, stage, counter):
    # knee angle (hip-knee-ankle)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    k = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    a = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    if not (h and k and a):
        return stage, counter, 0, "", False
    angle = calculate_angle(h[:2], k[:2], a[:2])
    feedback = ""
    done = False
    # angle ~ 180 straight, lower angle -> deeper squat
    if angle > 160:
        stage = "up"
    if angle < 100 and stage == "up":
        stage = "down"
        counter += 1
        feedback = "Good squat rep!"
        done = True
    # form feedback
    if angle > 175:
        feedback = "Stand taller"
    elif angle < 80:
        feedback = "Deep squat - control it"
    return stage, counter, 0, feedback, done

def detector_front_squat(landmarks, stage, counter):
    # similar to squat; user front-loaded weight - same heuristics
    return detector_squat(landmarks, stage, counter)

def detector_rdl(landmarks, stage, counter):
    # Romanian deadlift: hinge at hip. Use shoulder-hip-knee angle (back angle)
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    k = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    if not (s and h and k):
        return stage, counter, 0, "", False
    angle = calculate_angle(s[:2], h[:2], k[:2])  # back angle around hip
    feedback = ""
    done = False
    # standing straight angle ~ 180; hip hinge reduces angle
    if angle > 160:
        stage = "up"
    if angle < 120 and stage == "up":
        stage = "down"
        counter += 1
        feedback = "Good hinge!"
        done = True
    if angle < 90:
        feedback = "Too much bend; keep back straight"
    return stage, counter, 0, feedback, done

def detector_deadlift(landmarks, stage, counter):
    return detector_rdl(landmarks, stage, counter)

def detector_bent_over_row(landmarks, stage, counter):
    # Use arm pull: hip-shoulder-elbow or shoulder-hip-knee to ensure bent over
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    e = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    if not (s and h and e):
        return stage, counter, 0, "", False
    arm_angle = calculate_angle(h[:2], s[:2], e[:2])
    feedback = ""
    done = False
    # detect pull: arm_angle small when elbow pulled to torso
    if arm_angle > 160:
        stage = "down"
    if arm_angle < 60 and stage == "down":
        stage = "up"
        counter += 1
        feedback = "Row rep!"
        done = True
    return stage, counter, 0, feedback, done

def detector_pullup(landmarks, stage, counter, prev_shoulder_y):
    # detect upward movement of shoulders relative to hips
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    if not (s and h):
        return stage, counter, 0, "", False, prev_shoulder_y
    shoulder_y = s[1]
    hip_y = h[1]
    feedback = ""
    done = False
    # if shoulder goes substantially above hip (smaller y), count as a pull-up
    if prev_shoulder_y is None:
        prev_shoulder_y = shoulder_y
        return stage, counter, 0, "", False, prev_shoulder_y
    vertical_movement = prev_shoulder_y - shoulder_y
    # threshold based on normalized coordinates
    if vertical_movement > 0.06 and stage != "up":
        stage = "up"
        # wait for down to count completion
    if stage == "up" and (shoulder_y - hip_y) > 0.08:  # returned back down
        stage = "down"
        counter += 1
        feedback = "Pull-up!"
        done = True
    prev_shoulder_y = shoulder_y
    return stage, counter, 0, feedback, done, prev_shoulder_y

def detector_glute_bridge(landmarks, stage, counter):
    # detect hip extension: angle at hip (shoulder-hip-knee)
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    k = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    if not (s and h and k):
        return stage, counter, 0, "", False
    angle = calculate_angle(s[:2], h[:2], k[:2])
    feedback = ""
    done = False
    # lying down angle small at rest, larger when bridge (hip extension) - thresholds may vary
    if angle > 120 and stage != "up":
        stage = "up"
    if angle < 100 and stage == "up":
        stage = "down"
        counter += 1
        feedback = "Glute bridge rep!"
        done = True
    return stage, counter, 0, feedback, done

def detector_side_bend(landmarks, stage, counter):
    # detect torso lateral bend - compare shoulder x relative to hip x or torso angle
    left_sh = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_sh = safe_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = safe_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    if not (left_sh and right_sh and left_hip and right_hip):
        return stage, counter, 0, "", False
    # torso lean measured by difference in shoulder y's maybe - simpler: use distance between shoulders and hips vertical offset
    torso_center_y = (left_sh[1] + right_sh[1]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2
    diff = torso_center_y - hip_center_y
    feedback = ""
    done = False
    # if torso center moves down/up significantly, count as bend (very rough)
    if diff > 0.04 and stage != "down":
        stage = "down"
    if diff < 0.01 and stage == "down":
        stage = "up"
        counter += 1
        feedback = "Side bend rep"
        done = True
    return stage, counter, 0, feedback, done

def detector_knees_to_chest(landmarks, stage, counter):
    # for lying knees-to-chest, detect knee y movement upward (approx)
    left_k = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    left_hip = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    if not (left_k and left_hip):
        return stage, counter, 0, "", False
    diff = left_hip[1] - left_k[1]  # larger when knees pulled in (knee above hip)
    feedback = ""
    done = False
    if diff > 0.08 and stage != "in":
        stage = "in"
    if diff < 0.04 and stage == "in":
        stage = "out"
        counter += 1
        feedback = "Knees-to-chest rep"
        done = True
    return stage, counter, 0, feedback, done

def detector_tricep_dip(landmarks, stage, counter):
    # use elbow angle similar to push-downs but ensure vertical torso (hip shoulder y close)
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    e = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    w = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    if not (s and e and w and h):
        return stage, counter, 0, "", False
    angle = calculate_angle(s[:2], e[:2], w[:2])
    torso_diff = abs(s[1] - h[1])
    feedback = ""
    done = False
    # dips: elbow angle reduces
    if angle > 160:
        stage = "up"
    if angle < 70 and stage == "up":
        stage = "down"
        counter += 1
        feedback = "Dip rep"
        done = True
    if torso_diff > 0.25:
        feedback = "Body too tilted"
    return stage, counter, 0, feedback, done

def detector_overhead_tricep_extension(landmarks, stage, counter):
    # detect elbow angle with arm overhead (use right or left shoulder-elbow-wrist)
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    e = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    w = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    if not (s and e and w):
        return stage, counter, 0, "", False
    angle = calculate_angle(s[:2], e[:2], w[:2])
    feedback = ""
    done = False
    if angle > 160:
        stage = "down"
    if angle < 90 and stage == "down":
        stage = "up"
        counter += 1
        feedback = "Tricep extension rep"
        done = True
    return stage, counter, 0, feedback, done

def detector_skull_crusher(landmarks, stage, counter):
    return detector_overhead_tricep_extension(landmarks, stage, counter)

def detector_pushup_like(landmarks, stage, counter):
    # use shoulder-hip vertical distance to detect push up down
    s = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    h = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    if not (s and h):
        return stage, counter, 0, "", False
    diff = s[1] - h[1]  # reduces when body lowers
    feedback = ""
    done = False
    if diff < 0.02 and stage != "down":
        stage = "down"
    if diff > 0.05 and stage == "down":
        stage = "up"
        counter += 1
        feedback = "Push-up-ish rep"
        done = True
    return stage, counter, 0, feedback, done

# Misc detector: counts generic reps based on wrist vertical movement peaks
def detector_misc(landmarks, misc_state):
    # misc_state: dict storing prev_wrist_y and cooldown
    lw = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = safe_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
    hip = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    if not (lw and rw and hip):
        return misc_state, False, ""
    wrist_y = (lw[1] + rw[1]) / 2
    done = False
    feedback = ""
    prev = misc_state.get("prev_wrist_y", None)
    cooldown = misc_state.get("cooldown", 0)
    if prev is None:
        misc_state["prev_wrist_y"] = wrist_y
        return misc_state, False, ""
    movement = prev - wrist_y  # upward movement positive
    # threshold tuned for normalized coords
    if movement > 0.05 and cooldown <= 0:
        done = True
        feedback = "Misc rep"
        misc_state["cooldown"] = 10
    misc_state["prev_wrist_y"] = wrist_y
    if misc_state.get("cooldown", 0) > 0:
        misc_state["cooldown"] -= 1
    return misc_state, done, feedback

# ---------- Map groups & exercises ----------
muscle_groups = {
    1: ("Legs", {
        1: ("Barbell Back Squat", detector_squat),
        2: ("Front Squat", detector_front_squat),
        3: ("Romanian Deadlift", detector_rdl),
    }),
    2: ("Biceps", {
        1: ("Concentration Curls (short head)", detector_bicep_curl),
        2: ("Hammer Curls (long head)", detector_hammer_curl),
        3: ("EZ Bar Curls", detector_bicep_curl),
        4: ("Preacher Curl", detector_bicep_curl),
        5: ("Single Arm High Cable Bicep Curl", detector_bicep_curl),
    }),
    3: ("Triceps", {
        1: ("Overhead Tricep Extension", detector_overhead_tricep_extension),
        2: ("Tricep Dips", detector_tricep_dip),
        3: ("Skull Crushers", detector_skull_crusher),
        4: ("Tricep Pushdown (approx)", detector_overhead_tricep_extension),
        5: ("Close-Grip Bench Press (approx)", detector_pushup_like),
        6: ("Diamond Push Ups (approx)", detector_pushup_like),
    }),
    4: ("Back", {
        1: ("Glute Bridge", detector_glute_bridge),
        2: ("Knees to Chest", detector_knees_to_chest),
        3: ("Child's Pose (misc)", lambda lm, st, c: (st, c, 0, "Relax pose", False)),
        4: ("Barbell Bent-over Row", detector_bent_over_row),
        5: ("Kettlebell Swings (approx)", detector_rdl),
        6: ("Barbell Deadlift", detector_deadlift),
        7: ("Pull-up", None),  # pull-up handled specially
        8: ("Dumbbell Single-arm Row", detector_bent_over_row),
        9: ("Chest-supported Dumbbell Row", detector_bent_over_row),
        10: ("Inverted Row", detector_bent_over_row),
    }),
    5: ("Sides/Abs", {
        1: ("Side Bends", detector_side_bend),
        2: ("Knees to Chest (abs)", detector_knees_to_chest),
        3: ("Plank (misc)", lambda lm, st, c: (st, c, 0, "Hold plank", False)),
    }),
    6: ("Miscellaneous", {
        1: ("Misc Generic Exercise (count reps)", "MISC"),
    }),
}

# Calories per rep (very rough approximations)
calories_per_rep = defaultdict(lambda: 0.2, {
    "Barbell Back Squat": 0.35,
    "Front Squat": 0.35,
    "Romanian Deadlift": 0.32,
    "Concentration Curls (short head)": 0.18,
    "Hammer Curls (long head)": 0.20,
    "EZ Bar Curls": 0.18,
    "Preacher Curl": 0.18,
    "Single Arm High Cable Bicep Curl": 0.18,
    "Overhead Tricep Extension": 0.18,
    "Tricep Dips": 0.25,
    "Skull Crushers": 0.2,
    "Tricep Pushdown (approx)": 0.18,
    "Close-Grip Bench Press (approx)": 0.28,
    "Diamond Push Ups (approx)": 0.25,
    "Glute Bridge": 0.25,
    "Knees to Chest": 0.12,
    "Barbell Bent-over Row": 0.3,
    "Kettlebell Swings (approx)": 0.4,
    "Barbell Deadlift": 0.45,
    "Pull-up": 0.6,
    "Dumbbell Single-arm Row": 0.28,
    "Chest-supported Dumbbell Row": 0.28,
    "Inverted Row": 0.25,
    "Side Bends": 0.12,
})

# ---------- Menu helpers ----------
def select_muscle_group():
    print("\nChoose Muscle Group:")
    for k, (name, _) in muscle_groups.items():
        print(f"{k}. {name}")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
            if choice in muscle_groups:
                return choice
        except:
            pass
        print("Invalid choice. Try again.")

def select_exercise(group_choice):
    group_name, exercises = muscle_groups[group_choice]
    print(f"\nSelected: {group_name}. Choose Exercise:")
    for k, v in exercises.items():
        if v == "MISC":
            print(f"{k}. Misc Generic Exercise (counts reps)")
        else:
            print(f"{k}. {v[0]}")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
            if choice in exercises:
                return choice
        except:
            pass
        print("Invalid choice. Try again.")

# ---------- Rest timer ----------
def rest_timer(seconds, window_image=None):
    # non-blocking GUI-friendly rest timer: we will block but we print and sleep (simple)
    for i in range(seconds, 0, -1):
        print(f"Resting: {i}s", end="\r")
        time.sleep(1)
    print(" " * 30, end="\r")

# ---------- Main app ----------
def main():
    print("=== Workout Tracker: full version ===")
    group_choice = select_muscle_group()
    exercise_choice = select_exercise(group_choice)

    group_name, exercises = muscle_groups[group_choice]
    ex_entry = exercises[exercise_choice]

    if ex_entry == "MISC":
        exercise_name = "Misc Generic Exercise (user-defined)"
        detector = None
    else:
        exercise_name = ex_entry[0]
        detector = ex_entry[1]

    # Numeric shortcuts accepted: user typed numbers already
    target_reps = int(input("Target reps per set (e.g., 10): ").strip())
    target_sets = int(input("Target sets (e.g., 3): ").strip())
    rest_between_sets = int(input("Rest between sets in seconds (e.g., 30): ").strip())

    # State variables
    stage = None
    counter = 0
    sets_completed = 0
    total_reps_done = 0
    misc_state = {"prev_wrist_y": None, "cooldown": 0}
    prev_shoulder_y = None  # used for pull-up
    prev_time = time.time()

    cap = cv2.VideoCapture(0)
    full_screen_try = True
    try:
        cv2.namedWindow('Workout Tracker', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Workout Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        full_screen_try = False

    print(f"\nStart: {group_name} -> {exercise_name}")
    print("Press 'q' in the window to quit early.\n")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Special-case for pull-up which uses a different detector signature
        is_pullup = (exercise_name == "Pull-up")

        while cap.isOpened() and sets_completed < target_sets:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from webcam.")
                break

            # Resize to keep UI consistent - you can change
            frame = cv2.resize(frame, (960, 540))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            feedback = ""
            counted_this_frame = False

            try:
                landmarks = results.pose_landmarks.landmark
                if ex_entry == "MISC":
                    misc_state, done, feedback = detector_misc(landmarks, misc_state)
                    if done:
                        counter += 1
                        total_reps_done += 1
                        counted_this_frame = True
                else:
                    if is_pullup:
                        # pull-up special
                        stage, counter, _, feedback, done, prev_shoulder_y = detector_pullup(
                            landmarks, stage, counter, prev_shoulder_y
                        )
                        if done:
                            total_reps_done += 1
                            counted_this_frame = True
                    else:
                        # detectors expect (landmarks, stage, counter)
                        res = detector(landmarks, stage, counter)
                        # res may be 3-tuple or 5-tuple depending on detector signature
                        if isinstance(res, tuple) and len(res) >= 5:
                            # (stage, counter, sets_inc, feedback, done)
                            stage, counter, _, feedback, done = res[:5]
                        else:
                            stage, counter, _, feedback, done = res[0], res[1], 0, res[3] if len(res) > 3 else "", res[4] if len(res) > 4 else False
                        if done:
                            total_reps_done += 1
                            counted_this_frame = True

                # set completion
                if counter >= target_reps:
                    sets_completed += 1
                    print(f"\nSet {sets_completed}/{target_sets} done. Starting rest for {rest_between_sets}s.")
                    # accumulate reps exactly equal to target_reps
                    # (counter may exceed target_reps slightly; we treat target as set)
                    # reset counter for next set
                    counter = 0
                    # Rest
                    rest_timer(rest_between_sets)
                    # reset stage so the next set starts clean
                    stage = None
                    # small pause so we don't immediately count
                    misc_state["prev_wrist_y"] = None

            except Exception as e:
                # no landmarks or unexpected error - ignore but continue
                pass

            # UI overlay
            cv2.rectangle(image, (0, 0), (420, 140), (36, 120, 200), -1)
            cv2.putText(image, f"{group_name} : {exercise_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"Sets: {sets_completed}/{target_sets}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"Reps (cur set): {counter}/{target_reps}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"Total reps: {total_reps_done}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"Feedback: {feedback}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Workout Tracker", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("\nUser requested quit.")
                break

        cap.release()
        cv2.destroyAllWindows()

    # summary
    # calories computed using total_reps_done and calories_per_rep lookup
    cals = round(total_reps_done * calories_per_rep[exercise_name], 2)
    print("\n\n========== Session Summary ==========")
    print(f"Group      : {group_name}")
    print(f"Exercise   : {exercise_name}")
    print(f"Sets done  : {sets_completed}/{target_sets}")
    print(f"Total reps : {total_reps_done}")
    print(f"Calories ~ : {cals} kcal")
    print("=====================================")

    # log session
    session_summary = {
        "group": group_name,
        "exercise": exercise_name,
        "sets_completed": sets_completed,
        "target_sets": target_sets,
        "total_reps": total_reps_done,
        "calories": cals
    }
    log_session(session_summary)
    print("Session logged to workout_log.csv")

if __name__ == "__main__":
    main()
