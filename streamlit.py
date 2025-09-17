import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Workout Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- Streamlit App ----------
def main():
    st.title("💪 Workout Tracker")
    
    # Initialize session state
    if 'workout_started' not in st.session_state:
        st.session_state.workout_started = False
    if 'stage' not in st.session_state:
        st.session_state.stage = None
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'sets_completed' not in st.session_state:
        st.session_state.sets_completed = 0
    if 'total_reps_done' not in st.session_state:
        st.session_state.total_reps_done = 0
    if 'misc_state' not in st.session_state:
        st.session_state.misc_state = {"prev_wrist_y": None, "cooldown": 0}
    if 'prev_shoulder_y' not in st.session_state:
        st.session_state.prev_shoulder_y = None
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""
    if 'rest_timer_active' not in st.session_state:
        st.session_state.rest_timer_active = False
    if 'rest_time_left' not in st.session_state:
        st.session_state.rest_time_left = 0
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()
    
    # Sidebar for workout setup
    with st.sidebar:
        st.header("Workout Setup")
        
        # Muscle group selection
        group_options = {k: v[0] for k, v in muscle_groups.items()}
        group_choice = st.selectbox(
            "Choose Muscle Group:",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x]
        )
        
        # Exercise selection
        group_name, exercises = muscle_groups[group_choice]
        exercise_options = {k: v[0] if v != "MISC" else "Misc Generic Exercise (count reps)" for k, v in exercises.items()}
        exercise_choice = st.selectbox(
            "Choose Exercise:",
            options=list(exercise_options.keys()),
            format_func=lambda x: exercise_options[x]
        )
        
        # Workout parameters
        target_reps = st.number_input("Target reps per set:", min_value=1, value=10)
        target_sets = st.number_input("Target sets:", min_value=1, value=3)
        rest_between_sets = st.number_input("Rest between sets (seconds):", min_value=5, value=30)
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Workout") and not st.session_state.workout_started:
                st.session_state.workout_started = True
                st.session_state.stage = None
                st.session_state.counter = 0
                st.session_state.sets_completed = 0
                st.session_state.total_reps_done = 0
                st.session_state.misc_state = {"prev_wrist_y": None, "cooldown": 0}
                st.session_state.prev_shoulder_y = None
                st.session_state.feedback = ""
                st.session_state.rest_timer_active = False
                st.experimental_rerun()
        
        with col2:
            if st.button("Stop Workout") and st.session_state.workout_started:
                st.session_state.workout_started = False
                st.experimental_rerun()
        
        # Display workout history
        st.header("Workout History")
        try:
            history_df = pd.read_csv("workout_log.csv")
            st.dataframe(history_df.tail(5))
        except FileNotFoundError:
            st.info("No workout history yet. Complete a workout to see your history here.")
    
    # Main content area
    if st.session_state.workout_started:
        ex_entry = exercises[exercise_choice]
        
        if ex_entry == "MISC":
            exercise_name = "Misc Generic Exercise (user-defined)"
            detector = None
        else:
            exercise_name = ex_entry[0]
            detector = ex_entry[1]
        
        # Check if it's a pull-up (special case)
        is_pullup = (exercise_name == "Pull-up")
        
        # Display workout info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sets Completed", f"{st.session_state.sets_completed}/{target_sets}")
        with col2:
            st.metric("Current Reps", f"{st.session_state.counter}/{target_reps}")
        with col3:
            st.metric("Total Reps", st.session_state.total_reps_done)
        
        # Display feedback
        if st.session_state.feedback:
            st.info(st.session_state.feedback)
        
        # Rest timer
        if st.session_state.rest_timer_active:
            st.warning(f"Resting: {st.session_state.rest_time_left}s remaining")
            if time.time() - st.session_state.last_update_time >= 1:
                st.session_state.rest_time_left -= 1
                st.session_state.last_update_time = time.time()
                if st.session_state.rest_time_left <= 0:
                    st.session_state.rest_timer_active = False
                    st.session_state.counter = 0
                    st.session_state.stage = None
                    st.session_state.misc_state = {"prev_wrist_y": None, "cooldown": 0}
                st.experimental_rerun()
        
        # Video processing
        st.header("Camera Feed")
        
        # Create a placeholder for the video
        video_placeholder = st.empty()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Process video frames
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Process with MediaPipe
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Process landmarks if detected
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Only process if not in rest timer
                        if not st.session_state.rest_timer_active:
                            counted_this_frame = False
                            
                            try:
                                if ex_entry == "MISC":
                                    st.session_state.misc_state, done, feedback = detector_misc(landmarks, st.session_state.misc_state)
                                    if done:
                                        st.session_state.counter += 1
                                        st.session_state.total_reps_done += 1
                                        counted_this_frame = True
                                        st.session_state.feedback = feedback
                                else:
                                    if is_pullup:
                                        # pull-up special
                                        st.session_state.stage, st.session_state.counter, _, feedback, done, st.session_state.prev_shoulder_y = detector_pullup(
                                            landmarks, st.session_state.stage, st.session_state.counter, st.session_state.prev_shoulder_y
                                        )
                                        if done:
                                            st.session_state.total_reps_done += 1
                                            counted_this_frame = True
                                            st.session_state.feedback = feedback
                                    else:
                                        # detectors expect (landmarks, stage, counter)
                                        res = detector(landmarks, st.session_state.stage, st.session_state.counter)
                                        # res may be 3-tuple or 5-tuple depending on detector signature
                                        if isinstance(res, tuple) and len(res) >= 5:
                                            # (stage, counter, sets_inc, feedback, done)
                                            st.session_state.stage, st.session_state.counter, _, feedback, done = res[:5]
                                        else:
                                            st.session_state.stage, st.session_state.counter, _, feedback, done = res[0], res[1], 0, res[3] if len(res) > 3 else "", res[4] if len(res) > 4 else False
                                        if done:
                                            st.session_state.total_reps_done += 1
                                            counted_this_frame = True
                                            st.session_state.feedback = feedback
                                
                                # Check if set is completed
                                if st.session_state.counter >= target_reps:
                                    st.session_state.sets_completed += 1
                                    
                                    # Check if all sets are completed
                                    if st.session_state.sets_completed >= target_sets:
                                        # Workout completed
                                        st.session_state.workout_started = False
                                        
                                        # Calculate calories
                                        cals = round(st.session_state.total_reps_done * calories_per_rep[exercise_name], 2)
                                        
                                        # Log session
                                        session_summary = {
                                            "group": group_name,
                                            "exercise": exercise_name,
                                            "sets_completed": st.session_state.sets_completed,
                                            "target_sets": target_sets,
                                            "total_reps": st.session_state.total_reps_done,
                                            "calories": cals
                                        }
                                        log_session(session_summary)
                                        
                                        # Show summary
                                        st.success("Workout Completed! 🎉")
                                        st.subheader("Session Summary")
                                        st.write(f"**Group:** {group_name}")
                                        st.write(f"**Exercise:** {exercise_name}")
                                        st.write(f"**Sets done:** {st.session_state.sets_completed}/{target_sets}")
                                        st.write(f"**Total reps:** {st.session_state.total_reps_done}")
                                        st.write(f"**Calories burned:** ~{cals} kcal")
                                        
                                        # Reset workout
                                        st.session_state.workout_started = False
                                    else:
                                        # Start rest timer
                                        st.session_state.rest_timer_active = True
                                        st.session_state.rest_time_left = rest_between_sets
                                        st.session_state.last_update_time = time.time()
                                        st.session_state.feedback = f"Set {st.session_state.sets_completed}/{target_sets} completed! Resting for {rest_between_sets}s."
                            
                            except Exception as e:
                                # Error in processing
                                st.error(f"Error processing pose: {e}")
                        
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image, 
                            results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )
                
                # Add workout info to the frame
                cv2.rectangle(image, (0, 0), (300, 100), (36, 120, 200), -1)
                cv2.putText(image, f"{group_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"{exercise_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Sets: {st.session_state.sets_completed}/{target_sets}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Reps: {st.session_state.counter}/{target_reps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                video_placeholder.image(image, channels="BGR", use_column_width=True)
            
            cap.release()
        else:
            st.error("Cannot access webcam. Please check your camera settings.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Workout Tracker!
        
        This application uses your webcam and pose estimation to track your workouts.
        
        **How to use:**
        1. Select a muscle group and exercise from the sidebar
        2. Set your target reps and sets
        3. Click "Start Workout"
        4. Position yourself in front of the camera
        5. Perform your exercises with proper form
        
        The app will count your reps and sets automatically!
        """)
        
        # Display example image or video
        st.image("https://via.placeholder.com/640x360.png?text=Workout+Demo", use_column_width=True)

if __name__ == "__main__":
    main()