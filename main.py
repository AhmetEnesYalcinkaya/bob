import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import math
import mediapipe as mp

# Constants
SERVO_OPTIONS = ["Rotate", "Nod", "Tilt", "Eye L-R", "Eye U-D", "RT", "RS", "RE", "LT", "LS", "LE"]
PROFILES = ["Larry", "Harry", "Barry", "Sindy"]
DATA_DIR = "data"
MODEL = YOLO("yolov8n-pose.pt")

# Streamlit page configuration
st.set_page_config(layout="wide")


# Helper Functions
def create_profile_dataframe():
    df = pd.DataFrame({
        'Index': range(1, 12),
        'Is Alive': [False] * 11,
        'Servo Name': SERVO_OPTIONS,
        'Min Value': [0] * 11,
        'Default Value': [0] * 11,
        'Max Value': [0] * 11
    })
    df.set_index('Index', inplace=True)
    return df


def load_profile_dataframes():
    profile_dataframes = {}
    for profile in PROFILES:
        file_name = os.path.join(DATA_DIR, f"{profile}_data.csv")
        if os.path.exists(file_name):
            df = pd.read_csv(file_name, index_col=0)
            if 'Unnamed: 1' in df.columns:
                df.rename(columns={'Unnamed: 1': 'Is Alive'}, inplace=True)
        else:
            df = create_profile_dataframe()
        profile_dataframes[profile] = df
    return profile_dataframes


def save_profile_dataframes(profile_dataframes):
    for profile, df in profile_dataframes.items():
        file_name = os.path.join(DATA_DIR, f"{profile}_data.csv")
        df.to_csv(file_name)


def clear_temp_files(directory, extension=".mp4"):
    try:
        for temp_file in os.listdir(directory):
            file_path = os.path.join(directory, temp_file)
            if os.path.isfile(file_path) and file_path.endswith(extension):
                os.remove(file_path)
    except Exception as e:
        st.error(f"Error deleting old temporary files: {e}")


# Geometry Functions
def calculate_angle(point1, point2, point3):
    a, b, c = np.array(point1), np.array(point2), np.array(point3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def calc_servo_value(value, min_servo_value, max_servo_value, golden_ratio=1):
    scaled_value = min(180, max(0, value * golden_ratio))
    servo_value = (scaled_value * (max_servo_value - min_servo_value) / 180) + min_servo_value
    return int(servo_value)


def closest_point_on_line(p1, d1, d2):
    p1, d1, d2 = map(lambda x: np.array(x, dtype=np.float32), [p1, d1, d2])
    line_vec = d2 - d1
    p1_to_d1 = p1 - d1
    t = np.dot(p1_to_d1, line_vec) / np.dot(line_vec, line_vec)
    t = max(0, min(1, t))
    closest_point = d1 + t * line_vec
    distance = cv2.norm(p1 - closest_point)
    return distance, closest_point


def calculate_distance(point1, point2):
    return int(math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2))))


# File Operations
def save_txt(motor_id, start_time, stop_time, servo_start, servo_end):
    with open("motion_results.txt", "a") as file:
        file.write(f"0 {motor_id} {start_time} {stop_time} {servo_start} {servo_end}\n")


def update_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_content = [f"{len(lines)}\n"] + lines
    with open(file_path, 'w') as file:
        file.writelines(new_content)


# Face Landmark Processing
def process_face_landmarks(cropped_image, face_mesh, RIGHT_EYE, LEFT_EYE, LEFT_IRIS, RIGHT_IRIS):
    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmark_points = {
                'right_eye': RIGHT_EYE,
                'left_eye': LEFT_EYE,
                'left_iris': LEFT_IRIS,
                'right_iris': RIGHT_IRIS
            }

            coords = {}
            for name, points in landmark_points.items():
                coords[name] = [(int(face_landmarks.landmark[idx].x * cropped_image.shape[1]),
                                 int(face_landmarks.landmark[idx].y * cropped_image.shape[0])) for idx in points]

                for point in coords[name]:
                    cv2.circle(cropped_image, point, 2, (0, 255, 0), -1, cv2.LINE_AA)

            return coords['right_eye'], coords['left_eye'], coords['left_iris'], coords['right_iris']
    else:
        st.error("No face landmarks detected in the cropped image.")
        return None, None, None, None


def get_point_face_landmark(cropped_image, face_mesh, point_id):
    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            point_coord = [(int(face_landmarks.landmark[point_id].x * cropped_image.shape[1]),
                            int(face_landmarks.landmark[point_id].y * cropped_image.shape[0]))]

            for point in point_coord:
                cv2.circle(cropped_image, point, 2, (0, 255, 0), -1, cv2.LINE_AA)

            return point_coord
    else:
        st.error("No face landmarks detected in the cropped image.")
        return None


# Servo Processing
class ServoProcessor:
    def __init__(self, servo_id, tolerance, golden_ratio=3, min_servo_value=320, max_servo_value=920):
        self.servo_id = servo_id
        self.tolerance = tolerance
        self.golden_ratio = golden_ratio
        self.min_servo_value = min_servo_value
        self.max_servo_value = max_servo_value
        self.servo_values = []
        self.counting = False
        self.servo_start = None
        self.start_frame_id = None

    def process_servo(self, frame_count, angle=None, dist1=None, dist2=None):
        if angle is None:
            servo_value = calc_servo_value(dist1 / dist2, self.min_servo_value, self.max_servo_value, self.golden_ratio)
        else:
            servo_value = calc_servo_value(angle, self.min_servo_value, self.max_servo_value, self.golden_ratio)

        self.servo_values.append(servo_value)

        if frame_count >= 4:
            previous_servo_value = self.servo_values[frame_count - 4]
            difference = abs(servo_value - previous_servo_value)
            if difference > self.tolerance:
                if not self.counting:
                    self.servo_start = servo_value
                    self.start_frame_id = frame_count
                    self.counting = True
            else:
                if self.counting:
                    servo_end = servo_value
                    end_frame_id = frame_count
                    save_txt(self.servo_id, self.start_frame_id, end_frame_id, self.servo_start, servo_end)
                    self.counting = False

    def finalize(self, frame_count):
        if self.counting:
            servo_end = self.servo_values[-1]
            end_frame_id = frame_count
            save_txt(self.servo_id, self.start_frame_id, end_frame_id, self.servo_start, servo_end)
            self.counting = False


# Video Processing
def process_video(video_path, output_path, selected_profile):
    cap = cv2.VideoCapture(video_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    progress_bar = st.progress(0)

    RIGHT_EYE = [33, 133, 159, 145]
    LEFT_EYE = [362, 263, 386, 374]
    LEFT_IRIS = [473]
    RIGHT_IRIS = [468]

    results = MODEL(source=video_path, stream=True)

    df = pd.read_csv(f"data/{selected_profile}_data.csv")
    servo_processors = {i: ServoProcessor(i, 20, 1, df["Min Value"][i - 1], df["Max Value"][i - 1]) for i in
                        range(1, 12) if df["Is Alive"][i - 1]}

    for frame_count, r in enumerate(results, 1):
        progress_bar.progress(int((frame_count / total_frames) * 100))

        ret, frame = cap.read()
        if not ret:
            break

        keypoints = r.keypoints
        if keypoints is not None:
            xy_coordinates = keypoints.xy.cpu().numpy()[0]

            # Calculate angles and distances
            left_wrist_elbow_shoulder_angle = 180 - calculate_angle(xy_coordinates[9], xy_coordinates[7],
                                                                    xy_coordinates[5])
            left_elbow_shoulder_hip_angle = calculate_angle(xy_coordinates[7], xy_coordinates[5], xy_coordinates[11])
            right_wrist_elbow_shoulder_angle = calculate_angle(xy_coordinates[10], xy_coordinates[8], xy_coordinates[6])
            right_elbow_shoulder_hip_angle = calculate_angle(xy_coordinates[8], xy_coordinates[6], xy_coordinates[12])
            dist_shoulders = calculate_distance(xy_coordinates[5], xy_coordinates[6])

            # Process face landmarks
            cropped_image = frame[int(xy_coordinates[6][1] - dist_shoulders):int(xy_coordinates[6][1]),
                            int(xy_coordinates[6][0]):int(xy_coordinates[6][0] + dist_shoulders)]
            right_eye_coords, left_eye_coords, left_iris_coords, right_iris_coords = process_face_landmarks(
                cropped_image, face_mesh, RIGHT_EYE, LEFT_EYE, LEFT_IRIS, RIGHT_IRIS)
            frame[int(xy_coordinates[6][1] - dist_shoulders):int(xy_coordinates[6][1]),
            int(xy_coordinates[6][0]):int(xy_coordinates[6][0] + dist_shoulders)] = cropped_image

            # Calculate various angles and ratios
            right_mid_iris = get_point_face_landmark(cropped_image, face_mesh, 468)
            r_right_eye_p, l_right_eye_p = right_eye_coords[0], right_eye_coords[1]
            u_right_eye_p = get_point_face_landmark(cropped_image, face_mesh, 65)
            d_right_eye_p = get_point_face_landmark(cropped_image, face_mesh, 230)

            eye_iris_r_dist = calculate_distance(right_mid_iris[0], r_right_eye_p)
            eye_l_r_dist = calculate_distance(l_right_eye_p, r_right_eye_p)
            eye_iris_d_dist = calculate_distance(right_mid_iris[0], d_right_eye_p[0])
            eye_d_u_dist = calculate_distance(u_right_eye_p[0], d_right_eye_p[0])

            eye_l_r_rate = ((eye_iris_r_dist / eye_l_r_dist) - 0.3) * 410
            eye_u_d_rate = ((eye_iris_d_dist / eye_d_u_dist) - 0.34) * 520

            # Calculate arm and head angles
            l_x_max = (xy_coordinates[5][0] + dist_shoulders) - xy_coordinates[11][0]
            dist_x_7_11 = abs(xy_coordinates[7][0] - xy_coordinates[11][0])
            left_wrist_cor = (dist_x_7_11 / l_x_max) * 180
            left_wrist_cor = (left_wrist_cor - 30) * 2

            r_x_max = xy_coordinates[12][0] - (xy_coordinates[6][0] - dist_shoulders)
            dist_x_8_12 = abs(xy_coordinates[12][0] - xy_coordinates[8][0])
            right_wrist_cor = 180 - (dist_x_8_12 / r_x_max) * 180

            ear3_ear4_dist = calculate_distance(xy_coordinates[4], xy_coordinates[3])
            nose_ear4_dist = calculate_distance(xy_coordinates[4], xy_coordinates[0])
            head_rotate = (nose_ear4_dist / ear3_ear4_dist) * 180

            head_angle_y = calculate_angle(xy_coordinates[1], xy_coordinates[2], [xy_coordinates[2][0], 0])
            head_tilt_angle = (head_angle_y - 75) * 4.4

            nose_shoulders_dist, _ = closest_point_on_line(xy_coordinates[0], xy_coordinates[6], xy_coordinates[5])
            head_nod_rate2angle = (0.92 - (nose_shoulders_dist / dist_shoulders)) / (0.92 - 0.42) * 180

            right_elbow_shoulder_hip_angle = 180 - ((right_elbow_shoulder_hip_angle - 15) / 75) * 180

            # Process servo movements
            servo_data = {
                1: head_rotate,
                2: 180 - head_nod_rate2angle,
                3: head_tilt_angle,
                4: eye_l_r_rate,
                5: eye_u_d_rate,
                6: right_wrist_cor,
                7: right_elbow_shoulder_hip_angle,
                8: right_wrist_elbow_shoulder_angle,
                9: left_wrist_cor,
                10: left_elbow_shoulder_hip_angle,
                11: left_wrist_elbow_shoulder_angle
            }

            for servo_id, angle in servo_data.items():
                if servo_id in servo_processors:
                    servo_processors[servo_id].process_servo(frame_count, angle)

        out.write(frame)

    # Finalize all servo processors
    for processor in servo_processors.values():
        processor.finalize(frame_count)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path


def download_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    st.download_button(
        label="Download motion_results.txt",
        data=data,
        file_name="motion_results.txt",
        mime='text/plain'
    )


# Main Streamlit App
def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if 'profile_dataframes' not in st.session_state:
        st.session_state.profile_dataframes = load_profile_dataframes()

    left, center, right = st.columns([1, 3, 1])

    with center:
        if 'customers' not in st.session_state:
            st.session_state.customers = []

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            selected_profile = st.selectbox("Select the Profile", PROFILES)
            st.write(f"**You selected: 🎃 {selected_profile}**")

        df = st.session_state.profile_dataframes[selected_profile]

        with st.form(key='data_form'):
            edited_df = st.data_editor(
                df.copy(),
                use_container_width=True,
                height=418,
                disabled=['Index', 'Servo Name']
            )

            submit_button = st.form_submit_button(label='Save Changes', type="primary")

        if submit_button:
            st.session_state.profile_dataframes[selected_profile] = edited_df
            save_profile_dataframes(st.session_state.profile_dataframes)
            st.success("Changes saved successfully!")

        uploaded_file = st.file_uploader("**Upload a video file**", type=["mp4", "avi", "mov"])

        if uploaded_file is not None and 'processed_video_path' not in st.session_state:
            clear_temp_files(DATA_DIR)

            with tempfile.NamedTemporaryFile(delete=False, dir=DATA_DIR, suffix='.mp4') as tffile:
                temp_video_path = tffile.name
                tffile.write(uploaded_file.read())

            output_video_path = os.path.join(DATA_DIR, "output_video_pose.mp4")

            try:
                status_message = st.info("Your video is being processed...")
                processed_video_path = process_video(temp_video_path, output_video_path, selected_profile)
                st.session_state.processed_video_path = processed_video_path
                status_message.success("All processes completed successfully!")

            except Exception as e:
                st.error(f"Error during pose detection: {e}")

        if 'processed_video_path' in st.session_state:
            update_txt_file("motion_results.txt")
            download_file("motion_results.txt")


if __name__ == "__main__":
    main()