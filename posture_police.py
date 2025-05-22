import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import math as m
import time
from win10toast import ToastNotifier


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Helper to get 3D point for a landmark
def get_3d_point(landmark, color_image, aligned_depth_frame, depth_intrin):
    x_px = int(landmark.x * color_image.shape[1])
    y_px = int(landmark.y * color_image.shape[0])
    if 0 <= x_px < color_image.shape[1] and 0 <= y_px < color_image.shape[0]:
        depth = aligned_depth_frame.get_distance(x_px, y_px)
        if depth > 0:
            return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [x_px, y_px], depth))
    return None

def world_to_pixel(point3d, depth_intrin):
    x, y, z = point3d
    if z == 0:
        return None
    pixel = rs.rs2_project_point_to_pixel(depth_intrin, [x, y, z])
    return int(pixel[0]), int(pixel[1])

# ========== MediaPipe Setup ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# ========== RealSense Setup ==========
pipeline = rs.pipeline()
config = rs.config()

# Device resolution and stream settings
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("This demo requires a Depth camera with an RGB sensor.")
    exit(0)

# Enable both streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

toaster = ToastNotifier()

bad_state_start_time = None
notification_sent = False
bad_states = {"alert"}
warning_duration = 10  # seconds

try:

    debounce_state = "good"
    debounce_counter = 0
    debounce_required_frames = 12  # ~0.4s if 30 FPS
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Run pose estimation
        rgb_for_mediapipe = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_for_mediapipe)

        # Get intrinsics to compute real-world 3D positions
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        if results.pose_landmarks:
            
            # RIGHT side landmarks (use LEFT if camera is on that side)
            lm = results.pose_landmarks.landmark
            shoulder = get_3d_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], color_image, aligned_depth_frame, depth_intrin)
            hip      = get_3d_point(lm[mp_pose.PoseLandmark.RIGHT_HIP], color_image, aligned_depth_frame, depth_intrin)
            ear      = get_3d_point(lm[mp_pose.PoseLandmark.RIGHT_EAR], color_image, aligned_depth_frame, depth_intrin)
            nose     = get_3d_point(lm[mp_pose.PoseLandmark.NOSE], color_image, aligned_depth_frame, depth_intrin)

            h, w = color_image.shape[:2]
            l_shldr_x = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)

            if shoulder is not None and hip is not None and ear is not None and nose is not None:
                # Vector from hip to shoulder
                back_vec = shoulder - hip
                vertical = np.array([0, -1, 0])  # Assuming Y axis is up/down

                # Angle(degrees) between back_vec and vertical
                cos_angle = np.dot(back_vec, vertical) / (np.linalg.norm(back_vec) * np.linalg.norm(vertical))
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                # Perpendicular distance from head to torso axis (3D)
                torso_vec = shoulder - hip
                torso_vec_norm = torso_vec / np.linalg.norm(torso_vec)
                head_vec = ear - shoulder
                projection_length = np.dot(head_vec, torso_vec_norm)
                projection_vec = projection_length * torso_vec_norm
                perpendicular_vec = head_vec - projection_vec
                perpendicular_dist_cm = abs(ear[0] - shoulder[0]) * 100  # X is the horizontal axis in meters


                # Calculate endpoints for drawing (2D)
                shoulder_px = world_to_pixel(shoulder, depth_intrin)
                hip_px = world_to_pixel(hip, depth_intrin)
                ear_px = world_to_pixel(ear, depth_intrin)
                projection_end_3d = shoulder + projection_vec
                projection_end_px = world_to_pixel(projection_end_3d, depth_intrin)
                

                # Draw torso vector (hip → shoulder)
                if hip_px and shoulder_px:
                    cv2.arrowedLine(color_image, hip_px, shoulder_px, (0, 255, 255), 2, tipLength=0.2)  # Yellow

                # Draw projection vector (shoulder → shoulder + projection_vec)
                if shoulder_px and projection_end_px:
                    cv2.arrowedLine(color_image, shoulder_px, projection_end_px, (255, 0, 255), 2, tipLength=0.2)  # Magenta
                # Draw perpendicular vector (shoulder + projection_vec → ear)
                if projection_end_px and ear_px:
                    cv2.arrowedLine(color_image, projection_end_px, ear_px, (0, 0, 255), 2, tipLength=0.2)  # Red
                # (Optional) Draw head vector (shoulder → ear)
                if shoulder_px and ear_px:
                    cv2.arrowedLine(color_image, shoulder_px, ear_px, (0, 255, 0), 1, tipLength=0.2)  # Green


                bad_head = False
                if perpendicular_dist_cm is not None:
                    bad_head = perpendicular_dist_cm > 8  

                # For camera on right side; use < 0 if on left side
                is_forward = back_vec[0] > 0

                # Threshold: only positive angle (leaning towards computer)
                bad_back = (angle_deg > 17) and is_forward


                 # Calculate distance between left shoulder and right shoulder points.
                offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

                feedback = []
                if bad_back:
                    feedback.append(f"Slouched back ({angle_deg:.1f}deg)")
                if bad_head:
                    feedback.append(f"Head forward ({perpendicular_dist_cm:.1f}cm)")

                if offset >= 60:
                    current_state = "not_aligned"
                elif feedback:
                    current_state = "alert"
                else:
                    current_state = "good"

                # Debounce logic
                if current_state == debounce_state:
                    debounce_counter = 0
                else:
                    debounce_counter += 1
                    if debounce_counter >= debounce_required_frames:
                        debounce_state = current_state
                        debounce_counter = 0
                
                if debounce_state in bad_states:
                    if bad_state_start_time is None:
                        bad_state_start_time = time.time()
                        notification_sent = False  # reset if just entered bad state

                    elapsed = time.time() - bad_state_start_time
                    if elapsed >= warning_duration and not notification_sent:
                        # Send Windows notification
                        toaster.show_toast("Posture Warning", 
                                        "You have been in a bad posture for at least 10 seconds.",
                                        duration=5,
                                        threaded=True)
                        notification_sent = True
                else:
                    # Good posture, reset
                    bad_state_start_time = None
                    notification_sent = False


                if debounce_state == "not_aligned":
                    cv2.putText(color_image, "Not aligned with the camera", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif debounce_state == "alert":
                    if feedback:
                        cv2.putText(color_image, "Posture Alert: " + ", ".join(feedback), (30, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    else:
                        cv2.putText(color_image, "Posture Alert!", (30, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                else:
                    cv2.putText(color_image, "Good sitting posture!", (30, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                
            # Draw full skeleton
            mp_drawing.draw_landmarks(
                color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
        else:
            cv2.putText(color_image, "Pose not detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Show image
        cv2.imshow('Pose with RealSense Depth Fusion', color_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()