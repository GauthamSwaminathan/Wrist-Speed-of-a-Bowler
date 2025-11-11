import cv2
import mediapipe as mp
import numpy as np
import os  # Added import for path manipulation

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

K = np.array([
    [3.19330310e+03, 0.00000000e+00, 1.91030751e+03],
    [0.00000000e+00, 3.19833958e+03, 1.13256305e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

def pixel_to_3d(u, v, depth, intrinsic_matrix):
    """
    Convert 2D image pixel coordinates to 3D world coordinates using depth.
    
    Args:
        u (float): x-coordinate in the image (pixel).
        v (float): y-coordinate in the image (pixel).
        depth (float): Depth at the pixel (Z-coordinate in meters).
        intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix.
    
    Returns:
        np.ndarray: 3D coordinates [X, Y, Z].
    """
    pixel_coord = np.array([u, v, 1])
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)
    camera_coord = intrinsic_inv @ pixel_coord
    camera_coord_3d = camera_coord * depth
    return camera_coord_3d

def compute_speed(pixel_coordinates, depth, intrinsic_matrix, prev_3d_coord, frame_rate):
    """
    Compute the speed of a pixel coordinate in the 3D world from depth data.

    Args:
        pixel_coordinates (tuple): (u, v) pixel coordinates for the current frame.
        depth (float): Depth value at the pixel in meters.
        intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        prev_3d_coord (np.ndarray): 3D coordinates from the previous frame.
        frame_rate (float): Frame rate of the video in frames per second.

    Returns:
        tuple: (speed in m/s, current 3D coordinates).
    """
    u, v = pixel_coordinates
    curr_3d_coord = pixel_to_3d(u, v, depth, intrinsic_matrix)
    
    if prev_3d_coord is not None:
        delta_time = 1 / frame_rate
        distance = np.linalg.norm(curr_3d_coord - prev_3d_coord)
        speed = distance / delta_time
    else:
        speed = 0.0

    return speed, curr_3d_coord

# Path to the input video file
video_path = r'Pose Estimation\Data\IMG_0019.MOV'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Compute output path in the same directory as the input video
video_dir = os.path.dirname(video_path)
input_filename = os.path.basename(video_path)
base_name, _ = os.path.splitext(input_filename)
output_filename = base_name + '.mp4'
output_path = os.path.join(video_dir, output_filename)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_3d_coord = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                x_norm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
                y_norm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                    
                # Convert to pixel coordinates
                x_pixel = int(x_norm * frame_width)
                y_pixel = int(y_norm * frame_height)
                
                # Calculate 3D (example depth value added, adjust as needed)
                depth = 5.96  # Placeholder depth value in meters
                speed, curr_3d_coord = compute_speed((x_pixel, y_pixel), depth, K, prev_3d_coord, fps)
                prev_3d_coord = curr_3d_coord
                
                # Draw white rectangle behind the text
                text = f"Speed: {speed:.2f} m/s"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                box_coords = ((x_pixel + 10, y_pixel - 10 - text_height - 5), (x_pixel + 10 + text_width + 10, y_pixel - 10 + 5))
                cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

                # Overlay the text on the image
                cv2.putText(image, text, 
                            (x_pixel + 10, y_pixel - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                                
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No pose landmarks detected in this frame.")
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))                
        
        # Write the frame to the output video
        out.write(image)

        # Display frame (optional, can be removed for saving only)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
