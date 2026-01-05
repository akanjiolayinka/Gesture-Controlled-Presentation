"""
Test camera and basic hand detection using MediaPipe Tasks API
"""

import cv2
import mediapipe as mp
import time
from pathlib import Path
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame"""
    h, w = frame.shape[:2]
    pts = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
    
    # Draw connections
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    
    # Draw landmarks
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

def test_camera():
    print("Testing available cameras...")
    
    # Try different camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {i}: {width}x{height} @ {fps:.1f} FPS")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    print("\nTesting hand detection with camera 0...")
    
    # Setup MediaPipe HandLandmarker
    model_path = Path(__file__).parent / "models" / "hand_landmarker.task"
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please run the main application first to download the model.")
        return
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Show your hand to the camera. Press 'q' to quit.")
        
        start_time = time.time()
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Flip for mirror effect
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Draw results
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    draw_hand_landmarks(image, hand_landmarks)
                cv2.putText(image, "Hand Detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "No Hand Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Camera Test', image)
            frame_count += 1
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")

if __name__ == "__main__":
    test_camera()
