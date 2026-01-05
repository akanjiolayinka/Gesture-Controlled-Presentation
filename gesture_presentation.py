"""
Gesture-Controlled Presentation System
Controls PowerPoint/Keynote/Google Slides with hand gestures
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from enum import Enum
import threading
import queue
import os
import urllib.request
from pathlib import Path
from typing import Optional

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# Configure pyautogui
pyautogui.FAILSAFE = True  # Move mouse to corner to emergency stop
pyautogui.PAUSE = 0.05  # Small pause between keystrokes

# MediaPipe Tasks API (mediapipe>=0.10.29) no longer exposes mp.solutions.*


HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


def _ensure_hand_model(model_path: Path) -> Path:
    """Ensure the MediaPipe HandLandmarker model exists locally.

    The official model is distributed by Google via a public GCS bucket.
    """
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Official MediaPipe model hosting pattern
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )

    try:
        print(f"Downloading hand model to: {model_path}")
        urllib.request.urlretrieve(url, model_path)
        if not model_path.exists() or model_path.stat().st_size == 0:
            raise RuntimeError("Downloaded model file is empty")
        return model_path
    except Exception as e:
        raise RuntimeError(
            "Could not download the MediaPipe hand model automatically. "
            "If you are offline, download 'hand_landmarker.task' and place it at: "
            f"{model_path}"
        ) from e


class HandTracker:
    """Hand landmark detector using MediaPipe Tasks HandLandmarker."""

    def __init__(
        self,
        num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        project_dir = Path(__file__).resolve().parent
        model_path = _ensure_hand_model(project_dir / "models" / "hand_landmarker.task")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Note: keep annotation simple for Pylance compatibility
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int):
        if self._landmarker is None:
            raise RuntimeError("HandTracker is closed")
        # Ensure monotonic timestamps for VIDEO mode
        timestamp_ms = max(int(timestamp_ms), int(self._last_ts_ms) + 1)
        self._last_ts_ms = timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    """Draw landmarks and connections onto a BGR frame."""
    h, w = frame.shape[:2]

    pts = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

class Gesture(Enum):
    NONE = 0
    SWIPE_RIGHT = 1
    SWIPE_LEFT = 2
    SWIPE_UP = 3
    SWIPE_DOWN = 4
    PINCH = 5
    FIST = 6
    POINT = 7
    PALM = 8
    TWO_FINGERS = 9

class GestureRecognizer:
    def __init__(self, sensitivity=1.0):
        self.sensitivity = sensitivity
        self.prev_hand_position = None
        self.gesture_buffer = []
        self.buffer_size = 5
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # seconds
        
    def get_landmark_coords(self, landmarks, image_shape):
        """Convert normalized landmarks to pixel coordinates"""
        h, w = image_shape[:2]
        coords = []
        for lm in landmarks:
            coords.append((int(lm.x * w), int(lm.y * h)))
        return coords
    
    def calculate_finger_states(self, landmarks):
        """Check which fingers are extended"""
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [2, 6, 10, 14, 18]  # PIP joints
        
        finger_states = []
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:  # Finger is extended
                finger_states.append(1)
            else:
                finger_states.append(0)
        
        # Special handling for thumb (different orientation)
        if landmarks[4].x < landmarks[2].x:  # Thumb is extended
            finger_states[0] = 1
        else:
            finger_states[0] = 0
            
        return finger_states
    
    def recognize_gesture(self, landmarks, image_shape, current_hand_position=None):
        """Main gesture recognition logic"""
        if not landmarks:
            return Gesture.NONE
        
        # Get finger states
        finger_states = self.calculate_finger_states(landmarks)
        
        # Define common gestures
        # Point gesture: only index finger extended
        if finger_states == [0, 1, 0, 0, 0]:
            return Gesture.POINT
        
        # Palm: all fingers extended
        elif finger_states == [1, 1, 1, 1, 1]:
            return Gesture.PALM
        
        # Fist: no fingers extended
        elif finger_states == [0, 0, 0, 0, 0]:
            return Gesture.FIST
        
        # Pinch: thumb and index close together
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        if distance < 0.05 * self.sensitivity and sum(finger_states[2:]) == 0:
            return Gesture.PINCH
        
        # Two fingers: index and middle extended
        elif finger_states == [0, 1, 1, 0, 0]:
            return Gesture.TWO_FINGERS
        
        # Detect swipes based on hand movement
        if current_hand_position and self.prev_hand_position:
            dx = current_hand_position[0] - self.prev_hand_position[0]
            dy = current_hand_position[1] - self.prev_hand_position[1]
            
            # Normalize by image size
            dx_norm = dx / image_shape[1]
            dy_norm = dy / image_shape[0]
            
            threshold = 0.1 * self.sensitivity
            
            if abs(dx_norm) > abs(dy_norm):  # Horizontal movement
                if dx_norm > threshold:
                    return Gesture.SWIPE_RIGHT
                elif dx_norm < -threshold:
                    return Gesture.SWIPE_LEFT
            else:  # Vertical movement
                if dy_norm > threshold:
                    return Gesture.SWIPE_DOWN
                elif dy_norm < -threshold:
                    return Gesture.SWIPE_UP
        
        # Update previous position
        if current_hand_position:
            self.prev_hand_position = current_hand_position
            
        return Gesture.NONE
    
    def smooth_gesture(self, gesture):
        """Apply smoothing to avoid false triggers"""
        self.gesture_buffer.append(gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
        
        # Return most common gesture in buffer
        if self.gesture_buffer:
            return max(set(self.gesture_buffer), key=self.gesture_buffer.count)
        return Gesture.NONE

class PresentationController:
    def __init__(self, app_mode="auto"):
        """
        app_mode: "auto", "powerpoint", "keynote", "google_slides", "browser"
        """
        self.app_mode = app_mode
        self.last_action_time = 0
        self.action_cooldown = 0.8  # Minimum seconds between actions
        self.gesture_history = []
        self.is_presenting = False
        self.verbose = True
        
        # Define key mappings for different apps
        self.key_mappings = {
            "powerpoint": {
                "next": ['right', 'pagedown', 'space', 'n'],
                "prev": ['left', 'pageup', 'p'],
                "start": 'f5',
                "stop": 'esc',
                "black": 'b',
                "white": 'w'
            },
            "keynote": {
                "next": ['right', 'space'],
                "prev": ['left'],
                "start": 'command+return',
                "stop": 'esc',
                "black": 'b',
                "white": 'w'
            },
            "browser": {
                "next": ['right', 'space'],
                "prev": ['left'],
                "start": 'f11',  # Full screen
                "stop": 'esc'
            },
            "auto": {  # Try common defaults
                "next": ['right', 'space', 'pagedown'],
                "prev": ['left', 'pageup'],
                "start": 'f5',
                "stop": 'esc'
            }
        }
        
    def can_perform_action(self):
        """Check if enough time has passed since last action"""
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            self.last_action_time = current_time
            return True
        return False
    
    def press_keys(self, keys):
        """Press one or multiple keys"""
        if isinstance(keys, str):
            pyautogui.press(keys)
        elif isinstance(keys, list):
            for key in keys:
                try:
                    pyautogui.press(key)
                    break  # Stop after first successful key press
                except:
                    continue
    
    def next_slide(self):
        if self.can_perform_action():
            keys = self.key_mappings[self.app_mode]["next"]
            self.press_keys(keys)
            if self.verbose:
                print("▶ Next slide")
            return True
        return False
    
    def previous_slide(self):
        if self.can_perform_action():
            keys = self.key_mappings[self.app_mode]["prev"]
            self.press_keys(keys)
            if self.verbose:
                print("◀ Previous slide")
            return True
        return False
    
    def start_presentation(self):
        if self.can_perform_action():
            key = self.key_mappings[self.app_mode]["start"]
            if isinstance(key, str):
                if '+' in key:
                    # Handle key combinations
                    keys = key.split('+')
                    pyautogui.hotkey(*keys)
                else:
                    pyautogui.press(key)
            self.is_presenting = True
            if self.verbose:
                print("🎬 Starting presentation")
            return True
        return False
    
    def stop_presentation(self):
        if self.can_perform_action():
            key = self.key_mappings[self.app_mode]["stop"]
            self.press_keys(key)
            self.is_presenting = False
            if self.verbose:
                print("⏹ Stopping presentation")
            return True
        return False
    
    def black_screen(self):
        if self.can_perform_action() and "black" in self.key_mappings[self.app_mode]:
            key = self.key_mappings[self.app_mode]["black"]
            self.press_keys(key)
            if self.verbose:
                print("⬛ Black screen")
            return True
        return False
    
    def white_screen(self):
        if self.can_perform_action() and "white" in self.key_mappings[self.app_mode]:
            key = self.key_mappings[self.app_mode]["white"]
            self.press_keys(key)
            if self.verbose:
                print("⬜ White screen")
            return True
        return False
    
    def handle_gesture(self, gesture):
        """Map gestures to actions"""
        self.gesture_history.append((gesture, time.time()))
        
        # Keep only recent history
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)
        
        if gesture == Gesture.SWIPE_RIGHT:
            return self.next_slide()
        elif gesture == Gesture.SWIPE_LEFT:
            return self.previous_slide()
        elif gesture == Gesture.SWIPE_DOWN:
            return self.start_presentation()
        elif gesture == Gesture.SWIPE_UP:
            return self.stop_presentation()
        elif gesture == Gesture.PINCH:
            return self.black_screen()
        elif gesture == Gesture.PALM:
            return self.white_screen()
        elif gesture == Gesture.POINT:
            # Could be used for laser pointer simulation
            pass
        elif gesture == Gesture.TWO_FINGERS:
            # Could be used for zoom
            pass
            
        return False

class GestureControlledPresentation:
    def __init__(self, camera_index=0, app_mode="auto", sensitivity=1.0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.show_debug = True
        self.show_landmarks = True
        self.gesture_display_duration = 1.0  # seconds
        
        # Initialize components
        self.gesture_recognizer = GestureRecognizer(sensitivity)
        self.presentation_controller = PresentationController(app_mode)
        
        # Gesture display
        self.last_detected_gesture = Gesture.NONE
        self.gesture_display_time = 0
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def initialize_camera(self, width=640, height=480):
        """Initialize webcam with specified settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized: {width}x{height}")
        return True
    
    def draw_gesture_info(self, frame, gesture, hand_landmarks=None):
        """Draw gesture information on frame"""
        h, w = frame.shape[:2]
        
        # Draw gesture name
        gesture_text = f"Gesture: {gesture.name.replace('_', ' ').title()}"
        text_color = (0, 255, 0) if gesture != Gesture.NONE else (255, 255, 255)
        
        cv2.putText(frame, gesture_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Draw presentation status
        status = "PRESENTING" if self.presentation_controller.is_presenting else "READY"
        status_color = (0, 255, 0) if self.presentation_controller.is_presenting else (0, 165, 255)
        cv2.putText(frame, f"Status: {status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw hand landmarks if enabled
        if self.show_landmarks and hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)
        
        # Draw control instructions
        instructions = [
            "Swipe Right: Next Slide",
            "Swipe Left: Prev Slide",
            "Swipe Down: Start Presentation",
            "Swipe Up: Stop Presentation",
            "Pinch: Black Screen",
            "Palm: White Screen",
            "Press 'q': Quit",
            "Press 's': Start/Stop Debug",
            "Press 'l': Toggle Landmarks",
            "Press 'r': Reset"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 150 + i * 20
            if y_pos < 50:  # Don't draw off-screen
                break
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("=" * 50)
        print("Gesture-Controlled Presentation System")
        print("=" * 50)
        print("\nInitializing...")
        
        # Initialize camera
        try:
            self.initialize_camera()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return
        
        # Initialize hand tracking
        hands = HandTracker(
            num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.running = True
        print("\n✅ System ready!")
        print("\nControls:")
        print("- Swipe Right: Next Slide")
        print("- Swipe Left: Previous Slide")
        print("- Swipe Down: Start Presentation")
        print("- Swipe Up: Stop Presentation")
        print("- Pinch: Black Screen")
        print("- Open Palm: White Screen")
        print("\nPress 'q' to quit")
        print("=" * 50)
        
        # Main loop
        start_time = time.time()
        cap = self.cap
        if cap is None:
            print("Camera not initialized")
            hands.close()
            return

        while self.running and cap.isOpened():
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # Read frame
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame for hand detection
            timestamp_ms = int((time.time() - start_time) * 1000)
            results = hands.detect(frame_rgb, timestamp_ms)
            
            current_gesture = Gesture.NONE
            current_hand_landmarks = None
            
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    current_hand_landmarks = hand_landmarks
                    
                    # Get hand position (using wrist)
                    wrist = hand_landmarks[0]
                    h, w = frame.shape[:2]
                    hand_position = (int(wrist.x * w), int(wrist.y * h))
                    
                    # Recognize gesture
                    raw_gesture = self.gesture_recognizer.recognize_gesture(
                        hand_landmarks, 
                        frame.shape, 
                        hand_position
                    )
                    
                    # Apply smoothing
                    smoothed_gesture = self.gesture_recognizer.smooth_gesture(raw_gesture)
                    
                    # Handle gesture if it's new
                    if smoothed_gesture != Gesture.NONE and smoothed_gesture != self.last_detected_gesture:
                        self.last_detected_gesture = smoothed_gesture
                        self.gesture_display_time = time.time()
                        
                        # Control presentation
                        self.presentation_controller.handle_gesture(smoothed_gesture)
                    
                    current_gesture = smoothed_gesture
            
            # Clear gesture if no hand detected for a while
            if not results.hand_landmarks:
                if time.time() - self.gesture_display_time > self.gesture_display_duration:
                    self.last_detected_gesture = Gesture.NONE
            
            # Draw debug information
            if self.show_debug:
                frame = self.draw_gesture_info(frame, current_gesture, current_hand_landmarks)
            
            # Show frame
            cv2.imshow('Gesture Presentation Controller', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                print("\nShutting down...")
            elif key == ord('s'):
                self.show_debug = not self.show_debug
                print(f"Debug display: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('r'):
                self.last_detected_gesture = Gesture.NONE
                self.gesture_recognizer.prev_hand_position = None
                print("Reset gesture tracking")
            elif key == ord('1'):
                # Manual control for testing
                self.presentation_controller.next_slide()
            elif key == ord('2'):
                self.presentation_controller.previous_slide()
        
        # Cleanup
        self.cleanup()
        hands.close()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def main():
    """Main entry point with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gesture-Controlled Presentation System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--app', type=str, default='auto', 
                       choices=['auto', 'powerpoint', 'keynote', 'google_slides', 'browser'],
                       help='Presentation application (default: auto)')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                       help='Gesture sensitivity (0.5 to 2.0, default: 1.0)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug display')
    
    args = parser.parse_args()
    
    # Create and run the application
    app = GestureControlledPresentation(
        camera_index=args.camera,
        app_mode=args.app,
        sensitivity=max(0.5, min(2.0, args.sensitivity))
    )
    
    if args.no_debug:
        app.show_debug = False
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()
