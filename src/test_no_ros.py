"""
Standalone test: YOLO pose detection + joint mapping, no ROS required.
Optionally adds DeepFace emotion detection and robot expression mapping.

Architecture:
  - YOLO pose (detector.py)        → body keypoints → joint angles (mapper.py)
  - DeepFace (emotion_detector.py) → emotion label  → expression command
                                                       (expression_mapper.py)
  The future humanoid robot can combine body motion retargeting with
  emotion-based expression mapping so it doesn't just copy movement,
  but also copies social expression.

Run from project root:
    python src/test_no_ros.py                                    # default video
    python src/test_no_ros.py --camera                           # live camera
    python src/test_no_ros.py --video labs/videos/workout.mp4    # video file
    python src/test_no_ros.py --camera --emotion                 # + emotion
    python src/test_no_ros.py --camera --emotion --show-expression
    python src/test_no_ros.py --emotion --emotion-every 20

Press Q to quit.

Future ROS integration:
  expression_command could be published to ROS 2 topics:
    /emotion           — std_msgs/String with the emotion label
    /robot_expression  — std_msgs/String (JSON) or custom msg with full command
"""

import sys
import os
import argparse
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from detector import PoseDetector
from mapper import keypoints_to_joints
from visualizer import Visualizer

DEFAULT_VIDEO = None


def open_source(video_path: str | None):
    """Return an OpenCV VideoCapture for either a video file or the camera."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        print(f"Reading from video: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera 0")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Flush first frames — macOS cameras return black until warmed up
        for _ in range(30):
            cap.grab()
        print("Reading from camera 0")
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="Path to video file (omit for live camera)")
    parser.add_argument("--camera", action="store_true",
                        help="Force live camera even if --video default exists")

    # ---- Emotion / expression arguments ----
    parser.add_argument("--emotion", action="store_true",
                        help="Enable DeepFace facial emotion detection")
    parser.add_argument("--emotion-every", type=int, default=10,
                        help="Run DeepFace every N frames (default 10)")
    parser.add_argument("--emotion-window", type=int, default=5,
                        help="Smoothing window size (default 5)")
    parser.add_argument("--emotion-backend", type=str, default="deepface", choices=["deepface", "none"],
                        help="Backend for emotion detection. Use 'none' to bypass TensorFlow AVX aborts on macOS.")
    parser.add_argument("--show-expression", action="store_true",
                        help="Display robot expression summary on screen")
    args = parser.parse_args()

    video_path = None if args.camera else args.video

    cap = open_source(video_path)
    detector = PoseDetector()
    viz = Visualizer()

    # ---- Optionally set up emotion detection pipeline ----
    # EmotionDetector classifies facial emotion via DeepFace.
    # ExpressionMapper converts emotion labels into robot expression commands.
    emo_detector = None
    expr_mapper = None
    if args.emotion:
        from emotion_detector import EmotionDetector
        from expression_mapper import ExpressionMapper
        emo_detector = EmotionDetector(
            every_n_frames=args.emotion_every,
            smoothing_window=args.emotion_window,
            backend=args.emotion_backend,
        )
        expr_mapper = ExpressionMapper()
        print(f"Emotion detection ON  (backend: {args.emotion_backend}, every {args.emotion_every} frames, "
              f"window {args.emotion_window})")

    print("Running — press Q in the window to quit.")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("End of video.")
                break
            frame_idx += 1

            # ---- YOLO pose detection (body keypoints → joint angles) ----
            keypoints = detector.detect(frame)

            joints = None
            if keypoints is not None:
                joints = keypoints_to_joints(keypoints)
                frame = detector.draw(frame, keypoints)

            # ---- DeepFace emotion detection (facial emotion → robot expression) ----
            emotion_label = None
            expression_cmd = None
            if emo_detector is not None:
                emotion_label, scores = emo_detector.detect(frame, frame_idx)
                expression_cmd = expr_mapper.map_emotion(emotion_label, scores)

                # Draw emotion label on the frame
                cv2.putText(
                    frame,
                    f"Emotion: {emotion_label}",
                    (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA,
                )

                # Optionally draw the robot expression summary
                if args.show_expression and expression_cmd:
                    summary = ExpressionMapper.summary_string(expression_cmd)
                    cv2.putText(
                        frame,
                        f"Robot: {summary}",
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2,
                        cv2.LINE_AA,
                    )

                # Future ROS integration:
                # Here you would publish expression_cmd to /robot_expression
                # and emotion_label to /emotion as ROS 2 messages.

            if not viz.show(frame, joints, keypoints):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viz.close()


if __name__ == "__main__":
    main()
