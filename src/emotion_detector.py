"""
Facial Emotion Detector using DeepFace.

This module classifies the dominant facial emotion from an OpenCV frame.
While YOLO pose (detector.py) handles body keypoints and joint-angle retargeting,
DeepFace here handles facial emotion classification — the "social expression" layer.

Together, body motion retargeting + emotion-based expression mapping lets a
humanoid robot copy both *movement* and *social expression*.

Design notes:
  - DeepFace is slow (~100-300 ms per call), so we only run it every N frames.
  - A rolling history window smooths the label so it doesn't flicker.
  - All errors default to "neutral" so the pipeline never crashes.

Future ROS integration:
  The detected emotion string and confidence scores could be published to a
  ROS 2 topic (e.g. /emotion) as a custom message or std_msgs/String.
"""

from collections import Counter, deque


class EmotionDetector:
    """
    Classify the dominant facial emotion from an OpenCV BGR frame.

    Parameters
    ----------
    every_n_frames : int
        Run DeepFace only once every N frames to save CPU time.
    smoothing_window : int
        Number of recent detections to keep; the most-common label wins.
    backend : str
        "deepface" to use DeepFace, "none" to mock detection and always return neutral.
    """

    # All seven emotions DeepFace can return
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, every_n_frames: int = 10, smoothing_window: int = 5, backend: str = "deepface"):
        self.every_n = max(1, every_n_frames)
        self.window = max(1, smoothing_window)
        self.backend = backend.lower()

        # Flag to permanently disable if TensorFlow/DeepFace fails to load
        self._disabled = False

        # Rolling history of raw detections for smoothing
        self._history: deque[str] = deque(maxlen=self.window)

        # Cache the last successful result so we can reuse it on skip frames
        self._last_emotion: str = "neutral"
        self._last_scores: dict[str, float] = {e: 0.0 for e in self.EMOTIONS}
        self._last_scores["neutral"] = 1.0

        # Internal frame counter (used when caller doesn't pass frame_idx)
        self._internal_idx: int = 0

        # Lazy-import flag for DeepFace (avoids import cost until needed)
        self._deepface = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame, frame_idx: int | None = None):
        """
        Detect the dominant emotion in *frame*.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image from OpenCV.
        frame_idx : int, optional
            External frame counter.  If None, an internal counter is used.

        Returns
        -------
        emotion : str
            Smoothed emotion label (e.g. "happy", "neutral").
        scores : dict[str, float]
            Per-emotion confidence scores from the most recent real detection.
        """
        # Determine which frame number to use for the every-N check
        if frame_idx is None:
            idx = self._internal_idx
            self._internal_idx += 1
        else:
            idx = frame_idx

        # Only call DeepFace every N frames (it is expensive)
        if idx % self.every_n == 0:
            emotion, scores = self._run_deepface(frame)
            self._last_emotion = emotion
            self._last_scores = scores
            self._history.append(emotion)

        # Return the smoothed (most-common) emotion over the history window
        smoothed = self._smoothed_emotion()
        return smoothed, self._last_scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_deepface(self, frame):
        """Run DeepFace.analyze and return (emotion, scores). Never raises."""
        if self.backend == "none" or self._disabled:
            return "neutral", self._neutral_scores()

        # Lazy import — avoids slow TensorFlow startup until actually needed
        if self._deepface is None:
            try:
                # Note: On some macOS setups, TensorFlow is compiled with AVX instructions.
                # If the Mac CPU (or Rosetta) does not support AVX, this import may cause a
                # fatal C-level SIGABRT which bypasses Python's try/except.
                # This is a macOS TensorFlow compatibility issue, not a YOLO/OpenCV issue.
                # Use --emotion-backend none to bypass this.
                from deepface import DeepFace
                self._deepface = DeepFace
            except Exception as e:
                print(f"[EmotionDetector] WARNING: DeepFace/TensorFlow failed to load. "
                      f"Emotion detection disabled. ({e})")
                self._disabled = True
                return "neutral", self._neutral_scores()

        try:
            # enforce_detection=False avoids crash when no face is found
            results = self._deepface.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            # DeepFace returns a list of dicts (one per face); take the first
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
            elif isinstance(results, dict):
                result = results
            else:
                return "neutral", self._neutral_scores()

            emotion = result.get("dominant_emotion", "neutral")
            scores = result.get("emotion", self._neutral_scores())
            return str(emotion), {str(k): float(v) for k, v in scores.items()}

        except Exception as e:
            # Any DeepFace / TensorFlow error → fall back safely
            if not self._disabled:
                print(f"[EmotionDetector] DeepFace analyze failed: {e}. "
                      "Emotion detection disabled.")
                self._disabled = True
            return self._last_emotion, self._last_scores

    def _smoothed_emotion(self) -> str:
        """Return the most-common emotion in the rolling history window."""
        if not self._history:
            return "neutral"
        counter = Counter(self._history)
        return counter.most_common(1)[0][0]

    @staticmethod
    def _neutral_scores() -> dict[str, float]:
        scores = {e: 0.0 for e in EmotionDetector.EMOTIONS}
        scores["neutral"] = 100.0
        return scores
