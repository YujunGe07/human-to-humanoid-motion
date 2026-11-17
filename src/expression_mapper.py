"""
Expression Mapper — convert human emotion labels into robot expression commands.

ExpressionMapper translates a classified emotion (from EmotionDetector / DeepFace)
into a structured command dictionary that future robot code can consume.

Architecture overview:
  1. YOLO pose (detector.py)  → body keypoints → joint angles (mapper.py)
     → robot body motion retargeting
  2. DeepFace (emotion_detector.py) → emotion label
     → ExpressionMapper (this file) → robot expression command
     → robot face / LED / gesture / posture control

The future humanoid robot can combine BOTH pipelines so it doesn't just
copy movement, but also copies social expression.

Future ROS integration:
  The expression_command dict returned by map_emotion() can be serialized as
  JSON and published to a ROS 2 topic (e.g. /robot_expression) using
  std_msgs/String or a custom ExpressionCommand.msg.
"""


class ExpressionMapper:
    """
    Map a human emotion label to a robot expression command dictionary.

    The returned dict contains fields that downstream hardware or simulation
    drivers can use to control head pose, face display, LED color, gesture,
    and motion style.
    """

    # ------------------------------------------------------------------
    # Emotion → robot-expression mapping table
    # Each entry defines what the robot should do for this emotion.
    # ------------------------------------------------------------------
    _EXPRESSION_TABLE: dict[str, dict] = {
        "happy": {
            "head_pitch": -0.1,       # head slightly up — looks upbeat
            "head_yaw": 0.0,
            "face": "smile",
            "eye_shape": "happy",
            "led_color": "green",
            "gesture": "open_arms",   # welcoming, energetic gesture
            "motion_style": "bouncy", # more energetic movement overall
        },
        "sad": {
            "head_pitch": 0.3,        # head tilted down
            "head_yaw": 0.0,
            "face": "sad",
            "eye_shape": "droopy",
            "led_color": "blue",
            "gesture": "arms_close",  # arms closer to body
            "motion_style": "slow",   # slower, subdued movement
        },
        "angry": {
            "head_pitch": 0.1,        # head stable or slightly down
            "head_yaw": 0.0,
            "face": "angry",
            "eye_shape": "narrow",    # furrowed / narrowed
            "led_color": "red",
            "gesture": "fists",       # tense fists
            "motion_style": "sharp",  # sharper, more aggressive movement
        },
        "surprise": {
            "head_pitch": -0.15,      # head slightly up
            "head_yaw": 0.0,
            "face": "surprise",
            "eye_shape": "wide",      # wide-open eyes
            "led_color": "yellow",
            "gesture": "hands_up",    # quick surprise reaction
            "motion_style": "quick",  # fast reactive movement
        },
        "fear": {
            "head_pitch": -0.05,      # head slightly back
            "head_yaw": 0.05,         # slight turn (looking away)
            "face": "fear",
            "eye_shape": "worried",
            "led_color": "purple",
            "gesture": "arms_guard",  # small defensive posture
            "motion_style": "cautious",
        },
        "disgust": {
            "head_pitch": 0.05,
            "head_yaw": 0.15,         # slight head turn away
            "face": "disgust",
            "eye_shape": "squint",
            "led_color": "orange",
            "gesture": "hand_wave",   # pushing-away gesture
            "motion_style": "recoil",
        },
        "neutral": {
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "face": "neutral",
            "eye_shape": "normal",
            "led_color": "white",
            "gesture": "none",
            "motion_style": "normal",
        },
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_emotion(self, emotion: str, scores: dict[str, float] | None = None) -> dict:
        """
        Convert an emotion label into a robot expression command.

        Parameters
        ----------
        emotion : str
            The classified emotion (e.g. "happy", "sad", "neutral").
        scores : dict, optional
            Per-emotion confidence scores from the detector. Used to set
            the *confidence* field in the output command.

        Returns
        -------
        expression_command : dict
            A dictionary that downstream robot drivers can consume.

        Example output::

            {
                "emotion": "happy",
                "confidence": 0.92,
                "head_pitch": -0.1,
                "head_yaw": 0.0,
                "face": "smile",
                "eye_shape": "happy",
                "led_color": "green",
                "gesture": "open_arms",
                "motion_style": "bouncy",
            }
        """
        # Normalize and fall back to neutral for unknown labels
        emotion_key = emotion.lower().strip()
        if emotion_key not in self._EXPRESSION_TABLE:
            emotion_key = "neutral"

        # Look up the robot expression fields for this emotion
        expr = dict(self._EXPRESSION_TABLE[emotion_key])  # shallow copy

        # Compute confidence from the scores dict if available
        confidence = 0.0
        if scores and emotion_key in scores:
            # DeepFace returns 0-100 percentages; normalise to 0-1
            raw = scores[emotion_key]
            confidence = raw / 100.0 if raw > 1.0 else raw

        # Build the final command
        command: dict = {
            "emotion": emotion_key,
            "confidence": round(confidence, 3),
        }
        command.update(expr)
        return command

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summary_string(cmd: dict) -> str:
        """
        One-line summary suitable for on-screen HUD display.

        Example: "smile | green LED | open_arms"
        """
        parts = []
        if cmd.get("face"):
            parts.append(cmd["face"])
        if cmd.get("led_color"):
            parts.append(f"{cmd['led_color']} LED")
        if cmd.get("gesture") and cmd["gesture"] != "none":
            parts.append(cmd["gesture"])
        return " | ".join(parts) if parts else "neutral"
