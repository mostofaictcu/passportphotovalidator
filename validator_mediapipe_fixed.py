#!/usr/bin/env python3
"""
Passport Photo Validator v3.1 for Bangladesh
MediaPipe-based with safe keypoint handling and warning suppression.
Fast detection-only by default; optional mesh for head pose/eyes.
"""

import os
import cv2
import numpy as np
import logging
import warnings

# Suppress MediaPipe/TensorFlow/absl noise before importing mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

logger = logging.getLogger(__name__)

# =============================================================================
# PASSPORT STANDARDS DATABASE
# =============================================================================
PASSPORT_STANDARDS = {
    "existing": {
        "MIN_SHARPNESS": 150,
        "LIGHT_BALANCE_THRESHOLD": 80,
        "MIN_RIGHT_MARGIN": 0.15,
        "MIN_BOTTOM_MARGIN": 0.15,
        "description": "Original hardcoded thresholds from v3.0"
    },
    "standard_images": {
        "vd1.jpeg": {
            "dimensions": (550, 550),
            "MIN_SHARPNESS": 178.03,
            "LIGHT_BALANCE_THRESHOLD": 47.88,
            "MIN_RIGHT_MARGIN": 0.285,
            "MIN_BOTTOM_MARGIN": 0.193,
            "face_center_x": 0.501,
            "face_center_y": 0.433,
            "face_width_ratio": 0.431,
            "face_height_ratio": 0.615,
            "head_tilt_yaw": 0.0,
            "head_tilt_pitch": 2.0,
            "head_tilt_roll": 0.5
        },
        "vd2.jpeg": {
            "dimensions": (550, 550),
            "MIN_SHARPNESS": 160.32,
            "LIGHT_BALANCE_THRESHOLD": 45.91,
            "MIN_RIGHT_MARGIN": 0.296,
            "MIN_BOTTOM_MARGIN": 0.204,
            "face_center_x": 0.501,
            "face_center_y": 0.422,
            "face_width_ratio": 0.409,
            "face_height_ratio": 0.593,
            "head_tilt_yaw": 0.0,
            "head_tilt_pitch": 1.5,
            "head_tilt_roll": 0.3
        },
        "mostofa_blue.jpg": {
            "dimensions": (300, 300),
            "MIN_SHARPNESS": 408.01,
            "LIGHT_BALANCE_THRESHOLD": 41.18,
            "MIN_RIGHT_MARGIN": 0.263,
            "MIN_BOTTOM_MARGIN": 0.167,
            "face_center_x": 0.498,
            "face_center_y": 0.452,
            "face_width_ratio": 0.473,
            "face_height_ratio": 0.667,
            "head_tilt_yaw": 0.0,
            "head_tilt_pitch": 0.0,
            "head_tilt_roll": 0.0
        },
        "passport_size.jpg": {
            "dimensions": (413, 531),
            "MIN_SHARPNESS": 163.93,
            "LIGHT_BALANCE_THRESHOLD": 76.56,
            "MIN_RIGHT_MARGIN": 0.179,
            "MIN_BOTTOM_MARGIN": 0.243,
            "face_center_x": 0.499,
            "face_center_y": 0.416,
            "face_width_ratio": 0.643,
            "face_height_ratio": 0.542,
            "head_tilt_yaw": 0.0,
            "head_tilt_pitch": 3.0,
            "head_tilt_roll": 1.0
        },
        "bdfemale.jpg": {
            "dimensions": (304, 384),
            "MIN_SHARPNESS": 968.21,
            "LIGHT_BALANCE_THRESHOLD": 80.96,
            "MIN_RIGHT_MARGIN": 0.27,
            "MIN_BOTTOM_MARGIN": 0.185,
            "face_center_x": 0.497,
            "face_center_y": 0.456,
            "face_width_ratio": 0.461,
            "face_height_ratio": 0.594,
            "head_tilt_yaw": 0.0,
            "head_tilt_pitch": 1.0,
            "head_tilt_roll": 0.0
        },
        "bdmale.jpg": {
            "dimensions": (300, 380),
            "MIN_SHARPNESS": 166.31,
            "LIGHT_BALANCE_THRESHOLD": 57.14,
            "MIN_RIGHT_MARGIN": 0.237,
            "MIN_BOTTOM_MARGIN": 0.2,
            "face_center_x": 0.498,
            "face_center_y": 0.447,
            "face_width_ratio": 0.527,
            "face_height_ratio": 0.553,
            "head_tilt_yaw": 0.0,
            "head_tilt_pitch": 2.5,
            "head_tilt_roll": 0.5
        }
    },
    "computed_from_standards": {
        "MIN_SHARPNESS": {
            "min": 160.32, "max": 968.21, "avg": 340.8,
            "description": "Higher is sharper. Range covers all 6 standard photos."
        },
        "LIGHT_BALANCE_THRESHOLD": {
            "min": 41.18, "max": 80.96, "avg": 58.27,
            "description": "Lower = stricter. Max is the most uneven standard photo."
        },
        "MIN_RIGHT_MARGIN": {
            "min": 0.179, "max": 0.296, "avg": 0.255,
            "description": "Minimum horizontal margin ratio seen in standard photos."
        },
        "MIN_BOTTOM_MARGIN": {
            "min": 0.167, "max": 0.243, "avg": 0.199,
            "description": "Minimum vertical margin ratio seen in standard photos."
        },
        "HEAD_TILT_YAW": {
            "min": 0.0, "max": 0.0, "avg": 0.0,
            "description": "Front-facing required. Yaw should be near 0."
        },
        "HEAD_TILT_PITCH": {
            "min": 0.0, "max": 3.0, "avg": 1.67,
            "description": "Slight pitch allowed. Range from standard photos."
        },
        "HEAD_TILT_ROLL": {
            "min": 0.0, "max": 1.0, "avg": 0.38,
            "description": "Slight roll allowed. Range from standard photos."
        }
    }
}


def _get_standard_range(metric_name):
    computed = PASSPORT_STANDARDS["computed_from_standards"]
    if metric_name not in computed:
        return None
    return computed[metric_name]["min"], computed[metric_name]["max"], computed[metric_name]["avg"]


def _is_within_standard_range(metric_name, value, tolerance_pct=0.0):
    range_data = _get_standard_range(metric_name)
    if range_data is None:
        return False, None
    min_val, max_val, avg_val = range_data
    if metric_name == "LIGHT_BALANCE_THRESHOLD":
        tol = tolerance_pct * (max_val - min_val) if (max_val - min_val) > 0 else 0
        lower, upper = min_val - tol, max_val + tol
    else:
        tol = tolerance_pct * (max_val - min_val) if (max_val - min_val) > 0 else 0
        lower, upper = min_val - tol, max_val + tol
    is_within = lower <= value <= upper
    return bool(is_within), {
        "value": round(float(value), 4),
        "standard_min": min_val,
        "standard_max": max_val,
        "standard_avg": avg_val,
        "tolerance_applied": round(tol, 4),
        "range_lower": round(lower, 4),
        "range_upper": round(upper, 4),
        "in_range": bool(is_within),
        "status": "within_standard_range" if is_within else (
            "above_standard_max" if value > upper else "below_standard_min"
        )
    }


class PassportPhotoValidatorV3:
    """
    MediaPipe-based passport photo validator.

    Args:
        image_path: Path to image file
        threshold_source: None (database ranges), 'existing', 'standard_min', 'standard_avg', 'strict'
        enable_mesh: If True, runs Face Mesh for head pose/eye/expression checks (slower).
                     If False, uses only Face Detection (fast, ~20-30ms).
    """

    def __init__(self, image_path, threshold_source=None, enable_mesh=False):
        self.image_path = image_path
        self.image = None
        self.gray = None
        self.height = None
        self.width = None
        self.has_alpha = False
        self.alpha_channel = None
        self.validation_results = {}
        self.validation_issues = []
        self.threshold_source = threshold_source
        self.enable_mesh = enable_mesh

        # MediaPipe results
        self.mp_face_bbox = None
        self.mp_face_keypoints = None
        self.mp_face_landmarks = None
        self.mp_head_pose = None

        # Dimensions
        self.STANDARD_WIDTH = 300
        self.STANDARD_HEIGHT = 300
        self.TOLERANCE = 1.0
        self.MIN_WIDTH = self.STANDARD_WIDTH * (1 - self.TOLERANCE)
        self.MAX_WIDTH = self.STANDARD_WIDTH * (1 + self.TOLERANCE)
        self.MIN_HEIGHT = self.STANDARD_HEIGHT * (1 - self.TOLERANCE)
        self.MAX_HEIGHT = self.STANDARD_HEIGHT * (1 + self.TOLERANCE)

        self._resolve_thresholds()
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh

            # Fast detection-only by default (model_selection=0 = short range, fastest)
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )

            # Mesh only initialized if requested (saves memory if not used)
            self.face_mesh = None
            if self.enable_mesh:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,  # False = much faster, still accurate enough
                    min_detection_confidence=0.5
                )
            self.mediapipe_available = True
        except ImportError:
            logger.error("MediaPipe not installed. Run: pip install mediapipe")
            self.mediapipe_available = False
            raise RuntimeError("MediaPipe is required. Install with: pip install mediapipe")

    def _resolve_thresholds(self):
        existing = PASSPORT_STANDARDS["existing"]
        computed = PASSPORT_STANDARDS["computed_from_standards"]
        source = self.threshold_source

        if source is None or source == "standard_range":
            self.MIN_SHARPNESS = None
            self.LIGHT_BALANCE_THRESHOLD = None
            self.MIN_RIGHT_MARGIN = None
            self.MIN_BOTTOM_MARGIN = None
            self.MAX_HEAD_TILT_YAW = 5.0
            self.MAX_HEAD_TILT_PITCH = 5.0
            self.MAX_HEAD_TILT_ROLL = 5.0
            self.use_standard_range = True
        elif source == "existing":
            self.MIN_SHARPNESS = existing["MIN_SHARPNESS"]
            self.LIGHT_BALANCE_THRESHOLD = existing["LIGHT_BALANCE_THRESHOLD"]
            self.MIN_RIGHT_MARGIN = existing["MIN_RIGHT_MARGIN"]
            self.MIN_BOTTOM_MARGIN = existing["MIN_BOTTOM_MARGIN"]
            self.MAX_HEAD_TILT_YAW = 5.0
            self.MAX_HEAD_TILT_PITCH = 5.0
            self.MAX_HEAD_TILT_ROLL = 5.0
            self.use_standard_range = False
        elif source == "standard_min":
            self.MIN_SHARPNESS = round(computed["MIN_SHARPNESS"]["min"], 2)
            self.LIGHT_BALANCE_THRESHOLD = round(computed["LIGHT_BALANCE_THRESHOLD"]["min"], 2)
            self.MIN_RIGHT_MARGIN = round(computed["MIN_RIGHT_MARGIN"]["min"], 3)
            self.MIN_BOTTOM_MARGIN = round(computed["MIN_BOTTOM_MARGIN"]["min"], 3)
            self.MAX_HEAD_TILT_YAW = 3.0
            self.MAX_HEAD_TILT_PITCH = 3.0
            self.MAX_HEAD_TILT_ROLL = 3.0
            self.use_standard_range = False
        elif source == "standard_avg":
            self.MIN_SHARPNESS = round(computed["MIN_SHARPNESS"]["avg"], 2)
            self.LIGHT_BALANCE_THRESHOLD = round(computed["LIGHT_BALANCE_THRESHOLD"]["avg"], 2)
            self.MIN_RIGHT_MARGIN = round(computed["MIN_RIGHT_MARGIN"]["avg"], 3)
            self.MIN_BOTTOM_MARGIN = round(computed["MIN_BOTTOM_MARGIN"]["avg"], 3)
            self.MAX_HEAD_TILT_YAW = 5.0
            self.MAX_HEAD_TILT_PITCH = 5.0
            self.MAX_HEAD_TILT_ROLL = 5.0
            self.use_standard_range = False
        elif source == "strict":
            self.MIN_SHARPNESS = round(max(existing["MIN_SHARPNESS"], computed["MIN_SHARPNESS"]["min"]), 2)
            self.LIGHT_BALANCE_THRESHOLD = round(min(existing["LIGHT_BALANCE_THRESHOLD"], computed["LIGHT_BALANCE_THRESHOLD"]["min"]), 2)
            self.MIN_RIGHT_MARGIN = round(max(existing["MIN_RIGHT_MARGIN"], computed["MIN_RIGHT_MARGIN"]["min"]), 3)
            self.MIN_BOTTOM_MARGIN = round(max(existing["MIN_BOTTOM_MARGIN"], computed["MIN_BOTTOM_MARGIN"]["min"]), 3)
            self.MAX_HEAD_TILT_YAW = 3.0
            self.MAX_HEAD_TILT_PITCH = 3.0
            self.MAX_HEAD_TILT_ROLL = 3.0
            self.use_standard_range = False
        else:
            raise ValueError(f"Unknown threshold_source: {source}")

    def load_image(self):
        try:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if self.image is None:
                logger.error(f"Failed to load image: {self.image_path}")
                return False

            if len(self.image.shape) == 3 and self.image.shape[2] == 4:
                self.has_alpha = True
                self.alpha_channel = self.image[:, :, 3]
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)
            else:
                self.has_alpha = False
                self.alpha_channel = None

            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.height, self.width = self.image.shape[:2]
            return True
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return False

    def _detect_face_mediapipe(self):
        """Run MediaPipe Face Detection. Safe keypoint handling."""
        if not self.mediapipe_available:
            return False

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # ---- Face Detection ----
        det_results = self.face_detection.process(rgb)
        if not det_results.detections:
            logger.warning("MediaPipe FaceDetection: no face found")
            return False

        detection = det_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        self.mp_face_bbox = (
            int(bbox.xmin * self.width),
            int(bbox.ymin * self.height),
            int(bbox.width * self.width),
            int(bbox.height * self.height)
        )

        # ---- Safe keypoint extraction ----
        # FIX: Some MediaPipe versions don't have .name or it's empty.
        # We use getattr with fallback to index-based names.
        keypoints = {}
        KEYPOINT_NAMES = ["LEFT_EYE", "RIGHT_EYE", "NOSE_TIP", "MOUTH_CENTER", "LEFT_EAR", "RIGHT_EAR"]
        for i, kp in enumerate(detection.location_data.relative_keypoints):
            # Try .name first, fallback to index-based name
            kp_name = getattr(kp, 'name', None)
            if not kp_name or kp_name == '':
                kp_name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"keypoint_{i}"
            keypoints[kp_name] = (int(kp.x * self.width), int(kp.y * self.height))
        self.mp_face_keypoints = keypoints

        # ---- Optional Face Mesh ----
        if self.enable_mesh and self.face_mesh is not None:
            mesh_results = self.face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                self.mp_face_landmarks = mesh_results.multi_face_landmarks[0]
                self._estimate_head_pose()

        return True

    def _estimate_head_pose(self):
        """Estimate head pose from 3D face landmarks."""
        if self.mp_face_landmarks is None:
            return

        landmarks = self.mp_face_landmarks.landmark

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        image_points = np.array([
            (landmarks[1].x * self.width, landmarks[1].y * self.height),
            (landmarks[152].x * self.width, landmarks[152].y * self.height),
            (landmarks[263].x * self.width, landmarks[263].y * self.height),
            (landmarks[33].x * self.width, landmarks[33].y * self.height),
            (landmarks[287].x * self.width, landmarks[287].y * self.height),
            (landmarks[57].x * self.width, landmarks[57].y * self.height)
        ], dtype=np.float64)

        focal_length = self.width
        center = (self.width / 2, self.height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            yaw, pitch, roll = euler_angles.flatten()
            yaw = ((yaw + 180) % 360) - 180
            pitch = ((pitch + 180) % 360) - 180
            roll = ((roll + 180) % 360) - 180
            self.mp_head_pose = (float(yaw), float(pitch), float(roll))

    def _get_landmark_bbox(self):
        """Get tighter bounding box from landmarks if available, else detection bbox."""
        if self.mp_face_landmarks is None:
            return self.mp_face_bbox
        xs = [lm.x * self.width for lm in self.mp_face_landmarks.landmark]
        ys = [lm.y * self.height for lm in self.mp_face_landmarks.landmark]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def check_image_dimensions(self):
        valid = (self.MIN_HEIGHT <= self.height <= self.MAX_HEIGHT and
                 self.MIN_WIDTH <= self.width <= self.MAX_WIDTH)
        aspect_ratio = self.width / self.height
        standard_aspect = self.STANDARD_WIDTH / self.STANDARD_HEIGHT
        self.validation_results['dimensions'] = {
            'valid': bool(valid),
            'width': int(self.width), 'height': int(self.height),
            'aspect_ratio': round(float(aspect_ratio), 2),
            'standard_aspect_ratio': round(float(standard_aspect), 2),
            'width_range': f"{self.MIN_WIDTH:.0f}-{self.MAX_WIDTH:.0f}px",
            'height_range': f"{self.MIN_HEIGHT:.0f}-{self.MAX_HEIGHT:.0f}px"
        }
        return bool(valid)

    def check_photo_sharpness(self):
        try:
            laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
            sharpness_score = float(laplacian.var())
            rounded_score = round(sharpness_score, 2)

            if self.use_standard_range:
                is_valid, range_info = _is_within_standard_range('MIN_SHARPNESS', rounded_score, tolerance_pct=0.0)
                threshold_used = f"range [{range_info['standard_min']}-{range_info['standard_max']}]"
            else:
                is_valid = rounded_score >= self.MIN_SHARPNESS
                range_info = None
                threshold_used = self.MIN_SHARPNESS

            if not is_valid:
                self.validation_issues.append('ISSUE_PHOTO_SHARPNESS_BAD')

            self.validation_results['sharpness'] = {
                'valid': bool(is_valid),
                'laplacian_variance': rounded_score,
                'threshold_used': threshold_used,
                'threshold_source': self.threshold_source,
                'status': 'Sharp' if is_valid else 'Blurry/Bad Quality',
                'issue_code': 'ISSUE_PHOTO_SHARPNESS_BAD' if not is_valid else None,
                'standard_range': range_info
            }
            return bool(is_valid)
        except Exception as e:
            self.validation_results['sharpness'] = {'valid': False, 'error': str(e)}
            return False

    def check_face_light_balance(self, face_bbox=None):
        try:
            if face_bbox is None:
                analysis_region = self.gray
            else:
                x, y, w, h = face_bbox
                pad = int(w * 0.1)
                x_start = max(0, x - pad)
                y_start = max(0, y - pad)
                x_end = min(self.width, x + w + pad)
                y_end = min(self.height, y + h + pad)
                analysis_region = self.gray[y_start:y_end, x_start:x_end]

            mid_point = analysis_region.shape[1] // 2
            left_half = analysis_region[:, :mid_point]
            right_half = analysis_region[:, mid_point:]
            mid_height = analysis_region.shape[0] // 2
            top_half = analysis_region[:mid_height, :]
            bottom_half = analysis_region[mid_height:, :]

            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)
            top_brightness = np.mean(top_half)
            bottom_brightness = np.mean(bottom_half)
            lr_diff = abs(left_brightness - right_brightness)
            tb_diff = abs(top_brightness - bottom_brightness)
            overall_variance = float(np.var(analysis_region))
            effective_threshold = round(float(np.sqrt(overall_variance)), 2)

            # Very comfortable light balance ranges:
            # Standard images: LR diff 5.4-30.6, TB diff 5.2-48.1, variance 1695-6555
            # Comfortable zone allows for imperfect studio lighting, shadows, etc.
            balanced_lr = lr_diff < 50
            balanced_tb = tb_diff < 80

            if self.use_standard_range:
                # 20% tolerance on standard range for very comfortable checking
                is_within_range, range_info = _is_within_standard_range('LIGHT_BALANCE_THRESHOLD', effective_threshold, tolerance_pct=0.2)
                is_valid = is_within_range and balanced_lr and balanced_tb
                threshold_used = f"range [{range_info['range_lower']:.1f}-{range_info['range_upper']:.1f}]"
            else:
                # Very comfortable: variance up to 10000 (threshold 100)
                even_variance = overall_variance < (self.LIGHT_BALANCE_THRESHOLD ** 2)
                is_valid = balanced_lr and balanced_tb and even_variance
                range_info = None
                threshold_used = self.LIGHT_BALANCE_THRESHOLD

            if not is_valid:
                self.validation_issues.append('ISSUE_FACE_LIGHT_NOT_BALANCE')

            self.validation_results['light_balance'] = {
                'valid': bool(is_valid),
                'left_brightness': round(float(left_brightness), 2),
                'right_brightness': round(float(right_brightness), 2),
                'top_brightness': round(float(top_brightness), 2),
                'bottom_brightness': round(float(bottom_brightness), 2),
                'lr_difference': round(float(lr_diff), 2),
                'tb_difference': round(float(tb_diff), 2),
                'overall_variance': round(float(overall_variance), 2),
                'effective_threshold': effective_threshold,
                'threshold_used': threshold_used,
                'threshold_source': self.threshold_source,
                'balanced_lr': bool(balanced_lr),
                'balanced_tb': bool(balanced_tb),
                'issue_code': 'ISSUE_FACE_LIGHT_NOT_BALANCE' if not is_valid else None,
                'standard_range': range_info
            }
            return bool(is_valid)
        except Exception as e:
            self.validation_results['light_balance'] = {'valid': False, 'error': str(e)}
            return False

    def check_face_brightness(self, face_bbox=None):
        try:
            if face_bbox is None:
                self.validation_results['face_brightness'] = {
                    'valid': True, 'message': 'No face region to analyze'
                }
                return True

            x, y, w, h = face_bbox
            pad = int(w * 0.05)
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(self.width, x + w + pad)
            y_end = min(self.height, y + h + pad)
            face_roi = self.gray[y_start:y_end, x_start:x_end]

            face_brightness = np.mean(face_roi)
            overall_brightness = cv2.mean(self.gray)[0]

            background_mask = np.ones(self.gray.shape, dtype=np.uint8)
            background_mask[y_start:y_end, x_start:x_end] = 0
            background_pixels = self.gray[background_mask == 1]
            background_brightness = np.mean(background_pixels) if len(background_pixels) > 0 else overall_brightness

            # Comfortable ranges derived from standard images:
            # Face brightness: 92.4-154.6 (avg 121.9)
            # Brightness diff: 30.9-91.5 (bg always brighter than face)
            # Overall brightness: 117.2-203.2
            is_face_bright_enough = face_brightness >= 60
            is_face_not_overexposed = face_brightness <= 245
            brightness_diff = background_brightness - face_brightness
            is_brightness_balanced = -30 <= brightness_diff <= 100
            is_overall_not_overexposed = overall_brightness <= 240

            is_valid = (is_face_bright_enough and is_face_not_overexposed and
                       is_brightness_balanced and is_overall_not_overexposed)

            if not is_valid:
                self.validation_issues.append('ISSUE_FACE_BRIGHTNESS_BAD')

            self.validation_results['face_brightness'] = {
                'valid': bool(is_valid),
                'face_brightness': round(float(face_brightness), 2),
                'background_brightness': round(float(background_brightness), 2),
                'overall_brightness': round(float(overall_brightness), 2),
                'brightness_difference': round(float(brightness_diff), 2),
                'face_bright_enough': bool(is_face_bright_enough),
                'face_not_overexposed': bool(is_face_not_overexposed),
                'brightness_balanced': bool(is_brightness_balanced),
                'overall_not_overexposed': bool(is_overall_not_overexposed),
                'issue_code': 'ISSUE_FACE_BRIGHTNESS_BAD' if not is_valid else None
            }
            return bool(is_valid)
        except Exception as e:
            self.validation_results['face_brightness'] = {'valid': False, 'error': str(e)}
            return False

    def check_head_pose(self):
        if self.mp_head_pose is None:
            self.validation_results['head_pose'] = {
                'valid': True,  # Pass if mesh not enabled
                'message': 'Head pose check skipped (mesh not enabled)',
                'issue_code': None
            }
            return True

        yaw, pitch, roll = self.mp_head_pose

        if self.use_standard_range:
            yaw_valid, yaw_info = _is_within_standard_range('HEAD_TILT_YAW', abs(yaw), tolerance_pct=0.2)
            pitch_valid, pitch_info = _is_within_standard_range('HEAD_TILT_PITCH', abs(pitch), tolerance_pct=0.2)
            roll_valid, roll_info = _is_within_standard_range('HEAD_TILT_ROLL', abs(roll), tolerance_pct=0.2)
        else:
            yaw_valid = abs(yaw) <= self.MAX_HEAD_TILT_YAW
            pitch_valid = abs(pitch) <= self.MAX_HEAD_TILT_PITCH
            roll_valid = abs(roll) <= self.MAX_HEAD_TILT_ROLL
            yaw_info = pitch_info = roll_info = None

        is_valid = yaw_valid and pitch_valid and roll_valid

        if not is_valid:
            self.validation_issues.append('ISSUE_HEAD_NOT_FRONT_FACING')

        self.validation_results['head_pose'] = {
            'valid': bool(is_valid),
            'yaw_degrees': round(float(yaw), 2),
            'pitch_degrees': round(float(pitch), 2),
            'roll_degrees': round(float(roll), 2),
            'yaw_valid': bool(yaw_valid),
            'pitch_valid': bool(pitch_valid),
            'roll_valid': bool(roll_valid),
            'threshold_source': self.threshold_source,
            'message': 'Head is front-facing' if is_valid else 'Head is not front-facing',
            'issue_code': 'ISSUE_HEAD_NOT_FRONT_FACING' if not is_valid else None,
            'yaw_standard_range': yaw_info,
            'pitch_standard_range': pitch_info,
            'roll_standard_range': roll_info
        }
        return bool(is_valid)

    def check_eyes_open(self):
        if self.mp_face_landmarks is None:
            self.validation_results['eyes_open'] = {
                'valid': True,
                'message': 'Face mesh not enabled, skipping eye check'
            }
            return True

        landmarks = self.mp_face_landmarks.landmark

        def eye_openness(eye_indices):
            top = max(landmarks[i].y for i in eye_indices['top'])
            bottom = min(landmarks[i].y for i in eye_indices['bottom'])
            left = min(landmarks[i].x for i in eye_indices['corner'])
            right = max(landmarks[i].x for i in eye_indices['corner'])
            height = abs(top - bottom) * self.height
            width = abs(right - left) * self.width
            return height / width if width > 0 else 0

        left_eye = {'top': [159, 160], 'bottom': [145, 144], 'corner': [33, 133]}
        right_eye = {'top': [386, 385], 'bottom': [374, 373], 'corner': [362, 263]}

        left_ratio = eye_openness(left_eye)
        right_ratio = eye_openness(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2

        is_open = avg_ratio > 0.15
        is_symmetric = abs(left_ratio - right_ratio) < 0.1
        is_valid = is_open and is_symmetric

        if not is_valid:
            self.validation_issues.append('ISSUE_EYES_NOT_OPEN')

        self.validation_results['eyes_open'] = {
            'valid': bool(is_valid),
            'left_eye_ratio': round(float(left_ratio), 3),
            'right_eye_ratio': round(float(right_ratio), 3),
            'avg_ratio': round(float(avg_ratio), 3),
            'eyes_open': bool(is_open),
            'eyes_symmetric': bool(is_symmetric),
            'threshold': 0.15,
            'message': 'Eyes are open and symmetric' if is_valid else 'Eyes may be closed or uneven',
            'issue_code': 'ISSUE_EYES_NOT_OPEN' if not is_valid else None
        }
        return bool(is_valid)

    def check_neutral_expression(self):
        if self.mp_face_landmarks is None:
            self.validation_results['neutral_expression'] = {
                'valid': True,
                'message': 'Face mesh not enabled, skipping expression check'
            }
            return True

        landmarks = self.mp_face_landmarks.landmark
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        top_lip = landmarks[0]
        bottom_lip = landmarks[17]

        mouth_width = abs(right_corner.x - left_corner.x) * self.width
        mouth_height = abs(bottom_lip.y - top_lip.y) * self.height
        smile_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        corner_lift = (left_corner.y + right_corner.y) / 2 - top_lip.y
        is_smiling = smile_ratio > 0.3 or corner_lift < -0.02
        is_neutral = not is_smiling

        if not is_neutral:
            self.validation_issues.append('ISSUE_NOT_NEUTRAL_EXPRESSION')

        self.validation_results['neutral_expression'] = {
            'valid': bool(is_neutral),
            'mouth_width_px': round(float(mouth_width), 1),
            'mouth_height_px': round(float(mouth_height), 1),
            'smile_ratio': round(float(smile_ratio), 3),
            'is_neutral': bool(is_neutral),
            'message': 'Neutral expression' if is_neutral else 'Non-neutral expression detected',
            'issue_code': 'ISSUE_NOT_NEUTRAL_EXPRESSION' if not is_neutral else None
        }
        return bool(is_neutral)

    def check_shoulder_visibility(self, face_bbox):
        if face_bbox is None:
            return False, "No face detected"
        x, y, w, h = face_bbox
        face_bottom = y + h
        space_below_face = self.height - face_bottom
        required_shoulder_space = self.height * 0.15
        has_shoulder_space = space_below_face >= required_shoulder_space
        face_width_ratio = w / self.width
        shoulder_detection_possible = face_width_ratio < 0.8
        is_valid = has_shoulder_space and shoulder_detection_possible
        return is_valid, {
            'space_below_face_px': int(space_below_face),
            'required_space_px': int(required_shoulder_space),
            'space_percent': round(float((space_below_face / self.height) * 100), 1),
            'face_width_ratio': round(float(face_width_ratio), 2),
            'has_shoulder_space': bool(has_shoulder_space),
            'shoulders_detectable': bool(shoulder_detection_possible)
        }

    def check_right_side_visibility(self, face_bbox):
        if face_bbox is None:
            return False, {'valid': False, 'message': 'No face detected', 'issue_code': None}

        x, y, w, h = face_bbox
        right_margin = self.width - (x + w)
        left_margin = x

        if self.use_standard_range:
            min_margin_ratio = min(left_margin / self.width, right_margin / self.width)
            is_within_range, range_info = _is_within_standard_range('MIN_RIGHT_MARGIN', round(min_margin_ratio, 3), tolerance_pct=0.0)
            required_margin = self.width * range_info['standard_min'] if range_info else self.width * 0.15
        else:
            required_margin = self.width * self.MIN_RIGHT_MARGIN
            range_info = None

        has_right_margin = right_margin >= required_margin
        has_left_margin = left_margin >= required_margin
        face_width_ratio = w / self.width
        face_width_adequate = face_width_ratio >= 0.25
        left_margin_percent = (left_margin / self.width) * 100
        right_margin_percent = (right_margin / self.width) * 100
        margin_balance_diff = abs(left_margin_percent - right_margin_percent)
        margins_balanced = margin_balance_diff <= 10
        face_center_x = (x + w/2) / self.width
        is_centered = 0.4 <= face_center_x <= 0.6

        is_valid = (has_right_margin and has_left_margin and
                   face_width_adequate and margins_balanced and is_centered)

        if not is_valid:
            self.validation_issues.append('ISSUE_RIGHT_NOT_FULL')

        return bool(is_valid), {
            'valid': bool(is_valid),
            'face_x': int(x), 'face_width': int(w),
            'left_margin_px': int(left_margin), 'right_margin_px': int(right_margin),
            'left_margin_percent': round(float(left_margin_percent), 1),
            'right_margin_percent': round(float(right_margin_percent), 1),
            'required_margin_px': int(required_margin),
            'face_width_ratio': round(float(face_width_ratio), 3),
            'face_center_x_ratio': round(float(face_center_x), 3),
            'margin_balance_diff_percent': round(float(margin_balance_diff), 1),
            'has_right_margin': bool(has_right_margin),
            'has_left_margin': bool(has_left_margin),
            'face_width_adequate': bool(face_width_adequate),
            'margins_balanced': bool(margins_balanced),
            'is_centered': bool(is_centered),
            'message': 'Face properly framed with balanced margins' if is_valid else (
                'Face too small' if not face_width_adequate else
                'Face not centered' if not (margins_balanced and is_centered) else
                'Insufficient margin(s)'
            ),
            'issue_code': 'ISSUE_RIGHT_NOT_FULL' if not is_valid else None,
            'standard_range': range_info
        }

    def check_left_side_visibility(self, face_bbox):
        if face_bbox is None:
            return False, {'valid': False, 'message': 'No face detected', 'issue_code': None}

        x, y, w, h = face_bbox
        left_margin = x
        right_margin = self.width - (x + w)

        if self.use_standard_range:
            min_margin_ratio = min(left_margin / self.width, right_margin / self.width)
            is_within_range, range_info = _is_within_standard_range('MIN_RIGHT_MARGIN', round(min_margin_ratio, 3), tolerance_pct=0.0)
            required_margin = self.width * range_info['standard_min'] if range_info else self.width * 0.15
        else:
            required_margin = self.width * self.MIN_RIGHT_MARGIN
            range_info = None

        has_left_margin = left_margin >= required_margin
        has_right_margin = right_margin >= required_margin
        face_width_ratio = w / self.width
        face_width_adequate = face_width_ratio >= 0.25
        left_margin_percent = (left_margin / self.width) * 100
        right_margin_percent = (right_margin / self.width) * 100
        margin_balance_diff = abs(left_margin_percent - right_margin_percent)
        margins_balanced = margin_balance_diff <= 10
        face_center_x = (x + w/2) / self.width
        is_centered = 0.4 <= face_center_x <= 0.6

        is_valid = (has_left_margin and has_right_margin and
                   face_width_adequate and margins_balanced and is_centered)

        if not is_valid:
            self.validation_issues.append('ISSUE_LEFT_NOT_FULL')

        return bool(is_valid), {
            'valid': bool(is_valid),
            'face_x': int(x), 'face_width': int(w),
            'left_margin_px': int(left_margin), 'right_margin_px': int(right_margin),
            'left_margin_percent': round(float(left_margin_percent), 1),
            'right_margin_percent': round(float(right_margin_percent), 1),
            'required_margin_px': int(required_margin),
            'face_width_ratio': round(float(face_width_ratio), 3),
            'face_center_x_ratio': round(float(face_center_x), 3),
            'margin_balance_diff_percent': round(float(margin_balance_diff), 1),
            'has_left_margin': bool(has_left_margin),
            'has_right_margin': bool(has_right_margin),
            'face_width_adequate': bool(face_width_adequate),
            'margins_balanced': bool(margins_balanced),
            'is_centered': bool(is_centered),
            'message': 'Face properly framed with balanced margins' if is_valid else (
                'Face too small' if not face_width_adequate else
                'Face not centered' if not (margins_balanced and is_centered) else
                'Insufficient left margin'
            ),
            'issue_code': 'ISSUE_LEFT_NOT_FULL' if not is_valid else None,
            'standard_range': range_info
        }

    def check_bottom_face_visibility(self, face_bbox):
        if face_bbox is None:
            return False, {'valid': False, 'message': 'No face detected', 'issue_code': None}

        x, y, w, h = face_bbox
        bottom_margin = self.height - (y + h)
        top_margin = y

        if self.use_standard_range:
            min_margin_ratio = min(top_margin / self.height, bottom_margin / self.height)
            is_within_range, range_info = _is_within_standard_range('MIN_BOTTOM_MARGIN', round(min_margin_ratio, 3), tolerance_pct=0.0)
            required_margin = self.height * range_info['standard_min'] if range_info else self.height * 0.15
        else:
            required_margin = self.height * self.MIN_BOTTOM_MARGIN
            range_info = None

        has_bottom_margin = bottom_margin >= required_margin
        has_top_margin = top_margin >= (self.height * 0.10)
        face_height_ratio = h / self.height
        face_height_adequate = face_height_ratio >= 0.20
        top_margin_percent = (top_margin / self.height) * 100
        bottom_margin_percent = (bottom_margin / self.height) * 100
        margin_balance_diff = abs(top_margin_percent - bottom_margin_percent)
        margins_balanced = margin_balance_diff <= 25
        face_center_y = (y + h/2) / self.height
        is_vertically_centered = 0.35 <= face_center_y <= 0.65

        is_valid = (has_bottom_margin and has_top_margin and
                   face_height_adequate and margins_balanced and is_vertically_centered)

        if not is_valid:
            self.validation_issues.append('ISSUE_BOTTOM_NOT_FULL')

        return bool(is_valid), {
            'valid': bool(is_valid),
            'face_y': int(y), 'face_height': int(h),
            'top_margin_px': int(top_margin), 'bottom_margin_px': int(bottom_margin),
            'top_margin_percent': round(float(top_margin_percent), 1),
            'bottom_margin_percent': round(float(bottom_margin_percent), 1),
            'required_margin_px': int(required_margin),
            'face_height_ratio': round(float(face_height_ratio), 3),
            'face_center_y_ratio': round(float(face_center_y), 3),
            'margin_balance_diff_percent': round(float(margin_balance_diff), 1),
            'has_bottom_margin': bool(has_bottom_margin),
            'has_top_margin': bool(has_top_margin),
            'face_height_adequate': bool(face_height_adequate),
            'margins_balanced': bool(margins_balanced),
            'is_vertically_centered': bool(is_vertically_centered),
            'message': 'Face properly framed vertically' if is_valid else (
                'Face too small' if not face_height_adequate else
                'Face not centered' if not (margins_balanced and is_vertically_centered) else
                'Insufficient margin(s)'
            ),
            'issue_code': 'ISSUE_BOTTOM_NOT_FULL' if not is_valid else None,
            'standard_range': range_info
        }

    def check_selfie_detection(self, face_bbox):
        if face_bbox is None:
            return False, "No face detected"
        x, y, w, h = face_bbox
        face_area_ratio = (w * h) / (self.width * self.height)
        is_selfie = face_area_ratio > 0.40
        if is_selfie:
            return True, f"Selfie detected - face occupies {face_area_ratio*100:.1f}% of frame"
        return False, f"Face size normal - {face_area_ratio*100:.1f}% of frame"

    def detect_faces(self):
        if not self.mediapipe_available:
            self.validation_results['face_detection'] = {
                'valid': False, 'message': 'MediaPipe not available',
                'issue_code': 'ISSUE_MEDIAPIPE_UNAVAILABLE'
            }
            self.validation_issues.append('ISSUE_MEDIAPIPE_UNAVAILABLE')
            return None, False

        success = self._detect_face_mediapipe()
        if not success:
            self.validation_results['face_detection'] = {
                'valid': False, 'face_count': 0, 'required_faces': 1,
                'message': 'No face detected by MediaPipe',
                'issue_code': 'ISSUE_NO_FACE_DETECTED'
            }
            self.validation_issues.append('ISSUE_NO_FACE_DETECTED')
            return None, False

        face_bbox = self._get_landmark_bbox()
        is_selfie, selfie_msg = self.check_selfie_detection(face_bbox)
        has_shoulders, shoulder_info = self.check_shoulder_visibility(face_bbox)
        has_left_visibility, left_visibility_info = self.check_left_side_visibility(face_bbox)
        has_right_visibility, right_visibility_info = self.check_right_side_visibility(face_bbox)
        has_bottom_visibility, bottom_visibility_info = self.check_bottom_face_visibility(face_bbox)

        face_count = 1
        is_valid = (face_count == 1 and not is_selfie and
                   has_shoulders and has_left_visibility and
                   has_right_visibility and has_bottom_visibility)

        self.validation_results['face_detection'] = {
            'valid': bool(is_valid),
            'face_count': int(face_count),
            'required_faces': 1,
            'selfie_detected': bool(is_selfie),
            'selfie_note': selfie_msg,
            'shoulder_detection': shoulder_info,
            'left_side_detection': left_visibility_info,
            'right_side_detection': right_visibility_info,
            'bottom_detection': bottom_visibility_info,
            'mediapipe_bbox': self.mp_face_bbox,
            'mediapipe_keypoints': {k: v for k, v in (self.mp_face_keypoints or {}).items()},
            'has_face_mesh': self.mp_face_landmarks is not None
        }

        return face_bbox, bool(is_valid)

    def check_face_position(self, face_bbox):
        if face_bbox is None:
            self.validation_results['face_position'] = {'valid': False, 'message': 'No face detected'}
            return False

        x, y, w, h = face_bbox
        face_center_x = (x + w/2) / self.width
        face_center_y = (y + h/2) / self.height
        valid_x = 0.25 <= face_center_x <= 0.75
        valid_y = 0.25 <= face_center_y <= 0.55
        is_valid = valid_x and valid_y
        face_area_ratio = (w * h) / (self.width * self.height)

        self.validation_results['face_position'] = {
            'valid': bool(is_valid),
            'center_x_ratio': round(float(face_center_x), 2),
            'center_y_ratio': round(float(face_center_y), 2),
            'face_area_ratio': round(float(face_area_ratio), 2),
            'tolerance': '±25% horizontal, 25-55% vertical',
            'message': 'Face properly positioned' if is_valid else 'Face not properly positioned'
        }
        return bool(is_valid)

    def check_lighting(self):
        # Check if image has proper lighting with comfortable ranges.
        # Ranges derived from standard images:
        # - Avg brightness: 117.2 - 203.2 (comfortable: 50-250)
        # - Std dev: 47.9 - 68.5 (comfortable: < 100)
        avg_brightness = cv2.mean(self.gray)[0]
        mean = avg_brightness
        variance = np.mean((self.gray - mean) ** 2)
        std_dev = np.sqrt(variance)

        # Comfortable ranges (wider than original 70-230 and <80)
        brightness_valid = 50 <= avg_brightness <= 250
        lighting_even = std_dev < 100
        is_valid = brightness_valid and lighting_even

        self.validation_results['lighting'] = {
            'valid': bool(is_valid),
            'avg_brightness': round(float(avg_brightness), 2),
            'brightness_range': '50-250 (comfortable)',
            'std_deviation': round(float(std_dev), 2),
            'lighting_uniformity': 'even' if lighting_even else 'uneven',
            'threshold': 'std_dev < 100 (comfortable)',
            'constraints': {
                'min_avg_brightness': 50,
                'max_avg_brightness': 250,
                'max_std_dev': 100
            }
        }
        return bool(is_valid)

    def check_background_color(self):
        try:
            if self.has_alpha and self.alpha_channel is not None:
                transparent_pixels = np.sum(self.alpha_channel < 200)
                total_pixels = self.alpha_channel.size
                transparency_ratio = transparent_pixels / total_pixels
                if transparency_ratio > 0.1:
                    self.validation_results['background'] = {
                        'valid': True,
                        'background_type': 'transparent',
                        'transparency_ratio': round(float(transparency_ratio), 2),
                        'message': 'Transparent background - acceptable'
                    }
                    return True

            border_size = 20
            top = self.image[:border_size, :, :]
            bottom = self.image[-border_size:, :, :]
            left = self.image[:, :border_size, :]
            right = self.image[:, -border_size:, :]
            border_pixels = np.concatenate([
                top.reshape(-1, 3), bottom.reshape(-1, 3),
                left.reshape(-1, 3), right.reshape(-1, 3)
            ])
            mean_color = np.mean(border_pixels, axis=0)
            b_mean, g_mean, r_mean = mean_color
            brightness = (b_mean + g_mean + r_mean) / 3.0
            color_variance = max(abs(r_mean-g_mean), abs(g_mean-b_mean), abs(r_mean-b_mean))
            blue_dominance = b_mean - r_mean

            background_type = 'unknown'
            is_valid_bg = False

            if r_mean > 180 and g_mean > 180 and b_mean > 180 and color_variance < 40:
                background_type, is_valid_bg = 'white', True
            elif r_mean > 180 and g_mean > 180 and b_mean > 180 and color_variance < 60:
                background_type, is_valid_bg = 'light/off-white', True
            elif (r_mean > 100 and g_mean > 100 and b_mean > 140 and blue_dominance > 40):
                background_type, is_valid_bg = 'blue', True
            elif (b_mean > 80 and blue_dominance > 20 and brightness > 120):
                background_type, is_valid_bg = 'blue', True
            elif (b_mean > 80 and blue_dominance > 30 and brightness > 80):
                background_type, is_valid_bg = 'blue', True
            elif (color_variance < 15 and brightness > 100):
                background_type, is_valid_bg = 'grey', True
            elif (color_variance < 20 and brightness > 140):
                background_type, is_valid_bg = 'light grey', True
            elif brightness > 150:
                background_type, is_valid_bg = 'light/off-white', True
            else:
                background_type, is_valid_bg = 'other', False

            self.validation_results['background'] = {
                'valid': bool(is_valid_bg),
                'background_type': background_type,
                'brightness': round(float(brightness), 2),
                'color_rgb': {
                    'red': round(float(r_mean), 2),
                    'green': round(float(g_mean), 2),
                    'blue': round(float(b_mean), 2)
                },
                'accepted_colors': ['white', 'blue', 'transparent'],
                'message': f'Accepted - {background_type}' if is_valid_bg else f'Not accepted - {background_type}'
            }
            return bool(is_valid_bg)
        except Exception as e:
            self.validation_results['background'] = {'valid': False, 'error': str(e), 'background_type': 'unknown'}
            return False

    def validate(self):
        if not self.load_image():
            self.validation_results['overall'] = {'valid': False, 'message': 'Failed to load image'}
            return False

        face_bbox, faces_valid = self.detect_faces()

        checks = [
            ('dimensions', self.check_image_dimensions()),
            ('sharpness', self.check_photo_sharpness()),
            ('faces', faces_valid),
            ('face_position', self.check_face_position(face_bbox)),
            ('lighting', self.check_lighting()),
            ('light_balance', self.check_face_light_balance(face_bbox)),
            ('face_brightness', self.check_face_brightness(face_bbox)),
            ('background', self.check_background_color()),
            ('head_pose', self.check_head_pose()),
            ('eyes_open', self.check_eyes_open()),
            ('neutral_expression', self.check_neutral_expression()),
        ]

        all_valid = all(result for _, result in checks)

        issues_summary = {
            'has_issues': len(self.validation_issues) > 0,
            'issue_count': len(self.validation_issues),
            'issues': self.validation_issues
        }

        standards_reference = {
            'threshold_source': self.threshold_source,
            'use_standard_range': self.use_standard_range,
            'enable_mesh': self.enable_mesh,
            'passport_standards_available': list(PASSPORT_STANDARDS.keys())
        }
        if not self.use_standard_range:
            standards_reference['active_thresholds'] = {
                'MIN_SHARPNESS': self.MIN_SHARPNESS,
                'LIGHT_BALANCE_THRESHOLD': self.LIGHT_BALANCE_THRESHOLD,
                'MIN_RIGHT_MARGIN': self.MIN_RIGHT_MARGIN,
                'MIN_BOTTOM_MARGIN': self.MIN_BOTTOM_MARGIN
            }

        self.validation_results['overall'] = {
            'valid': bool(all_valid),
            'passed_checks': int(sum(1 for _, result in checks if result)),
            'total_checks': int(len(checks)),
            'status': 'PASSED' if all_valid else 'FAILED',
            'version': 'v3.1-mediapipe',
            'issues': issues_summary,
            'standards_reference': standards_reference
        }

        return bool(all_valid)
