#!/usr/bin/env python3
"""
Passport Photo Validator Core Module v2.2 for Bangladesh
Synchronous validation logic - runs in thread pool from async context.

PASSPORT_STANDARDS acts as a database of acceptable metric ranges extracted
from standard reference images. By default, uploaded images are validated
against these ranges. Existing thresholds are preserved for backward compat.
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# PASSPORT STANDARDS DATABASE
# Extracted from 6 standard Bangladesh passport reference images.
# Each metric contains: per-image data + computed min/max/avg ranges.
# =============================================================================
PASSPORT_STANDARDS = {
    "existing": {
        "MIN_SHARPNESS": 150,
        "LIGHT_BALANCE_THRESHOLD": 80,
        "MIN_RIGHT_MARGIN": 0.15,
        "MIN_BOTTOM_MARGIN": 0.15,
        "description": "Original hardcoded thresholds from v2.0"
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
            "face_height_ratio": 0.615
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
            "face_height_ratio": 0.593
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
            "face_height_ratio": 0.667
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
            "face_height_ratio": 0.542
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
            "face_height_ratio": 0.594
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
            "face_height_ratio": 0.553
        }
    },
    "computed_from_standards": {
        "MIN_SHARPNESS": {
            "min": 160.32,
            "max": 968.21,
            "avg": 340.8,
            "description": "Higher is sharper. Range covers all 6 standard photos."
        },
        "LIGHT_BALANCE_THRESHOLD": {
            "min": 41.18,
            "max": 80.96,
            "avg": 58.27,
            "description": "Lower = stricter. Max is the most uneven standard photo."
        },
        "MIN_RIGHT_MARGIN": {
            "min": 0.179,
            "max": 0.296,
            "avg": 0.255,
            "description": "Minimum horizontal margin ratio seen in standard photos."
        },
        "MIN_BOTTOM_MARGIN": {
            "min": 0.167,
            "max": 0.243,
            "avg": 0.199,
            "description": "Minimum vertical margin ratio seen in standard photos."
        }
    }
}


def _get_standard_range(metric_name):
    """Get min/max range for a metric from the standards database."""
    computed = PASSPORT_STANDARDS["computed_from_standards"]
    if metric_name not in computed:
        return None
    return computed[metric_name]["min"], computed[metric_name]["max"], computed[metric_name]["avg"]


def _is_within_standard_range(metric_name, value, tolerance_pct=0.0):
    """Check if a value lies within the standard image range (with optional tolerance).

    Returns: (is_within, range_info_dict)
    """
    range_data = _get_standard_range(metric_name)
    if range_data is None:
        return False, None

    min_val, max_val, avg_val = range_data

    # Apply tolerance to the range bounds
    if metric_name == "LIGHT_BALANCE_THRESHOLD":
        # For light balance, the threshold is a cap on sqrt(variance).
        # Lower values = more balanced. The range min is the best (most balanced),
        # max is the worst acceptable. So we allow values slightly above max.
        tol = tolerance_pct * (max_val - min_val) if (max_val - min_val) > 0 else 0
        lower = min_val - tol
        upper = max_val + tol
    else:
        # For sharpness and margins, higher is generally better.
        # The range min is the minimum acceptable standard.
        tol = tolerance_pct * (max_val - min_val) if (max_val - min_val) > 0 else 0
        lower = min_val - tol
        upper = max_val + tol

    is_within = lower <= value <= upper

    info = {
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
    return bool(is_within), info


class PassportPhotoValidatorV2:
    """Enhanced passport photo validator with database-driven standards.

    By default (threshold_source=None), validates against the PASSPORT_STANDARDS
    database ranges extracted from standard reference images.

    Optional threshold_source:
      - None / "standard_range"  -> check against standard image min/max ranges (default)
      - "existing"               -> use original v2.0 hardcoded thresholds
      - "standard_min"           -> use minimum value from standard images
      - "standard_avg"           -> use average value from standard images
      - "strict"                 -> use max(existing, standard_min) for each metric
    """

    def __init__(self, image_path, threshold_source=None):
        """Initialize validator with image path.

        Args:
            image_path: Path to the image file to validate
            threshold_source: None for database range checking, or one of the named modes
        """
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

        # Bangladesh passport photo standards (at 300 DPI)
        self.STANDARD_WIDTH = 300
        self.STANDARD_HEIGHT = 300
        self.TOLERANCE = 1.0
        self.MIN_WIDTH = self.STANDARD_WIDTH * (1 - self.TOLERANCE)
        self.MAX_WIDTH = self.STANDARD_WIDTH * (1 + self.TOLERANCE)
        self.MIN_HEIGHT = self.STANDARD_HEIGHT * (1 - self.TOLERANCE)
        self.MAX_HEIGHT = self.STANDARD_HEIGHT * (1 + self.TOLERANCE)

        # Resolve thresholds based on source
        self._resolve_thresholds()

        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _resolve_thresholds(self):
        """Resolve active thresholds."""
        existing = PASSPORT_STANDARDS["existing"]
        computed = PASSPORT_STANDARDS["computed_from_standards"]
        source = self.threshold_source

        if source is None or source == "standard_range":
            # Default: use standard ranges as database - no single threshold,
            # validation will call _is_within_standard_range() directly.
            self.MIN_SHARPNESS = None
            self.LIGHT_BALANCE_THRESHOLD = None
            self.MIN_RIGHT_MARGIN = None
            self.MIN_BOTTOM_MARGIN = None
            self.use_standard_range = True
        elif source == "existing":
            self.MIN_SHARPNESS = existing["MIN_SHARPNESS"]
            self.LIGHT_BALANCE_THRESHOLD = existing["LIGHT_BALANCE_THRESHOLD"]
            self.MIN_RIGHT_MARGIN = existing["MIN_RIGHT_MARGIN"]
            self.MIN_BOTTOM_MARGIN = existing["MIN_BOTTOM_MARGIN"]
            self.use_standard_range = False
        elif source == "standard_min":
            self.MIN_SHARPNESS = round(computed["MIN_SHARPNESS"]["min"], 2)
            self.LIGHT_BALANCE_THRESHOLD = round(computed["LIGHT_BALANCE_THRESHOLD"]["min"], 2)
            self.MIN_RIGHT_MARGIN = round(computed["MIN_RIGHT_MARGIN"]["min"], 3)
            self.MIN_BOTTOM_MARGIN = round(computed["MIN_BOTTOM_MARGIN"]["min"], 3)
            self.use_standard_range = False
        elif source == "standard_avg":
            self.MIN_SHARPNESS = round(computed["MIN_SHARPNESS"]["avg"], 2)
            self.LIGHT_BALANCE_THRESHOLD = round(computed["LIGHT_BALANCE_THRESHOLD"]["avg"], 2)
            self.MIN_RIGHT_MARGIN = round(computed["MIN_RIGHT_MARGIN"]["avg"], 3)
            self.MIN_BOTTOM_MARGIN = round(computed["MIN_BOTTOM_MARGIN"]["avg"], 3)
            self.use_standard_range = False
        elif source == "strict":
            self.MIN_SHARPNESS = round(max(existing["MIN_SHARPNESS"], computed["MIN_SHARPNESS"]["min"]), 2)
            self.LIGHT_BALANCE_THRESHOLD = round(min(existing["LIGHT_BALANCE_THRESHOLD"], computed["LIGHT_BALANCE_THRESHOLD"]["min"]), 2)
            self.MIN_RIGHT_MARGIN = round(max(existing["MIN_RIGHT_MARGIN"], computed["MIN_RIGHT_MARGIN"]["min"]), 3)
            self.MIN_BOTTOM_MARGIN = round(max(existing["MIN_BOTTOM_MARGIN"], computed["MIN_BOTTOM_MARGIN"]["min"]), 3)
            self.use_standard_range = False
        else:
            raise ValueError(f"Unknown threshold_source: {source}. Use None, 'standard_range', 'existing', 'standard_min', 'standard_avg', or 'strict'.")

        logger.info(f"Threshold source: {source}, use_standard_range: {self.use_standard_range}")

    def load_image(self):
        """Load image from file"""
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

    def check_image_dimensions(self):
        """Check if image dimensions match Bangladesh passport size standards"""
        valid = (self.MIN_HEIGHT <= self.height <= self.MAX_HEIGHT and
                 self.MIN_WIDTH <= self.width <= self.MAX_WIDTH)

        aspect_ratio = self.width / self.height
        standard_aspect = self.STANDARD_WIDTH / self.STANDARD_HEIGHT

        self.validation_results['dimensions'] = {
            'valid': bool(valid),
            'width': int(self.width),
            'height': int(self.height),
            'aspect_ratio': round(float(aspect_ratio), 2),
            'standard_aspect_ratio': round(float(standard_aspect), 2),
            'width_range': f"{self.MIN_WIDTH:.0f}-{self.MAX_WIDTH:.0f}px",
            'height_range': f"{self.MIN_HEIGHT:.0f}-{self.MAX_HEIGHT:.0f}px",
            'tolerance': '±100% (very comfortable)'
        }
        return bool(valid)

    def check_photo_sharpness(self):
        """Check if photo sharpness is within standard database range."""
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
            logger.info(f"Sharpness check: {sharpness_score:.2f} -> {'OK' if is_valid else 'FAIL'}")
            return bool(is_valid)
        except Exception as e:
            logger.error(f"Error checking sharpness: {str(e)}")
            self.validation_results['sharpness'] = {'valid': False, 'error': str(e)}
            return False

    def check_face_light_balance(self, face_region=None):
        """Check if face lighting balance is within standard database range."""
        try:
            if face_region is None:
                analysis_region = self.gray
            else:
                (x, y, w, h) = face_region
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

            balanced_lr = lr_diff < 35
            balanced_tb = tb_diff < 35

            if self.use_standard_range:
                # Check if effective_threshold is within standard range
                is_within_range, range_info = _is_within_standard_range('LIGHT_BALANCE_THRESHOLD', effective_threshold, tolerance_pct=0.0)
                # For light balance, being within the standard range means the variance is acceptable
                # The standard range is [41.18, 80.96] - if effective_threshold is within this, it's OK
                is_valid = is_within_range and balanced_lr and balanced_tb
                threshold_used = f"range [{range_info['standard_min']}-{range_info['standard_max']}]"
            else:
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
            logger.info(f"Light Balance - LR Diff: {lr_diff:.2f}, TB Diff: {tb_diff:.2f}, Effective: {effective_threshold:.2f} -> {'OK' if is_valid else 'FAIL'}")
            return bool(is_valid)
        except Exception as e:
            logger.error(f"Error checking light balance: {str(e)}")
            self.validation_results['light_balance'] = {'valid': False, 'error': str(e)}
            return False

    def check_face_brightness(self, face_region=None):
        """Check if face brightness is adequate"""
        try:
            if face_region is None or len(face_region) == 0:
                self.validation_results['face_brightness'] = {
                    'valid': True,
                    'message': 'No face region to analyze'
                }
                return True

            (x, y, w, h) = face_region[0] if isinstance(face_region, (list, tuple)) and len(face_region) > 3 else face_region

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

            is_face_bright_enough = face_brightness >= 80
            is_face_not_overexposed = face_brightness <= 235

            brightness_diff = background_brightness - face_brightness
            is_brightness_balanced = -20 <= brightness_diff <= 65
            is_overall_not_overexposed = overall_brightness <= 220

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
                'constraints': {
                    'min_face_brightness': 80,
                    'max_face_brightness': 235,
                    'max_darker_than_bg': 65,
                    'max_brighter_than_bg': 20,
                    'max_overall_brightness': 220
                },
                'issue_code': 'ISSUE_FACE_BRIGHTNESS_BAD' if not is_valid else None
            }
            logger.info(f"Face Brightness - Face:{face_brightness:.1f} BG:{background_brightness:.1f} Overall:{overall_brightness:.1f} Diff:{brightness_diff:.1f} {'OK' if is_valid else 'FAIL'}")
            return bool(is_valid)
        except Exception as e:
            logger.error(f"Error checking face brightness: {str(e)}")
            self.validation_results['face_brightness'] = {'valid': False, 'error': str(e)}
            return False

    def check_shoulder_visibility(self, faces):
        """Check if shoulders are visible below the face"""
        if len(faces) == 0:
            return False, "No face detected"

        (x, y, w, h) = faces[0]
        face_bottom = y + h
        image_bottom = self.height
        space_below_face = image_bottom - face_bottom
        required_shoulder_space = self.height * 0.15

        has_shoulder_space = space_below_face >= required_shoulder_space
        face_width_ratio = w / self.width
        shoulder_detection_possible = face_width_ratio < 0.8

        is_valid = has_shoulder_space and shoulder_detection_possible
        shoulder_space_percent = (space_below_face / self.height) * 100

        return is_valid, {
            'space_below_face_px': int(space_below_face),
            'required_space_px': int(required_shoulder_space),
            'space_percent': round(float(shoulder_space_percent), 1),
            'face_width_ratio': round(float(face_width_ratio), 2),
            'has_shoulder_space': bool(has_shoulder_space),
            'shoulders_detectable': bool(shoulder_detection_possible)
        }

    def check_right_side_visibility(self, faces):
        """Check if face right side is fully visible."""
        if len(faces) == 0:
            return False, {'valid': False, 'message': 'No face detected', 'issue_code': None}

        (x, y, w, h) = faces[0]
        face_right_edge = x + w
        image_right_edge = self.width
        right_margin = image_right_edge - face_right_edge
        left_margin = x

        if self.use_standard_range:
            # Use standard range for margin checking
            min_margin_ratio = min(left_margin / self.width, right_margin / self.width)
            is_within_range, range_info = _is_within_standard_range('MIN_RIGHT_MARGIN', round(min_margin_ratio, 3), tolerance_pct=0.0)
            required_margin = self.width * range_info['standard_min'] if range_info else self.width * 0.15
            has_right_margin = right_margin >= required_margin
            has_left_margin = left_margin >= required_margin
        else:
            required_margin = self.width * self.MIN_RIGHT_MARGIN
            has_right_margin = right_margin >= required_margin
            has_left_margin = left_margin >= required_margin
            range_info = None

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

        result = {
            'valid': bool(is_valid),
            'face_x': int(x),
            'face_width': int(w),
            'face_right_edge': int(face_right_edge),
            'image_width': int(self.width),
            'left_margin_px': int(left_margin),
            'right_margin_px': int(right_margin),
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
                'Face too small (width ratio < 25%)' if not face_width_adequate else
                'Face not properly centered or margins unbalanced' if not (margins_balanced and is_centered) else
                'Insufficient margin(s) - face too close to edge'
            ),
            'issue_code': 'ISSUE_RIGHT_NOT_FULL' if not is_valid else None,
            'standard_range': range_info
        }
        logger.info(f"Right Side Visibility - L:{left_margin_percent:.1f}% R:{right_margin_percent:.1f}% Balance:{margin_balance_diff:.1f}% FaceW:{face_width_ratio:.3f} Center:{face_center_x:.2f} {'OK' if is_valid else 'FAIL'}")
        return bool(is_valid), result

    def check_left_side_visibility(self, faces):
        """Check if face left side is fully visible."""
        if len(faces) == 0:
            return False, {'valid': False, 'message': 'No face detected', 'issue_code': None}

        (x, y, w, h) = faces[0]
        face_left_edge = x
        image_left_edge = 0
        left_margin = face_left_edge - image_left_edge
        right_margin = self.width - (x + w)

        if self.use_standard_range:
            min_margin_ratio = min(left_margin / self.width, right_margin / self.width)
            is_within_range, range_info = _is_within_standard_range('MIN_RIGHT_MARGIN', round(min_margin_ratio, 3), tolerance_pct=0.0)
            required_margin = self.width * range_info['standard_min'] if range_info else self.width * 0.15
            has_left_margin = left_margin >= required_margin
            has_right_margin = right_margin >= required_margin
        else:
            required_margin = self.width * self.MIN_RIGHT_MARGIN
            has_left_margin = left_margin >= required_margin
            has_right_margin = right_margin >= required_margin
            range_info = None

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

        result = {
            'valid': bool(is_valid),
            'face_x': int(x),
            'face_width': int(w),
            'face_left_edge': int(face_left_edge),
            'image_width': int(self.width),
            'left_margin_px': int(left_margin),
            'right_margin_px': int(right_margin),
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
                'Face too small (width ratio < 25%)' if not face_width_adequate else
                'Face not properly centered or margins unbalanced' if not (margins_balanced and is_centered) else
                'Insufficient left margin - face positioned too far right'
            ),
            'issue_code': 'ISSUE_LEFT_NOT_FULL' if not is_valid else None,
            'standard_range': range_info
        }
        logger.info(f"Left Side Visibility - L:{left_margin_percent:.1f}% R:{right_margin_percent:.1f}% Balance:{margin_balance_diff:.1f}% FaceW:{face_width_ratio:.3f} Center:{face_center_x:.2f} {'OK' if is_valid else 'FAIL'}")
        return bool(is_valid), result

    def check_bottom_face_visibility(self, faces):
        """Check if face bottom is fully visible."""
        if len(faces) == 0:
            return False, {'valid': False, 'message': 'No face detected', 'issue_code': None}

        (x, y, w, h) = faces[0]
        face_bottom_edge = y + h
        image_bottom_edge = self.height
        bottom_margin = image_bottom_edge - face_bottom_edge
        top_margin = y

        if self.use_standard_range:
            min_margin_ratio = min(top_margin / self.height, bottom_margin / self.height)
            is_within_range, range_info = _is_within_standard_range('MIN_BOTTOM_MARGIN', round(min_margin_ratio, 3), tolerance_pct=0.0)
            required_margin = self.height * range_info['standard_min'] if range_info else self.height * 0.15
            has_bottom_margin = bottom_margin >= required_margin
            has_top_margin = top_margin >= (self.height * 0.10)
        else:
            required_margin = self.height * self.MIN_BOTTOM_MARGIN
            has_bottom_margin = bottom_margin >= required_margin
            has_top_margin = top_margin >= (self.height * 0.10)
            range_info = None

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

        result = {
            'valid': bool(is_valid),
            'face_y': int(y),
            'face_height': int(h),
            'face_bottom_edge': int(face_bottom_edge),
            'image_height': int(self.height),
            'top_margin_px': int(top_margin),
            'bottom_margin_px': int(bottom_margin),
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
                'Face too small (height ratio < 20%)' if not face_height_adequate else
                'Face not properly centered or margins unbalanced' if not (margins_balanced and is_vertically_centered) else
                'Insufficient margin(s) - face top/bottom too close to edge'
            ),
            'issue_code': 'ISSUE_BOTTOM_NOT_FULL' if not is_valid else None,
            'standard_range': range_info
        }
        logger.info(f"Bottom Visibility - T:{top_margin_percent:.1f}% B:{bottom_margin_percent:.1f}% Balance:{margin_balance_diff:.1f}% FaceH:{face_height_ratio:.3f} Center:{face_center_y:.2f} {'OK' if is_valid else 'FAIL'}")
        return bool(is_valid), result

    def check_selfie_detection(self, faces):
        """Detect if the image is a selfie (face too close/large)"""
        if len(faces) == 0:
            return False, "No face detected"

        (x, y, w, h) = faces[0]
        face_area_ratio = (w * h) / (self.width * self.height)
        is_selfie = face_area_ratio > 0.40

        if is_selfie:
            return True, f"Selfie detected - face occupies {face_area_ratio*100:.1f}% of frame (too close)"

        return False, f"Face size normal - {face_area_ratio*100:.1f}% of frame"

    def detect_faces(self):
        """Detect faces in the image"""
        faces = self.face_cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        face_count = len(faces)
        is_valid = face_count == 1

        is_selfie, selfie_msg = self.check_selfie_detection(faces)
        has_shoulders, shoulder_info = self.check_shoulder_visibility(faces)
        has_left_visibility, left_visibility_info = self.check_left_side_visibility(faces)
        has_right_visibility, right_visibility_info = self.check_right_side_visibility(faces)
        has_bottom_visibility, bottom_visibility_info = self.check_bottom_face_visibility(faces)

        self.validation_results['face_detection'] = {
            'valid': bool(is_valid and not is_selfie),
            'face_count': int(face_count),
            'required_faces': 1,
            'selfie_detected': bool(is_selfie),
            'selfie_note': selfie_msg,
            'shoulder_detection': shoulder_info,
            'left_side_detection': left_visibility_info,
            'right_side_detection': right_visibility_info,
            'bottom_detection': bottom_visibility_info
        }

        return faces, bool(is_valid and not is_selfie and has_shoulders and has_left_visibility and has_right_visibility and has_bottom_visibility)

    def check_face_position(self, faces):
        """Check if face is properly centered in the frame with shoulders visible"""
        if len(faces) == 0:
            self.validation_results['face_position'] = {'valid': False, 'message': 'No face detected'}
            return False

        (x, y, w, h) = faces[0]
        face_center_x = (x + w/2) / self.width
        face_center_y = (y + h/2) / self.height

        center_tolerance = 0.25
        valid_x = (0.5 - center_tolerance) <= face_center_x <= (0.5 + center_tolerance)
        valid_y = (0.25) <= face_center_y <= (0.55)

        is_valid = valid_x and valid_y
        face_area_ratio = (w * h) / (self.width * self.height)

        self.validation_results['face_position'] = {
            'valid': bool(is_valid),
            'center_x_ratio': round(float(face_center_x), 2),
            'center_y_ratio': round(float(face_center_y), 2),
            'face_area_ratio': round(float(face_area_ratio), 2),
            'tolerance': '±25% horizontal, 25-55% vertical (with shoulder space)',
            'message': 'Face properly positioned with shoulder space' if is_valid else 'Face not properly positioned'
        }

        return bool(is_valid)

    def check_lighting(self):
        """Check if image has proper lighting"""
        avg_brightness = cv2.mean(self.gray)[0]
        mean = avg_brightness
        variance = np.mean((self.gray - mean) ** 2)
        std_dev = np.sqrt(variance)

        brightness_valid = 70 <= avg_brightness <= 230
        lighting_even = std_dev < 80

        is_valid = brightness_valid and lighting_even

        self.validation_results['lighting'] = {
            'valid': bool(is_valid),
            'avg_brightness': round(float(avg_brightness), 2),
            'brightness_range': '70-230 (comfortable)',
            'std_deviation': round(float(std_dev), 2),
            'lighting_uniformity': 'even' if lighting_even else 'uneven',
            'threshold': 'std_dev < 80'
        }

        return bool(is_valid)

    def check_background_color(self):
        """Check if background is standard passport color: white or blue"""
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
                        'message': 'Transparent background - acceptable for passport'
                    }
                    return True

            border_size = 20
            top = self.image[:border_size, :, :]
            bottom = self.image[-border_size:, :, :]
            left = self.image[:, :border_size, :]
            right = self.image[:, -border_size:, :]

            border_pixels = np.concatenate([
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3)
            ])

            mean_color = np.mean(border_pixels, axis=0)
            b_mean = mean_color[0]
            g_mean = mean_color[1]
            r_mean = mean_color[2]

            brightness = (b_mean + g_mean + r_mean) / 3.0

            color_variance = max(abs(r_mean-g_mean), abs(g_mean-b_mean), abs(r_mean-b_mean))
            blue_dominance = b_mean - r_mean

            background_type = 'unknown'
            is_valid_bg = False

            if r_mean > 180 and g_mean > 180 and b_mean > 180 and color_variance < 40:
                background_type = 'white'
                is_valid_bg = True
            elif r_mean > 180 and g_mean > 180 and b_mean > 180 and color_variance < 60:
                background_type = 'light/off-white'
                is_valid_bg = True
            elif (r_mean > 100 and g_mean > 100 and b_mean > 140 and blue_dominance > 40):
                background_type = 'blue'
                is_valid_bg = True
            elif (b_mean > 80 and blue_dominance > 20 and brightness > 120):
                background_type = 'blue'
                is_valid_bg = True
            elif (b_mean > 80 and blue_dominance > 30 and brightness > 80):
                background_type = 'blue'
                is_valid_bg = True
            elif (color_variance < 15 and brightness > 100):
                background_type = 'grey'
                is_valid_bg = True
            elif (color_variance < 20 and brightness > 140):
                background_type = 'light grey'
                is_valid_bg = True
            elif brightness > 150:
                background_type = 'light/off-white'
                is_valid_bg = True
            else:
                background_type = 'other'
                is_valid_bg = False

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
                'message': f'Accepted - {background_type} background' if is_valid_bg else f'Not accepted - {background_type} background'
            }

            return bool(is_valid_bg)
        except Exception as e:
            self.validation_results['background'] = {
                'valid': False,
                'error': str(e),
                'background_type': 'unknown'
            }
            return False

    def validate(self):
        """Run all validation checks"""
        if not self.load_image():
            self.validation_results['overall'] = {'valid': False, 'message': 'Failed to load image'}
            return False

        faces, faces_valid = self.detect_faces()

        checks = [
            ('dimensions', self.check_image_dimensions()),
            ('sharpness', self.check_photo_sharpness()),
            ('faces', faces_valid),
            ('face_position', self.check_face_position(faces)),
            ('lighting', self.check_lighting()),
            ('light_balance', self.check_face_light_balance(faces[0] if len(faces) > 0 else None)),
            ('face_brightness', self.check_face_brightness(faces[0] if len(faces) > 0 else None)),
            ('background', self.check_background_color())
        ]

        all_valid = all(result for _, result in checks)

        issues_summary = {
            'has_issues': len(self.validation_issues) > 0,
            'issue_count': len(self.validation_issues),
            'issues': self.validation_issues
        }

        # Build standards reference summary for the response
        standards_reference = {
            'threshold_source': self.threshold_source,
            'use_standard_range': self.use_standard_range,
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
            'version': 'v2.2',
            'issues': issues_summary,
            'standards_reference': standards_reference
        }

        return bool(all_valid)
