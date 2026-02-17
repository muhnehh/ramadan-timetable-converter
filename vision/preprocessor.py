"""
Vision Preprocessing Module (v2 — Smart Multi-Strategy)
Produces MULTIPLE preprocessed variants from a single input image
so the OCR engine can pick the best result.

Handles:
  - Phone photos (blurry, skewed, glare, shadows)
  - Screenshots (colored blocks, dark/light themes)
  - Scanned PDFs
  - Colored timetable grids
"""

import os
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import structlog

logger = structlog.get_logger()


class ImagePreprocessor:
    """
    Produces multiple preprocessed variants of each input image.
    The OCR extractor tries all variants and keeps the best extraction.
    """

    def process(self, file_path: str) -> List[np.ndarray]:
        """
        Main entry point. Returns list of preprocessed image variants.
        For a single-page image this returns 4-6 variants.
        For a PDF each page gets multiple variants.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            raw_images = self._pdf_to_images(file_path)
        else:
            img = cv2.imread(file_path)
            if img is None:
                # Fallback: PIL handles more formats (HEIC photos, etc.)
                try:
                    pil_img = Image.open(file_path).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    raise ValueError(f"Could not read image: {file_path}")
            raw_images = [img]

        all_variants = []
        for i, img in enumerate(raw_images):
            logger.info("preprocessing_page", page=i + 1,
                        size=f"{img.shape[1]}x{img.shape[0]}")
            variants = self._generate_variants(img)
            all_variants.extend(variants)
            logger.info("variants_generated", page=i + 1, count=len(variants))

        return all_variants

    # ------------------------------------------------------------------
    # PDF handling
    # ------------------------------------------------------------------
    def _pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF pages to OpenCV images."""
        try:
            from pdf2image import convert_from_path
            pil_images = convert_from_path(pdf_path, dpi=300)
            return [
                cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
                for p in pil_images
            ]
        except Exception as e:
            logger.warning("pdf2image_failed", error=str(e))
            raise ValueError(
                f"Could not convert PDF. Install poppler. Error: {e}"
            )

    # ------------------------------------------------------------------
    # Multi-variant generation — the key to robust OCR
    # ------------------------------------------------------------------
    def _generate_variants(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessed versions of the same image.
        Each variant targets a different kind of timetable input.
        """
        variants: List[np.ndarray] = []

        # Resize to workable size
        img = self._limit_size(img, max_dim=3000)

        # ----- V0: Raw color image (Tesseract handles color OK) -----
        variants.append(img.copy())

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ----- V1: Deskewed + CLAHE enhanced grayscale -----
        deskewed = self._deskew(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(deskewed)
        variants.append(enhanced)

        # ----- V2: Sharpen + Otsu threshold -----
        sharpened = self._sharpen(enhanced)
        _, otsu = cv2.threshold(sharpened, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        # ----- V3: Adaptive threshold (handles uneven lighting/photos) -----
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        variants.append(adaptive)

        # ----- V4: Color-aware extraction (colored timetable blocks) -----
        color_cleaned = self._color_aware_extraction(img)
        if color_cleaned is not None:
            variants.append(color_cleaned)

        # ----- V5: Inverted for dark backgrounds -----
        mean_val = np.mean(gray)
        if mean_val < 128:
            inverted = cv2.bitwise_not(enhanced)
            variants.append(inverted)

        # ----- V6: Upscaled + denoised (small or blurry text) -----
        h_img, w_img = gray.shape[:2]
        if max(h_img, w_img) < 1500:
            upscaled = cv2.resize(enhanced, None, fx=2, fy=2,
                                  interpolation=cv2.INTER_CUBIC)
            denoised = cv2.fastNlMeansDenoising(upscaled, h=10)
            variants.append(denoised)

        # ----- V7: Morphological cleaning (close gaps in text) -----
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, morph_kernel)
        variants.append(morphed)

        return variants

    # ------------------------------------------------------------------
    # Color-aware extraction (handles colored timetable blocks)
    # ------------------------------------------------------------------
    def _color_aware_extraction(self, img: np.ndarray) -> np.ndarray:
        """
        University timetables use colored blocks for classes.
        This normalises colors so text is readable by OCR.
        """
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, s, _ = cv2.split(hsv)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Check if image actually has significant color
            color_pct = np.mean(s > 40)
            if color_pct < 0.05:
                return None  # Not a colored timetable

            # Convert to Lab color space for better text/bg separation
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, _, _ = cv2.split(lab)

            # CLAHE on luminance
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)

            # Adaptive threshold on enhanced luminance
            result = cv2.adaptiveThreshold(
                enhanced_l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 8
            )
            return result
        except Exception as e:
            logger.warning("color_aware_failed", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Sharpen
    # ------------------------------------------------------------------
    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Unsharp masking for crisper text edges."""
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _limit_size(self, img: np.ndarray, max_dim: int = 3000) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Correct rotation using Hough line detection."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=80, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 30:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3 or abs(median_angle) > 20:
            return gray

        logger.info("deskew", angle=round(median_angle, 2))
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
