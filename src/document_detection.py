"""Phase 3: find the page in a photo and flatten it (perspective correction).

Idea: threshold or edge-detect the photo, take the biggest external contour,
approximate it with a polygon, and if it has 4 corners warp it to a rectangle.
"""
import cv2
import numpy as np


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()          # y - x
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_from_mask(mask, min_area_ratio=0.20, max_area_ratio=0.98):
    """Largest 4-corner convex outline in a binary mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = mask.shape[0] * mask.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area:
            break
        if area > max_area_ratio * img_area:
            continue                           # that is the image border, not a page
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.04, 0.05):   # try progressively coarser polygons
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def find_document_corners(img, work_size=600):
    """Return the 4 page corners (tl, tr, br, bl) in original pixel coordinates, or None."""
    h, w = img.shape[:2]
    scale = min(1.0, work_size / max(h, w))
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    masks = []
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, close_k))          # bright page
    masks.append(cv2.morphologyEx(255 - otsu, cv2.MORPH_CLOSE, close_k))    # dark page
    edges = cv2.Canny(blur, 50, 150)
    masks.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k, iterations=2))

    for m in masks:
        quad = _quad_from_mask(m)
        if quad is not None:
            return order_points(quad / scale)
    return None


def four_point_warp(img, corners):
    """Warp the quadrilateral given by `corners` to an upright rectangle."""
    tl, tr, br, bl = order_points(corners)
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 50 or height < 50:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC)


def detect_and_warp(img):
    """Return (flattened_image, corners). If no page is found: (img, None)."""
    corners = find_document_corners(img)
    if corners is None:
        return img, None
    warped = four_point_warp(img, corners)
    if warped is None:
        return img, None
    return warped, corners
