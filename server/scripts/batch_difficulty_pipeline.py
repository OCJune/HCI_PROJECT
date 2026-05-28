import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from coloringbook_utils import (  # noqa: E402
    assign_region_color_numbers,
    clean_edges,
    edge_density,
    estimate_background_color,
    kmeans_quantization_with_labels,
    region_metrics,
    save_image_rgb,
    segment_connected_components,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DIFFICULTY_MARGINS = {"easy": 0, "normal": 10, "hard": 20}

MIN_K_LOWER_BOUND = 3
MIN_K_UPPER_BOUND = 40
K_UPPER_BOUND = 60

PRE_KMEANS_MEAN_SHIFT_ENABLED = True
MEAN_SHIFT_SPATIAL_RADIUS = 12
MEAN_SHIFT_COLOR_RADIUS = 28
SIMPLIFY_DIAMETER = 9
SIMPLIFY_SIGMA_COLOR = 90
SIMPLIFY_SIGMA_SPACE = 90

SEGMENTATION_OUTPUT_SCALE = 3.0
OBJECT_MIN_AREA = 260
OBJECT_CLOSE_KERNEL = 5
SEGMENTATION_CONNECT_KERNEL = 3
SEGMENTATION_CONNECT_ITER = 1
SEGMENTATION_EDGE_THICKNESS = 2
SEGMENTATION_LINE_COLOR = (0, 0, 0)
DETAIL_LINE_GRAY = 150
MIN_REGION_AREA = 60
MIN_COLORABLE_REGION_AREA = 1
DETAIL_RENDER_MIN_AREA = 2
DETAIL_RENDER_MIN_ARC_LENGTH = 6
DETAIL_RENDER_MIN_POINTS = 3
NUMBER_TEXT_GRAY = 130
NUMBER_FONT_MAX_SCALE = 0.90
NUMBER_FONT_MIN_SCALE = 0.10
NUMBER_FONT_PADDING = 1
NUMBER_FONT_SCALE_MULTIPLIER = 1.25
LABEL_ISLAND_CLEANUP_ENABLED = True
LABEL_ISLAND_MIN_AREA = 36
LABEL_ISLAND_FORCE_MERGE_AREA = 10
LABEL_ISLAND_MAX_COLOR_DISTANCE = 85.0

DETAIL_CANNY_LOW = 90
DETAIL_CANNY_HIGH = 200
DETAIL_MIN_AREA = 3
DETAIL_MAX_AREA = 1200
DARK_DETAIL_THRESHOLD = 70
DARK_DETAIL_MIN_AREA = 5
DARK_DETAIL_MAX_AREA = 900
SEGMENTATION_DETAIL_CANNY_LOW = 35
SEGMENTATION_DETAIL_CANNY_HIGH = 95
SEGMENTATION_DETAIL_MIN_PIXELS = 10
SEGMENTATION_DETAIL_MAX_PIXELS = 2600
SEGMENTATION_DETAIL_MIN_SPAN = 28
SEGMENTATION_DETAIL_MAX_SPAN = 900
SEGMENTATION_DETAIL_NON_DARK_THRESHOLD = 82
SEGMENTATION_DARK_REGION_MIN_AREA = 6
SEGMENTATION_DARK_REGION_MAX_AREA = 2400
SEGMENTATION_DARK_REGION_MAX_ASPECT = 8.0
SEGMENTATION_DARK_REGION_MIN_EXTENT = 0.08
SEGMENTATION_CLOSED_DETAIL_LOW = 45
SEGMENTATION_CLOSED_DETAIL_HIGH = 130
SEGMENTATION_CLOSED_DETAIL_MIN_PIXELS = 8
SEGMENTATION_CLOSED_DETAIL_MAX_PIXELS = 1800
SEGMENTATION_CLOSED_DETAIL_MIN_CONTOUR_AREA = 18
SEGMENTATION_CLOSED_DETAIL_MAX_CONTOUR_AREA = 9000
SEGMENTATION_CLOSED_DETAIL_MAX_SPAN = 220
SEGMENTATION_CLOSED_DETAIL_MIN_EXTENT = 0.06
SEGMENTATION_CLOSED_DETAIL_CLOSE_KERNEL = 5

AUTO_MIN_WORKING_LONG_EDGE = 1100
AUTO_DEFAULT_WORKING_LONG_EDGE = 1800
AUTO_MAX_WORKING_LONG_EDGE = 2400
AUTO_HIGH_DETAIL_EDGE_DENSITY = 0.075
AUTO_LOW_DETAIL_EDGE_DENSITY = 0.028
AUTO_BLUR_VARIANCE_THRESHOLD = 85.0
AUTO_SIMPLE_COLOR_LAB_SPREAD = 45.0
AUTO_SIMPLE_EDGE_DENSITY = 0.060
AUTO_SIMPLE_K_UPPER_BOUND = 18
AUTO_SIMPLE_BACKGROUND_MERGE_DISTANCE = 42.0


def resize_to_long_edge(image, target_long_edge):
    if target_long_edge is None or target_long_edge <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge == int(target_long_edge):
        return image, 1.0
    scale = float(target_long_edge) / float(long_edge)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=interpolation,
    )
    return resized, scale


def analyze_image_profile(image, max_sample_long_edge=900):
    # Stage 1: measure input complexity on a bounded sample so policy selection
    # stays fast even when the uploaded image is very large.
    h, w = image.shape[:2]
    long_edge = max(h, w)
    sample, sample_scale = resize_to_long_edge(
        image,
        min(long_edge, max_sample_long_edge),
    )
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
    edges = cv2.Canny(gray, 70, 160)
    for channel in cv2.split(lab):
        edges = cv2.bitwise_or(edges, cv2.Canny(channel, 70, 160))

    edge_density_value = float(np.count_nonzero(edges)) / float(edges.size)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab_pixels = lab.reshape(-1, 3).astype(np.float32)
    if len(lab_pixels) > 30000:
        rng = np.random.default_rng(11)
        lab_pixels = lab_pixels[rng.choice(len(lab_pixels), 30000, replace=False)]
    lab_spread = float(np.mean(np.linalg.norm(lab_pixels - lab_pixels.mean(axis=0), axis=1)))
    coarse_lab = (lab.reshape(-1, 3).astype(np.uint16) // 16).astype(np.uint16)
    coarse_lab_colors = int(len(np.unique(coarse_lab, axis=0)))

    return {
        "original_width": int(w),
        "original_height": int(h),
        "original_long_edge": int(long_edge),
        "sample_scale": float(sample_scale),
        "edge_density": edge_density_value,
        "blur_variance": blur_variance,
        "lab_spread": lab_spread,
        "coarse_lab_colors": coarse_lab_colors,
    }


def choose_processing_policy(profile, requested_max_size=None):
    # Stage 2: convert image measurements into service-safe processing knobs.
    # The pipeline should adapt by image type instead of relying on filename rules.
    original_long_edge = int(profile["original_long_edge"])
    edge_density_value = float(profile["edge_density"])
    blur_variance = float(profile["blur_variance"])
    is_small_source = original_long_edge < AUTO_MIN_WORKING_LONG_EDGE
    is_high_detail = edge_density_value >= AUTO_HIGH_DETAIL_EDGE_DENSITY
    is_low_detail = edge_density_value <= AUTO_LOW_DETAIL_EDGE_DENSITY
    is_blurry = blur_variance < AUTO_BLUR_VARIANCE_THRESHOLD
    is_simple_color_art = (
        float(profile["lab_spread"]) <= AUTO_SIMPLE_COLOR_LAB_SPREAD
        and edge_density_value <= AUTO_SIMPLE_EDGE_DENSITY
    )

    if requested_max_size is not None and requested_max_size > 0:
        # Explicit max size is treated as an operational upper bound.
        target_long_edge = min(original_long_edge, int(requested_max_size))
    elif is_small_source:
        target_long_edge = AUTO_MIN_WORKING_LONG_EDGE
    elif is_high_detail:
        target_long_edge = min(original_long_edge, AUTO_MAX_WORKING_LONG_EDGE)
    elif is_low_detail or is_blurry:
        target_long_edge = min(original_long_edge, AUTO_DEFAULT_WORKING_LONG_EDGE)
    else:
        target_long_edge = min(original_long_edge, AUTO_MAX_WORKING_LONG_EDGE)

    if target_long_edge < AUTO_MIN_WORKING_LONG_EDGE and is_small_source:
        target_long_edge = AUTO_MIN_WORKING_LONG_EDGE

    detail_boost = "high" if (is_small_source or is_high_detail) else "normal"
    if is_simple_color_art:
        # Simple flat-color art is vulnerable to over-segmentation, so keep
        # detail promotion and color-count growth conservative.
        detail_boost = "simple"
    if is_blurry and not is_high_detail:
        detail_boost = "soft"

    source_low = SEGMENTATION_DETAIL_CANNY_LOW
    source_high = SEGMENTATION_DETAIL_CANNY_HIGH
    closed_low = SEGMENTATION_CLOSED_DETAIL_LOW
    closed_high = SEGMENTATION_CLOSED_DETAIL_HIGH
    connect_kernel = SEGMENTATION_CONNECT_KERNEL
    connect_iter = SEGMENTATION_CONNECT_ITER

    if detail_boost == "high":
        source_low = 28
        source_high = 82
        closed_low = 38
        closed_high = 112
        connect_kernel = 5
        connect_iter = 1
    elif detail_boost == "simple":
        source_low = 46
        source_high = 130
        closed_low = 58
        closed_high = 155
        connect_kernel = 5
        connect_iter = 1
    elif detail_boost == "soft":
        source_low = 42
        source_high = 115
        closed_low = 52
        closed_high = 145
        connect_kernel = 5
        connect_iter = 1

    return {
        "target_long_edge": int(target_long_edge),
        "detail_boost": detail_boost,
        "source_detail_low": int(source_low),
        "source_detail_high": int(source_high),
        "closed_detail_low": int(closed_low),
        "closed_detail_high": int(closed_high),
        "connect_kernel": int(connect_kernel),
        "connect_iter": int(connect_iter),
        "simple_color_art": bool(is_simple_color_art),
        "k_upper_bound": AUTO_SIMPLE_K_UPPER_BOUND if is_simple_color_art else K_UPPER_BOUND,
        "difficulty_margins": {"easy": 0, "normal": 4, "hard": 8}
        if is_simple_color_art
        else DIFFICULTY_MARGINS,
    }


def load_image_preserve_size(path, max_size=None, auto_policy=True):
    # Stage 3: load the upload, choose a policy, then resize to the working
    # resolution used for quantization and segmentation.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found or unsupported: {path}")
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    profile = analyze_image_profile(original)
    if auto_policy:
        policy = choose_processing_policy(profile)
        if max_size is not None:
            policy["target_long_edge"] = min(int(policy["target_long_edge"]), int(max_size))
            if profile["original_long_edge"] < AUTO_MIN_WORKING_LONG_EDGE:
                policy["target_long_edge"] = max(
                    int(policy["target_long_edge"]),
                    AUTO_MIN_WORKING_LONG_EDGE,
                )
    else:
        target_long_edge = profile["original_long_edge"]
        if max_size is not None:
            target_long_edge = min(target_long_edge, int(max_size))
        policy = {
            "target_long_edge": int(target_long_edge),
            "detail_boost": "fixed",
            "source_detail_low": SEGMENTATION_DETAIL_CANNY_LOW,
            "source_detail_high": SEGMENTATION_DETAIL_CANNY_HIGH,
            "closed_detail_low": SEGMENTATION_CLOSED_DETAIL_LOW,
            "closed_detail_high": SEGMENTATION_CLOSED_DETAIL_HIGH,
            "connect_kernel": SEGMENTATION_CONNECT_KERNEL,
            "connect_iter": SEGMENTATION_CONNECT_ITER,
            "simple_color_art": False,
            "k_upper_bound": K_UPPER_BOUND,
            "difficulty_margins": DIFFICULTY_MARGINS,
        }
    image, scale = resize_to_long_edge(original, policy["target_long_edge"])
    profile["working_width"] = int(image.shape[1])
    profile["working_height"] = int(image.shape[0])
    profile["working_scale"] = float(scale)
    return image, profile, policy


def smooth_before_kmeans(image):
    # Stage 4: remove small color texture before K-Means so color regions follow
    # meaningful surfaces instead of JPEG noise or anti-aliased pixels.
    if PRE_KMEANS_MEAN_SHIFT_ENABLED:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        shifted = cv2.pyrMeanShiftFiltering(
            bgr,
            sp=MEAN_SHIFT_SPATIAL_RADIUS,
            sr=MEAN_SHIFT_COLOR_RADIUS,
        )
        image = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
    return cv2.bilateralFilter(
        image,
        SIMPLIFY_DIAMETER,
        SIMPLIFY_SIGMA_COLOR,
        SIMPLIFY_SIGMA_SPACE,
    )


def estimate_min_k_from_smoothed_colors(image, max_sample=40000, rgb_bin_size=16):
    # Stage 5: estimate a per-image minimum palette size from coarse color
    # variety and Lab-space spread.
    pixels = image.reshape(-1, 3)
    if len(pixels) > max_sample:
        rng = np.random.default_rng(7)
        pixels = pixels[rng.choice(len(pixels), max_sample, replace=False)]

    coarse_rgb = (pixels.astype(np.uint16) // rgb_bin_size).astype(np.uint16)
    coarse_unique_colors = int(len(np.unique(coarse_rgb, axis=0)))
    lab = cv2.cvtColor(
        pixels.reshape(1, -1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)[0]
    lab_spread = float(np.mean(np.linalg.norm(lab - lab.mean(axis=0), axis=1)))
    estimated = int(np.ceil(0.9 * np.log2(max(coarse_unique_colors, 2)) + lab_spread / 14.0))
    min_k = int(np.clip(estimated, MIN_K_LOWER_BOUND, MIN_K_UPPER_BOUND))
    return min_k, {
        "coarse_unique_colors": coarse_unique_colors,
        "lab_spread": lab_spread,
        "estimated_min_k": min_k,
    }


def difficulty_k_values(min_k, policy=None):
    # Stage 6: expand the palette size by difficulty. Simple artwork uses a
    # smaller growth curve so hard mode adds structure, not background texture.
    policy = policy or {}
    margins = policy.get("difficulty_margins", DIFFICULTY_MARGINS)
    upper_bound = int(policy.get("k_upper_bound", K_UPPER_BOUND))
    return {
        name: int(np.clip(min_k + margin, MIN_K_LOWER_BOUND, upper_bound))
        for name, margin in margins.items()
    }


def upscale_for_segmentation_output(image, quantized, labels, palette, scale):
    if scale <= 1.0:
        return image, quantized, labels
    h, w = labels.shape[:2]
    size = (int(round(w * scale)), int(round(h * scale)))
    up_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    up_labels = cv2.resize(labels.astype(np.int32), size, interpolation=cv2.INTER_NEAREST).astype(np.int32)
    up_quantized = palette[up_labels]
    return up_image, up_quantized, up_labels


def merge_background_similar_labels(labels, palette, image, distance_threshold):
    # Stage 7: for flat artwork, merge near-background palette labels before
    # edge extraction so faint background noise does not become colorable areas.
    background_color = estimate_background_color(image).astype(np.float32)
    palette_float = palette.astype(np.float32)
    distances = np.linalg.norm(palette_float - background_color, axis=1)
    background_label = int(np.argmin(distances))
    merge_labels = np.where(distances <= float(distance_threshold))[0]
    if merge_labels.size == 0:
        return labels, palette

    merged_labels = labels.copy()
    for label_id in merge_labels:
        merged_labels[labels == int(label_id)] = background_label

    merged_palette = palette.copy()
    merged_palette[background_label] = np.clip(background_color, 0, 255).astype(np.uint8)
    return merged_labels, merged_palette


def absorb_tiny_label_islands(labels, palette):
    # Remove tiny isolated K-Means label islands before edge extraction. This
    # prevents a few pixels of quantization noise from becoming colorable regions.
    if not LABEL_ISLAND_CLEANUP_ENABLED:
        return labels

    cleaned = labels.copy().astype(np.int32)
    palette_float = palette.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    for label_id in np.unique(labels):
        source_mask = (cleaned == int(label_id)).astype(np.uint8)
        n_components, components, stats, _ = cv2.connectedComponentsWithStats(source_mask, 8)
        for component_id in range(1, n_components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= LABEL_ISLAND_MIN_AREA:
                continue

            component = components == component_id
            border = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
            neighbor_labels = cleaned[border]
            neighbor_labels = neighbor_labels[neighbor_labels != int(label_id)]
            if neighbor_labels.size == 0:
                continue

            counts = np.bincount(neighbor_labels.astype(np.int32), minlength=len(palette))
            candidates = []
            for neighbor_id in np.flatnonzero(counts):
                if neighbor_id >= len(palette):
                    continue
                distance = float(np.linalg.norm(palette_float[int(label_id)] - palette_float[int(neighbor_id)]))
                if area <= LABEL_ISLAND_FORCE_MERGE_AREA or distance <= LABEL_ISLAND_MAX_COLOR_DISTANCE:
                    candidates.append((int(counts[neighbor_id]), -distance, int(neighbor_id)))

            if candidates:
                _, _, target_label = max(candidates)
                cleaned[component] = target_label

    return cleaned


def object_first_edges(label_map, min_area, close_kernel=5):
    # Stage 8: derive the primary color-region borders from quantized labels.
    # These black lines define where the user can color without crossing regions.
    labels = np.asarray(label_map, dtype=np.int32)
    edge = np.zeros(labels.shape, dtype=np.uint8)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    for label_id in np.unique(labels):
        mask = (labels == label_id).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(edge, [contour], -1, 255, 1, cv2.LINE_8)
    return edge


def connect_segmentation_edges(edges, kernel_size=3, iterations=1):
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def detail_expression_edges(
    image,
    object_edges,
    low,
    high,
    min_area,
    max_area,
    dark_threshold,
    dark_min_area,
    dark_max_area,
):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detail = cv2.Canny(gray, low, high)
    detail = cv2.bitwise_and(detail, cv2.bitwise_not(object_edges))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(detail, 8)
    filtered_edges = np.zeros_like(detail)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            filtered_edges[labels == i] = 255

    dark = (gray < dark_threshold).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    filtered_dark = np.zeros_like(dark)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if dark_min_area <= area <= dark_max_area:
            filtered_dark[labels == i] = 255
    return cv2.bitwise_or(filtered_edges, filtered_dark)


def source_detail_boundary_edges(
    image,
    object_edges,
    low=SEGMENTATION_DETAIL_CANNY_LOW,
    high=SEGMENTATION_DETAIL_CANNY_HIGH,
    min_pixels=SEGMENTATION_DETAIL_MIN_PIXELS,
    max_pixels=SEGMENTATION_DETAIL_MAX_PIXELS,
    min_span=SEGMENTATION_DETAIL_MIN_SPAN,
    max_span=SEGMENTATION_DETAIL_MAX_SPAN,
    non_dark_threshold=SEGMENTATION_DETAIL_NON_DARK_THRESHOLD,
):
    # Stage 9: promote important source-image edges that K-Means can miss,
    # such as facial marks, small object boundaries, and high-contrast details.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    edges = cv2.Canny(gray, low, high)
    for channel in cv2.split(lab):
        edges = cv2.bitwise_or(edges, cv2.Canny(channel, low, high))

    non_dark = (gray > non_dark_threshold).astype(np.uint8) * 255
    edges = cv2.bitwise_and(edges, non_dark)
    object_guard = cv2.dilate(object_edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(object_guard))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    promoted = np.zeros_like(edges)
    for component_id in range(1, n_labels):
        pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        if not (min_pixels <= pixels <= max_pixels):
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        span = max(w, h)
        if not (min_span <= span <= max_span):
            continue

        component = (labels[y:y + h, x:x + w] == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.arcLength(contour, False) < min_span:
                continue
            contour = contour + np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(promoted, [contour], -1, 255, 1, cv2.LINE_8)

    return promoted


def dark_detail_region_edges(
    image,
    dark_threshold=DARK_DETAIL_THRESHOLD,
    min_area=SEGMENTATION_DARK_REGION_MIN_AREA,
    max_area=SEGMENTATION_DARK_REGION_MAX_AREA,
    max_aspect=SEGMENTATION_DARK_REGION_MAX_ASPECT,
    min_extent=SEGMENTATION_DARK_REGION_MIN_EXTENT,
):
    # Stage 10: capture compact dark regions, for example pupils, tiny eyes,
    # mouth marks, or other features that should become closed coloring limits.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark = (gray < dark_threshold).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    promoted = np.zeros_like(dark)

    for component_id in range(1, n_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(1, min(w, h))
        extent = area / max(1, w * h)
        if aspect > max_aspect or extent < min_extent:
            continue

        component = (labels[y:y + h, x:x + w] == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 2:
                continue
            contour = contour + np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(promoted, [contour], -1, 255, 1, cv2.LINE_8)

    return promoted


def closed_detail_shape_edges(
    image,
    object_edges,
    low=SEGMENTATION_CLOSED_DETAIL_LOW,
    high=SEGMENTATION_CLOSED_DETAIL_HIGH,
    min_pixels=SEGMENTATION_CLOSED_DETAIL_MIN_PIXELS,
    max_pixels=SEGMENTATION_CLOSED_DETAIL_MAX_PIXELS,
    min_contour_area=SEGMENTATION_CLOSED_DETAIL_MIN_CONTOUR_AREA,
    max_contour_area=SEGMENTATION_CLOSED_DETAIL_MAX_CONTOUR_AREA,
    max_span=SEGMENTATION_CLOSED_DETAIL_MAX_SPAN,
    min_extent=SEGMENTATION_CLOSED_DETAIL_MIN_EXTENT,
    close_kernel=SEGMENTATION_CLOSED_DETAIL_CLOSE_KERNEL,
):
    # Stage 11: close near-complete detail contours before promotion so small
    # shapes are less likely to remain as broken, uncolorable strokes.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    edges = cv2.Canny(gray, low, high)
    for channel in cv2.split(lab):
        edges = cv2.bitwise_or(edges, cv2.Canny(channel, low, high))

    object_guard = cv2.dilate(object_edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(object_guard))

    kernel_size = max(3, int(close_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_edges, 8)
    promoted = np.zeros_like(edges)
    for component_id in range(1, n_labels):
        pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        if not (min_pixels <= pixels <= max_pixels):
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if max(w, h) > max_span:
            continue
        extent = pixels / max(1, w * h)
        if extent < min_extent:
            continue

        component = (labels[y:y + h, x:x + w] == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if not (min_contour_area <= contour_area <= max_contour_area):
                continue
            contour = contour + np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(promoted, [contour], -1, 255, 1, cv2.LINE_8)

    return promoted


def border_connected_region_ids(region_map):
    border_ids = np.concatenate([
        region_map[0, :],
        region_map[-1, :],
        region_map[:, 0],
        region_map[:, -1],
    ])
    return set(int(v) for v in np.unique(border_ids) if int(v) > 0)


def best_label_point(region_map, region_id, bbox):
    x, y, w, h = bbox
    component = (region_map[y:y + h, x:x + w] == region_id).astype(np.uint8)
    if component.size == 0 or np.count_nonzero(component) == 0:
        return x + w / 2, y + h / 2
    dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return x + max_loc[0], y + max_loc[1]


def fit_number_font_scale(
    text,
    bbox,
    max_scale=NUMBER_FONT_MAX_SCALE,
    min_scale=NUMBER_FONT_MIN_SCALE,
    padding=NUMBER_FONT_PADDING,
):
    _, _, w, h = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max_scale
    while scale > min_scale:
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        if tw <= max(1, w - padding * 2) and th <= max(1, h - padding * 2):
            return scale
        scale -= 0.04
    return min_scale


def region_boundary_mask(region_map, min_component_area=10):
    regions_arr = np.asarray(region_map, dtype=np.int32)
    boundary = np.zeros(regions_arr.shape, dtype=np.uint8)
    horizontal = regions_arr[:, 1:] != regions_arr[:, :-1]
    vertical = regions_arr[1:, :] != regions_arr[:-1, :]
    boundary[:, 1:][horizontal] = 255
    boundary[1:, :][vertical] = 255
    boundary[regions_arr == 0] = 0
    if min_component_area > 1:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(boundary, 8)
        filtered = np.zeros_like(boundary)
        for component_id in range(1, n_labels):
            if stats[component_id, cv2.CC_STAT_AREA] >= min_component_area:
                filtered[labels == component_id] = 255
        boundary = filtered
    return boundary


def draw_simplified_boundary_layer(canvas, region_map, line_color=SEGMENTATION_LINE_COLOR, epsilon_ratio=0.004):
    boundary = region_boundary_mask(region_map)
    contours, _ = cv2.findContours(boundary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 6:
            continue
        arc = cv2.arcLength(contour, False)
        if arc < 8:
            continue
        epsilon = max(1.2, epsilon_ratio * arc)
        simplified = cv2.approxPolyDP(contour, epsilon, False)
        cv2.polylines(canvas, [simplified], False, line_color, 1, cv2.LINE_AA)
    return canvas


def draw_paint_by_number_style(
    detail_edges,
    regions,
    region_map,
    min_region_area,
    segmentation_edges=None,
):
    # Stage 12: render the printable coloring-book page. Segmentation lines are
    # black limits; detail-only strokes are gray so they do not imply fill areas.
    h, w = region_map.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    if segmentation_edges is None:
        draw_simplified_boundary_layer(canvas, region_map)
    else:
        canvas[segmentation_edges > 0] = SEGMENTATION_LINE_COLOR

    detail = clean_edges(detail_edges, open_iter=0, close_iter=0, thickness=1)
    detail_contours, _ = cv2.findContours(detail, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    for contour in detail_contours:
        contour_area = cv2.contourArea(contour)
        arc_length = cv2.arcLength(contour, False)
        has_visible_stroke = (
            contour_area >= DETAIL_RENDER_MIN_AREA
            or (
                len(contour) >= DETAIL_RENDER_MIN_POINTS
                and arc_length >= DETAIL_RENDER_MIN_ARC_LENGTH
            )
        )
        if has_visible_stroke:
            cv2.drawContours(
                canvas,
                [contour],
                -1,
                (DETAIL_LINE_GRAY, DETAIL_LINE_GRAY, DETAIL_LINE_GRAY),
                1,
                cv2.LINE_AA,
            )

    font = cv2.FONT_HERSHEY_SIMPLEX
    for region in sorted(regions, key=lambda r: r["area"], reverse=True):
        if region["area"] < min_region_area:
            continue
        text = str(region.get("color_id", region["id"]))
        cx, cy = best_label_point(region_map, int(region["id"]), region["bbox"])
        scale = min(
            NUMBER_FONT_MAX_SCALE,
            fit_number_font_scale(text, region["bbox"]) * NUMBER_FONT_SCALE_MULTIPLIER,
        )
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        x0, y0, bw, bh = region["bbox"]
        x = int(np.clip(cx - tw / 2, x0 + 2, max(x0 + 2, x0 + bw - tw - 2)))
        y = int(np.clip(cy + th / 2, y0 + th + 2, max(y0 + th + 2, y0 + bh - 2)))
        cv2.putText(canvas, text, (x, y), font, scale, (NUMBER_TEXT_GRAY, NUMBER_TEXT_GRAY, NUMBER_TEXT_GRAY), 1, cv2.LINE_AA)
    return canvas


def draw_segmentation_filled_preview(segmentation_edges, regions, region_map, label_map, palette):
    # Stage 13: render a validation preview that fills exactly the detected
    # regions, making leaks, missing borders, and tiny holes easy to inspect.
    h, w = region_map.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for region in regions:
        mask = region_map == int(region["id"])
        if not np.any(mask):
            continue
        labels_in_region = label_map[mask]
        if labels_in_region.size == 0:
            continue
        color_label = int(np.bincount(labels_in_region.astype(np.int32)).argmax())
        if 0 <= color_label < len(palette):
            canvas[mask] = palette[color_label]

    canvas[segmentation_edges > 0] = SEGMENTATION_LINE_COLOR
    return canvas


def render_difficulty(image, simplified, target_k, policy=None):
    # Stage 14: generate one difficulty level end-to-end from a working image.
    # This function intentionally keeps segmentation edges and printed lines
    # aligned so the visible coloring-book boundaries match the fill preview.
    policy = policy or {}
    area_scale = max(1.0, float(SEGMENTATION_OUTPUT_SCALE) ** 2)
    object_min_area = int(round(OBJECT_MIN_AREA * area_scale))
    min_region_area = int(round(MIN_REGION_AREA * area_scale))
    min_colorable_region_area = max(1, int(round(MIN_COLORABLE_REGION_AREA * area_scale)))

    kmeans_img, palette, labels = kmeans_quantization_with_labels(simplified, target_k)
    if policy.get("simple_color_art", False):
        # Flat artwork often has multiple near-white/noisy labels. Merge them
        # before edge extraction to avoid artificial background islands.
        labels, palette = merge_background_similar_labels(
            labels,
            palette,
            simplified,
            AUTO_SIMPLE_BACKGROUND_MERGE_DISTANCE,
        )
        kmeans_img = palette[labels]
    labels = absorb_tiny_label_islands(labels, palette)
    kmeans_img = palette[labels]
    up_image, up_kmeans, up_labels = upscale_for_segmentation_output(
        image,
        kmeans_img,
        labels,
        palette,
        SEGMENTATION_OUTPUT_SCALE,
    )

    object_raw = object_first_edges(up_labels, min_area=object_min_area, close_kernel=OBJECT_CLOSE_KERNEL)
    up_simplified = cv2.resize(
        simplified,
        (up_labels.shape[1], up_labels.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    source_boundary_edges = source_detail_boundary_edges(
        up_simplified,
        object_raw,
        low=policy.get("source_detail_low", SEGMENTATION_DETAIL_CANNY_LOW),
        high=policy.get("source_detail_high", SEGMENTATION_DETAIL_CANNY_HIGH),
    )
    dark_region_edges = dark_detail_region_edges(
        up_image,
        min_area=int(round(SEGMENTATION_DARK_REGION_MIN_AREA * area_scale)),
        max_area=int(round(SEGMENTATION_DARK_REGION_MAX_AREA * area_scale)),
    )
    closed_detail_edges = closed_detail_shape_edges(
        up_image,
        object_raw,
        low=policy.get("closed_detail_low", SEGMENTATION_CLOSED_DETAIL_LOW),
        high=policy.get("closed_detail_high", SEGMENTATION_CLOSED_DETAIL_HIGH),
        min_pixels=int(round(SEGMENTATION_CLOSED_DETAIL_MIN_PIXELS * area_scale)),
        max_pixels=int(round(SEGMENTATION_CLOSED_DETAIL_MAX_PIXELS * area_scale)),
        min_contour_area=SEGMENTATION_CLOSED_DETAIL_MIN_CONTOUR_AREA * area_scale,
        max_contour_area=SEGMENTATION_CLOSED_DETAIL_MAX_CONTOUR_AREA * area_scale,
        max_span=int(round(SEGMENTATION_CLOSED_DETAIL_MAX_SPAN * SEGMENTATION_OUTPUT_SCALE)),
    )
    segmentation_source_edges = cv2.bitwise_or(
        cv2.bitwise_or(cv2.bitwise_or(object_raw, source_boundary_edges), dark_region_edges),
        closed_detail_edges,
    )
    # Morphological closing reconnects small gaps so colorable regions are
    # bounded by continuous black lines.
    object_seg_connected = connect_segmentation_edges(
        segmentation_source_edges,
        kernel_size=policy.get("connect_kernel", SEGMENTATION_CONNECT_KERNEL),
        iterations=policy.get("connect_iter", SEGMENTATION_CONNECT_ITER),
    )
    object_seg_clean = clean_edges(
        object_seg_connected,
        open_iter=0,
        close_iter=1,
        thickness=SEGMENTATION_EDGE_THICKNESS,
    )
    object_final_clean = clean_edges(segmentation_source_edges, open_iter=0, close_iter=1, thickness=1)
    detail_edges = detail_expression_edges(
        up_image,
        object_final_clean,
        low=DETAIL_CANNY_LOW,
        high=DETAIL_CANNY_HIGH,
        min_area=int(round(DETAIL_MIN_AREA * area_scale)),
        max_area=int(round(DETAIL_MAX_AREA * area_scale)),
        dark_threshold=DARK_DETAIL_THRESHOLD,
        dark_min_area=int(round(DARK_DETAIL_MIN_AREA * area_scale)),
        dark_max_area=int(round(DARK_DETAIL_MAX_AREA * area_scale)),
    )

    segmentation_line_image = cv2.bitwise_not(object_seg_clean)
    # Connected components on the inverse line image are the actual fillable
    # regions used for numbering and for the filled validation preview.
    region_map, regions = segment_connected_components(segmentation_line_image, min_colorable_region_area)
    background_color = estimate_background_color(up_kmeans)
    regions = assign_region_color_numbers(
        regions,
        region_map,
        up_labels,
        palette,
        background_color=background_color,
        background_color_threshold=16,
        merge_background_similar=True,
    )
    background_ids = border_connected_region_ids(region_map)
    for region in regions:
        if int(region["id"]) in background_ids:
            region["is_background"] = True

    # Final output pair: printable line art plus a color-filled segmentation
    # preview for quality checking.
    result = draw_paint_by_number_style(
        detail_edges,
        regions,
        region_map,
        min_region_area,
        segmentation_edges=object_seg_clean,
    )
    filled_preview = draw_segmentation_filled_preview(
        object_seg_clean,
        regions,
        region_map,
        up_labels,
        palette,
    )
    numbered_regions = [region for region in regions if not region.get("is_background", False)]
    metrics = region_metrics(numbered_regions, up_image.shape, small_area=300)
    metrics["edge_density"] = edge_density(object_final_clean)
    return result, filled_preview, metrics


def run_batch(data_dir, output_dir, max_size=None, auto_policy=True):
    # Stage 15: batch orchestration for service-style processing. Each input
    # image is profiled once, then rendered at easy/normal/hard settings.
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {data_dir}")

    summary_rows = []
    for image_path in image_paths:
        start = time.perf_counter()
        # Load once per image so all difficulty levels share the same adaptive
        # policy and working resolution.
        image, profile, policy = load_image_preserve_size(
            image_path,
            max_size=max_size,
            auto_policy=auto_policy,
        )
        simplified = smooth_before_kmeans(image)
        min_k, _ = estimate_min_k_from_smoothed_colors(simplified)
        k_values = difficulty_k_values(min_k, policy=policy)
        print(
            (
                f"[{image_path.name}] "
                f"original={profile['original_width']}x{profile['original_height']} "
                f"working={profile['working_width']}x{profile['working_height']} "
                f"edge_density={profile['edge_density']:.4f} "
                f"blur={profile['blur_variance']:.1f} "
                f"lab_spread={profile['lab_spread']:.1f} "
                f"policy={policy['detail_boost']} "
                f"minK={min_k}, K={k_values}"
            ),
            flush=True,
        )
        for difficulty, target_k in k_values.items():
            # Render and save both the printable page and the filled preview.
            result, filled_preview, metrics = render_difficulty(image, simplified, target_k, policy=policy)
            output_path = output_dir / f"{image_path.stem}_{difficulty}_k{target_k}.png"
            save_image_rgb(str(output_path), result)
            filled_output_path = output_dir / f"{image_path.stem}_{difficulty}_k{target_k}_segmentation_filled.png"
            save_image_rgb(str(filled_output_path), filled_preview)
            summary_rows.append({
                "image": image_path.name,
                "difficulty": difficulty,
                "K": target_k,
                "regions": metrics["regions"],
                "small_regions": metrics["small_regions"],
                "edge_density": f"{metrics['edge_density']:.4f}",
                "original_size": f"{profile['original_width']}x{profile['original_height']}",
                "working_size": f"{profile['working_width']}x{profile['working_height']}",
                "policy": policy["detail_boost"],
                "k_upper_bound": policy["k_upper_bound"],
                "output": output_path.name,
            })
        print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)

    summary_path = output_dir / "batch_difficulty_summary.csv"
    columns = [
        "image",
        "difficulty",
        "K",
        "regions",
        "small_regions",
        "edge_density",
        "original_size",
        "working_size",
        "policy",
        "k_upper_bound",
        "output",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[col]) for col in columns) + "\n")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optional upper bound for the working long edge. Auto policy still upscales small sources.",
    )
    parser.add_argument(
        "--no-auto-policy",
        action="store_true",
        help="Disable image-profile-based policy selection and only apply --max-size if provided.",
    )
    args = parser.parse_args()
    summary_path = run_batch(
        args.data_dir,
        args.output_dir,
        max_size=args.max_size,
        auto_policy=not args.no_auto_policy,
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
