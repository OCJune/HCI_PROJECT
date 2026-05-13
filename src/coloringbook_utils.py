import os
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "outputs"
DATA_DIR = "data"


def ensure_dirs():
    """Create the folders used by every notebook."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def create_sample_image(path="data/sample_input.png", size=420):
    """Create a simple sample image so every notebook runs without external files."""
    ensure_dirs()
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (30, 30), (390, 390), (235, 235, 235), -1)
    cv2.circle(canvas, (145, 150), 82, (235, 80, 90), -1)
    cv2.circle(canvas, (280, 145), 70, (70, 150, 235), -1)
    cv2.rectangle(canvas, (90, 250), (210, 360), (90, 190, 105), -1)
    pts = np.array([[250, 245], [360, 310], [280, 370]], np.int32)
    cv2.fillPoly(canvas, [pts], (240, 190, 60))
    cv2.line(canvas, (40, 215), (375, 215), (120, 70, 170), 16)
    cv2.GaussianBlur(canvas, (3, 3), 0, dst=canvas)
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return path


def load_image(image_path=None, max_size=900):
    """Load RGB image. If no path is given, load or create the sample image."""
    ensure_dirs()
    if image_path is None or image_path == "":
        image_path = "data/sample_input.png"
        if not os.path.exists(image_path):
            create_sample_image(image_path)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found or unsupported: {image_path}")

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def save_image_rgb(path, image):
    """Save an RGB or grayscale image with OpenCV."""
    ensure_dirs()
    if image.ndim == 2:
        cv2.imwrite(path, image)
    else:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def timed_call(func, *args, **kwargs):
    """Run a function and return (result, runtime_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def show_images(items, cols=3, figsize=(14, 8), cmap=None, save_path=None):
    """Display a list of (title, image) pairs."""
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap=cmap if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_palette(palette, title="RGB Palette", save_path=None):
    """Visualize RGB palette values as color swatches."""
    palette = np.asarray(palette, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(max(8, len(palette) * 0.8), 1.8))
    swatches = np.zeros((60, len(palette) * 60, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        swatches[:, i * 60:(i + 1) * 60] = color
        ax.text(i * 60 + 30, 78, f"{i + 1}\n{tuple(int(v) for v in color)}",
                ha="center", va="top", fontsize=8)
    ax.imshow(swatches)
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


def kmeans_quantization(image, k=10, attempts=3):
    """Color quantization using OpenCV K-Means clustering."""
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(image.shape)
    return quantized, centers


def kmeans_quantization_with_labels(image, k=10, attempts=3):
    """Return quantized image, palette, and per-pixel color label map."""
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    label_map = labels.reshape(image.shape[:2]).astype(np.int32)
    quantized = centers[label_map]
    return quantized, centers, label_map


def posterization(image, k=10):
    """Fast uniform posterization. K is mapped to a per-channel level count."""
    levels = max(2, int(np.ceil(k ** (1 / 3))))
    bins = np.linspace(0, 256, levels + 1)
    centers = ((bins[:-1] + bins[1:]) / 2).astype(np.uint8)
    idx = np.digitize(image, bins[1:-1], right=False)
    result = centers[idx]
    palette = np.unique(result.reshape(-1, 3), axis=0)
    if len(palette) > k:
        result, palette = kmeans_quantization(result, k)
    return result.astype(np.uint8), palette.astype(np.uint8)


def _median_cut_box(pixels, target_boxes):
    boxes = deque([pixels])
    while len(boxes) < target_boxes:
        box = boxes.popleft()
        if len(box) <= 1:
            boxes.append(box)
            break
        ranges = np.ptp(box, axis=0)
        channel = int(np.argmax(ranges))
        sorted_box = box[np.argsort(box[:, channel])]
        mid = len(sorted_box) // 2
        boxes.append(sorted_box[:mid])
        boxes.append(sorted_box[mid:])
    return list(boxes)


def median_cut_quantization(image, k=10, sample_size=60000):
    """Educational Median Cut implementation for comparison."""
    pixels = image.reshape(-1, 3)
    if len(pixels) > sample_size:
        rng = np.random.default_rng(7)
        pixels_for_boxes = pixels[rng.choice(len(pixels), sample_size, replace=False)]
    else:
        pixels_for_boxes = pixels
    boxes = _median_cut_box(pixels_for_boxes.astype(np.uint8), k)
    palette = np.array([np.mean(box, axis=0) for box in boxes], dtype=np.uint8)

    flat = image.reshape(-1, 3).astype(np.int16)
    pal = palette.astype(np.int16)
    distances = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    result = palette[labels].reshape(image.shape)
    return result, palette


def sobel_edges(image, threshold=70):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    _, binary = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    return binary


def laplacian_edges(image, threshold=25):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
    abs_lap = cv2.convertScaleAbs(lap)
    _, binary = cv2.threshold(abs_lap, threshold, 255, cv2.THRESH_BINARY)
    return binary


def canny_edges(image, low=60, high=150):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, low, high)


def color_boundary_edges_from_labels(label_map):
    """Detect borders where neighboring quantized color labels are different."""
    edges = np.zeros(label_map.shape, dtype=np.uint8)
    edges[:, 1:] |= (label_map[:, 1:] != label_map[:, :-1]).astype(np.uint8) * 255
    edges[1:, :] |= (label_map[1:, :] != label_map[:-1, :]).astype(np.uint8) * 255
    return edges


def color_boundary_edges(image, min_delta=18):
    """Detect chromatic boundaries that grayscale edge detectors can miss."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.int16)
    edges = np.zeros(image.shape[:2], dtype=np.uint8)
    horizontal = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2) >= min_delta
    vertical = np.linalg.norm(lab[1:, :] - lab[:-1, :], axis=2) >= min_delta
    edges[:, 1:][horizontal] = 255
    edges[1:, :][vertical] = 255
    return edges


def hybrid_canny_color_edges(image, low=60, high=150, label_map=None, color_delta=18):
    """Combine grayscale Canny edges with color-region boundaries.

    Canny can miss borders between colors with similar brightness. The color edge
    term preserves boundaries where neighboring quantized regions or Lab colors
    differ, which is important for overlapping colored objects.
    """
    canny = canny_edges(image, low, high)
    if label_map is None:
        color_edges = color_boundary_edges(image, min_delta=color_delta)
    else:
        color_edges = color_boundary_edges_from_labels(label_map)
    return cv2.bitwise_or(canny, color_edges)


def clean_edges(edges, open_iter=0, close_iter=1, thickness=1):
    """Apply morphology and line thickness control to binary edge map."""
    kernel = np.ones((3, 3), np.uint8)
    result = edges.copy()
    if open_iter > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if thickness > 1:
        result = cv2.dilate(result, kernel, iterations=thickness - 1)
    return result


def coloring_line_image(edges):
    """Convert white-edge-on-black map to black-line-on-white coloring-book page."""
    return cv2.bitwise_not(edges)


def edge_density(edges):
    return float(np.count_nonzero(edges) / edges.size)


def segment_connected_components(line_image, min_area=120):
    """Segment colorable white regions separated by black lines."""
    if line_image.ndim == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = line_image
    white_regions = (gray > 200).astype(np.uint8) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_regions, 8)

    regions = []
    region_map = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        mask = labels == label
        region_map[mask] = new_id
        regions.append({
            "id": new_id,
            "area": area,
            "bbox": (x, y, w, h),
            "centroid": (float(cx), float(cy)),
        })
        new_id += 1
    return region_map, regions


def estimate_background_labels(label_map, max_labels=1, min_border_fraction=0.15):
    """Estimate background color labels from the image border.

    The background often appears on the outer border. Using the dominant border
    label lets us suppress background-colored islands even when they are enclosed
    between foreground objects and do not touch the image edge.
    """
    top = label_map[0, :]
    bottom = label_map[-1, :]
    left = label_map[:, 0]
    right = label_map[:, -1]
    border_labels = np.concatenate([top, bottom, left, right]).astype(np.int32)
    counts = np.bincount(border_labels)
    order = np.argsort(counts)[::-1]
    total = max(1, border_labels.size)
    background = []
    for label in order[:max_labels]:
        if counts[label] / total >= min_border_fraction:
            background.append(int(label))
    return set(background)


def estimate_background_color(image, sample_border=12):
    """Estimate the RGB background color from border pixels.

    A median color is more robust than a mean when small foreground objects touch
    the border. The result is used to skip regions whose dominant palette color
    is visually close to the page background, even if K-Means gave it a separate
    label.
    """
    h, w = image.shape[:2]
    border = max(1, min(sample_border, h // 2, w // 2))
    samples = np.concatenate([
        image[:border, :, :].reshape(-1, 3),
        image[-border:, :, :].reshape(-1, 3),
        image[:, :border, :].reshape(-1, 3),
        image[:, -border:, :].reshape(-1, 3),
    ], axis=0)
    return np.median(samples, axis=0).astype(np.uint8)


def color_distance_lab(color_a, color_b):
    """Return perceptual-ish distance between two RGB colors in Lab space."""
    arr = np.array([[color_a, color_b]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)[0]
    return float(np.linalg.norm(lab[0] - lab[1]))


def assign_region_color_numbers(
    regions,
    region_map,
    label_map,
    palette=None,
    background_labels=None,
    background_color=None,
    background_color_threshold=16,
    merge_background_similar=True,
):
    """Assign each segmented region the dominant K-Means color number.

    Region ids are unique shape ids, but coloring-book numbers should represent
    palette colors. This function adds color_label, color_id, and color_rgb to
    each region so separated areas with the same dominant color get the same
    printed number.
    """
    if background_labels is None:
        background_labels = estimate_background_labels(label_map)
    background_labels = set(background_labels)
    background_label = min(background_labels) if background_labels else None
    if background_color is not None and palette is not None and len(palette) > 0:
        distances = [color_distance_lab(color, background_color) for color in palette]
        background_label = int(np.argmin(distances))

    updated = []
    for region in regions:
        mask = region_map == region["id"]
        labels = label_map[mask]
        if labels.size == 0:
            color_label = -1
        else:
            color_label = int(np.bincount(labels.astype(np.int32)).argmax())

        enriched = dict(region)
        enriched["color_label"] = color_label
        enriched["color_id"] = color_label + 1 if color_label >= 0 else region["id"]
        if palette is not None and 0 <= color_label < len(palette):
            enriched["color_rgb"] = tuple(int(v) for v in palette[color_label])
        else:
            enriched["color_rgb"] = None

        similar_to_background = False
        background_distance = None
        if background_color is not None and enriched["color_rgb"] is not None:
            background_distance = color_distance_lab(enriched["color_rgb"], background_color)
            similar_to_background = background_distance <= background_color_threshold

        if merge_background_similar and similar_to_background and background_label is not None:
            color_label = background_label
            enriched["color_label"] = color_label
            enriched["color_id"] = color_label + 1
            if palette is not None and 0 <= color_label < len(palette):
                enriched["color_rgb"] = tuple(int(v) for v in palette[color_label])

        enriched["background_distance"] = background_distance
        enriched["is_background"] = color_label in background_labels or similar_to_background
        updated.append(enriched)
    return updated


def colorable_regions(regions):
    """Return regions that receive printed color numbers."""
    return list(regions)


def contour_regions(line_image, min_area=120):
    """Find external contours as an alternative region proposal method."""
    if line_image.ndim == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = line_image
    contours, _ = cv2.findContours((gray > 200).astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    kept = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        kept.append(contour)
        cv2.drawContours(output, [contour], -1, (255, 0, 0), 2)
    return output, kept


def watershed_segmentation(image):
    """Watershed comparison method. It is useful but often over-segments simple pages."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)
    result = image.copy()
    result[markers == -1] = [255, 0, 0]
    return result, markers


def label_regions(
    line_image,
    regions,
    font_scale=0.45,
    skip_background=False,
    region_map=None,
    avoid_overlap=False,
):
    """Insert color numbers inside each segmented region.

    If region_map is supplied, the label is placed near the largest empty point
    inside that exact connected component. This is closer to paint-by-number
    sheets than using only the geometric centroid.
    """
    if line_image.ndim == 2:
        canvas = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = line_image.copy()

    occupied = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    sorted_regions = sorted(regions, key=lambda r: r["area"], reverse=True)

    for region in sorted_regions:
        if skip_background and region.get("is_background", False):
            continue
        text = str(region.get("color_id", region["id"]))
        x0, y0, w, h = region["bbox"]

        if region_map is not None:
            cx, cy = _best_label_point(region_map, region["id"], region["bbox"])
        else:
            cx, cy = region["centroid"]

        scale = font_scale
        thickness = 1
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

        x = int(np.clip(cx - tw / 2, x0 + 1, max(x0 + 1, x0 + w - tw - 1)))
        y = int(np.clip(cy + th / 2, y0 + th + 1, max(y0 + th + 1, y0 + h - 1)))
        box = (x - 2, y - th - 2, x + tw + 2, y + base + 2)

        chosen = (x, y, box)
        if avoid_overlap:
            overlaps = any(
                not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                for b in occupied
            )
            if overlaps:
                chosen = None
        if chosen is None:
            continue
        x, y, box = chosen
        occupied.append(box)
        cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def _best_label_point(region_map, region_id, bbox):
    """Find an interior point that is far from the component boundary."""
    x, y, w, h = bbox
    component = (region_map[y:y + h, x:x + w] == region_id).astype(np.uint8)
    if component.size == 0 or np.count_nonzero(component) == 0:
        return (x + w / 2, y + h / 2)

    padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    px, py = max_loc
    return float(x + px), float(y + py)


def color_region_preview(region_map):
    """Render connected-component labels with random colors for visual inspection."""
    rng = np.random.default_rng(4)
    preview = np.full((*region_map.shape, 3), 255, dtype=np.uint8)
    for label in np.unique(region_map):
        if label == 0:
            continue
        preview[region_map == label] = rng.integers(50, 235, size=3, dtype=np.uint8)
    return preview


def region_metrics(regions, image_shape, small_area=300):
    areas = [r["area"] for r in regions]
    return {
        "regions": len(regions),
        "average_area": float(np.mean(areas)) if areas else 0.0,
        "small_regions": int(sum(a < small_area for a in areas)),
        "region_coverage": float(sum(areas) / (image_shape[0] * image_shape[1])) if areas else 0.0,
    }


def complexity_by_k(image, k_values=(5, 10, 20), min_area=120):
    """Measure how color count changes runtime, edge density, and region complexity."""
    rows = []
    for k in k_values:
        (quantized, _, labels), q_time = timed_call(kmeans_quantization_with_labels, image, k)
        edges, e_time = timed_call(hybrid_canny_color_edges, quantized, 60, 150, labels)
        edges = clean_edges(edges, close_iter=1, thickness=1)
        line = coloring_line_image(edges)
        (region_map, regions), s_time = timed_call(segment_connected_components, line, min_area)
        metrics = region_metrics(regions, image.shape)
        rows.append({
            "K": k,
            "runtime_sec": q_time + e_time + s_time,
            "edge_density": edge_density(edges),
            "regions": metrics["regions"],
            "average_area": metrics["average_area"],
            "small_regions": metrics["small_regions"],
        })
    return rows


def print_table(rows, columns=None):
    """Print simple aligned text table without requiring pandas."""
    if not rows:
        print("(no rows)")
        return
    columns = columns or list(rows[0].keys())
    prepared = []
    for row in rows:
        prepared.append([_format_cell(row.get(col, "")) for col in columns])
    widths = [max(len(str(col)), *(len(row[i]) for row in prepared)) for i, col in enumerate(columns)]
    header = " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
    sep = "-+-".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in prepared:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


def _format_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
