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
    """Visualize RGB palette values as color swatches.

    Figure width scales with the palette size so the index and RGB labels
    under each swatch never overlap, even for large K. The swatches are
    stretched to fill the width (aspect="auto") and the index number and RGB
    value are placed on separate lines for readability.
    """
    palette = np.asarray(palette, dtype=np.uint8)
    n = len(palette)
    cell_w = 100
    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 2.4))
    swatches = np.zeros((60, n * cell_w, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        swatches[:, i * cell_w:(i + 1) * cell_w] = color
        cx = i * cell_w + cell_w / 2
        ax.text(cx, 70, str(i + 1), ha="center", va="top",
                fontsize=10, fontweight="bold")
        ax.text(cx, 92, str(tuple(int(v) for v in color)),
                ha="center", va="top", fontsize=8)
    ax.imshow(swatches, aspect="auto")
    ax.set_xlim(0, n * cell_w)
    ax.set_ylim(120, 0)
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


def color_index_table_image(palette, regions=None, title="Color Index"):
    """Create a legend image mapping printed numbers to palette colors."""
    palette = np.asarray(palette, dtype=np.uint8)
    counts = {}
    if regions is not None:
        for region in regions:
            color_id = int(region.get("color_id", 0))
            if color_id > 0:
                counts[color_id] = counts.get(color_id, 0) + 1

    row_h = 42
    header_h = 82
    margin = 18
    width = 470
    height = header_h + row_h * len(palette) + margin
    table = np.full((height, width, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(table, title, (18, 32), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(table, "No", (18, 66), font, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(table, "Color", (72, 66), font, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(table, "RGB", (146, 66), font, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(table, "HEX", (302, 66), font, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(table, "Regions", (382, 66), font, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(table, (18, 74), (width - 18, 74), (190, 190, 190), 1)

    for i, color in enumerate(palette):
        color_id = i + 1
        y0 = header_h + i * row_h
        y_mid = y0 + 27
        if i % 2 == 0:
            table[y0:y0 + row_h, 10:width - 10] = (248, 248, 248)

        rgb = tuple(int(v) for v in color)
        hex_value = "#{:02X}{:02X}{:02X}".format(*rgb)
        cv2.putText(table, str(color_id), (20, y_mid), font, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(table, (76, y0 + 8), (118, y0 + 32), rgb, -1)
        cv2.rectangle(table, (76, y0 + 8), (118, y0 + 32), (80, 80, 80), 1)
        cv2.putText(table, str(rgb), (146, y_mid), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(table, hex_value, (302, y_mid), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(table, str(counts.get(color_id, 0)), (405, y_mid), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    return table


def save_color_index_table(palette, save_path, regions=None, title="Color Index"):
    """Save a legend image mapping printed numbers to palette colors."""
    table = color_index_table_image(palette, regions=regions, title=title)
    save_image_rgb(save_path, table)
    return table


def combine_with_color_index(image, palette, regions=None, save_path=None, title="Color Index"):
    """Place a color-index table to the right of a result image."""
    table = color_index_table_image(palette, regions=regions, title=title)
    target_h = image.shape[0]
    scale = target_h / table.shape[0]
    table_w = max(1, int(table.shape[1] * scale))
    table = cv2.resize(table, (table_w, target_h), interpolation=cv2.INTER_AREA)

    gutter = 18
    combined = np.full((target_h, image.shape[1] + gutter + table_w, 3), 255, dtype=np.uint8)
    combined[:, :image.shape[1]] = image
    combined[:, image.shape[1] + gutter:] = table
    if save_path:
        save_image_rgb(save_path, combined)
    return combined


def color_region_edge_preview(line_image, region_map, regions, palette=None, thickness=4):
    """Draw each segmented region boundary with its assigned palette color."""
    if line_image.ndim == 2:
        canvas = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = line_image.copy()

    palette = np.asarray(palette, dtype=np.uint8) if palette is not None else None
    for region in regions:
        mask = (region_map == region["id"]).astype(np.uint8) * 255
        if np.count_nonzero(mask) == 0:
            continue

        color = region.get("color_rgb")
        color_id = int(region.get("color_id", 0))
        if color is None and palette is not None and 1 <= color_id <= len(palette):
            color = tuple(int(v) for v in palette[color_id - 1])
        if color is None:
            color = (255, 0, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (35, 35, 35), thickness + 2, cv2.LINE_AA)
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color), thickness, cv2.LINE_AA)

    return canvas


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


def quantization_error(original, quantized):
    """Measure how well a quantized image preserves the original colors.

    Both images are converted to the Lab color space, which is closer to human
    color perception than RGB. The score is the mean per-pixel Lab distance, so
    a lower value means the simplified image keeps the original colors better.
    Useful for comparing K-Means, Posterization, and Median Cut quantitatively.
    """
    lab_original = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_quantized = cv2.cvtColor(quantized, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.mean(np.linalg.norm(lab_original - lab_quantized, axis=2)))


def _zhang_suen_thinning(binary):
    """Zhang-Suen thinning algorithm for binary images.

    Input: binary image with values 0 or 255 (uint8). Returns thinned binary image (0/255).
    """
    img = (binary > 0).astype(np.uint8)
    prev = np.zeros_like(img)
    changed = True
    while changed:
        changed = False
        # step 1
        m = np.zeros_like(img)
        rows, cols = img.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P2 = img[i - 1, j]
                P3 = img[i - 1, j + 1]
                P4 = img[i, j + 1]
                P5 = img[i + 1, j + 1]
                P6 = img[i + 1, j]
                P7 = img[i + 1, j - 1]
                P8 = img[i, j - 1]
                P9 = img[i - 1, j - 1]
                P1 = img[i, j]
                if P1 == 1:
                    neighbors = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    transitions = ((P2 == 0 and P3 == 1) + (P3 == 0 and P4 == 1) +
                                   (P4 == 0 and P5 == 1) + (P5 == 0 and P6 == 1) +
                                   (P6 == 0 and P7 == 1) + (P7 == 0 and P8 == 1) +
                                   (P8 == 0 and P9 == 1) + (P9 == 0 and P2 == 1))
                    if 2 <= neighbors <= 6 and transitions == 1 and (P2 * P4 * P6 == 0) and (P4 * P6 * P8 == 0):
                        m[i, j] = 1
        img = img & (~m)
        if np.any(m):
            changed = True

        # step 2
        m = np.zeros_like(img)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P2 = img[i - 1, j]
                P3 = img[i - 1, j + 1]
                P4 = img[i, j + 1]
                P5 = img[i + 1, j + 1]
                P6 = img[i + 1, j]
                P7 = img[i + 1, j - 1]
                P8 = img[i, j - 1]
                P9 = img[i - 1, j - 1]
                P1 = img[i, j]
                if P1 == 1:
                    neighbors = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    transitions = ((P2 == 0 and P3 == 1) + (P3 == 0 and P4 == 1) +
                                   (P4 == 0 and P5 == 1) + (P5 == 0 and P6 == 1) +
                                   (P6 == 0 and P7 == 1) + (P7 == 0 and P8 == 1) +
                                   (P8 == 0 and P9 == 1) + (P9 == 0 and P2 == 1))
                    if 2 <= neighbors <= 6 and transitions == 1 and (P2 * P4 * P8 == 0) and (P2 * P6 * P8 == 0):
                        m[i, j] = 1
        img = img & (~m)
        if np.any(m):
            changed = True

    return (img * 255).astype(np.uint8)


def thin_edges(edge_map):
    """Thin a binary edge map. Tries OpenCV ximgproc.thinning, falls back to Zhang-Suen.

    Input: edge_map (uint8) with 0/255 values or boolean. Returns thinned 0/255 uint8.
    """
    if edge_map.dtype != np.uint8:
        edge = (edge_map > 0).astype(np.uint8) * 255
    else:
        edge = np.where(edge_map > 0, 255, 0).astype(np.uint8)

    try:
        # prefer fast OpenCV thinning if available
        thin = cv2.ximgproc.thinning(edge)
        return thin
    except Exception:
        return _zhang_suen_thinning(edge)


def refine_edges_subpixel(edge_map):
    """Refine edges by extracting contours and drawing smoothed subpixel lines.

    This converts contour points to fitted lines and redraws with anti-aliased
    1-pixel thickness to produce visually thinner, smoother strokes.
    """
    edge = np.where(edge_map > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = edge.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if len(cnt) < 6:
            # small contours: draw directly
            cv2.drawContours(canvas, [cnt], -1, 255, 1, lineType=cv2.LINE_AA)
            continue
        # fit a polyline approximation then draw segments with antialiasing
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=False)
        pts = approx.reshape(-1, 2)
        for i in range(len(pts) - 1):
            p1 = tuple(map(int, pts[i]))
            p2 = tuple(map(int, pts[i + 1]))
            cv2.line(canvas, p1, p2, 255, 1, lineType=cv2.LINE_AA)
    return canvas


def enhance_eye_like_edges(image, edge_map, eye_mask=None):
    """High-level helper: thin edges, then refine contours to produce finer strokes.

    - `image`: RGB image (used for optional smoothing; currently unused but kept for extensibility)
    - `edge_map`: binary edge map (0/255)
    - `eye_mask`: optional binary mask to restrict processing area (0/255)
    Returns refined binary edge image (0/255).
    """
    if eye_mask is not None:
        mask = (eye_mask > 0).astype(np.uint8)
    else:
        mask = None

    # operate on whole map but can be masked later
    th = thin_edges(edge_map)
    refined = refine_edges_subpixel(th)
    if mask is not None:
        refined = np.where(mask, refined, 0).astype(np.uint8)
    return refined


def count_unique_colors(image):
    """Count how many distinct RGB colors actually remain in an image.

    The requested K is only an upper bound. Posterization in particular can
    return fewer colors than asked, so reporting the real count makes the
    algorithm comparison table more honest.
    """
    return int(len(np.unique(image.reshape(-1, image.shape[-1]), axis=0)))


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


def hybrid_canny_color_edges(image, low=50, high=130, label_map=None, color_delta=15):
    """Combine grayscale Canny edges with color boundaries while preserving sharp eye details.

    Applies a localized bilateral filter with strict color and spatial constraints 
    to preserve fine high-frequency details (like eyelashes, pupils, and iris textures)
    while successfully flattening large texture noise regions (like skin or gradients).

    To prevent doubled/overlapping edges at small detail regions (e.g. eyes):
    - Color boundary edges that fall within 1px of an existing Canny edge are suppressed.
    - Color edges are dilated by 1px before the Canny mask is applied, so genuine
      new boundaries (different color, no Canny response) still pass through.
    """
    # Optimized bilateral filter parameters to protect fine details (d=5, sigmas=30)
    smoothed = cv2.bilateralFilter(image, d=5, sigmaColor=30, sigmaSpace=30)
    
    # Grayscale conversion for Canny edge detection
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, low, high)
    
    # Detect color-based boundaries
    if label_map is None:
        color_edges = color_boundary_edges(smoothed, min_delta=color_delta)
    else:
        color_edges = color_boundary_edges_from_labels(label_map)

    # Suppress color edges that are already covered by Canny to avoid doubling/thickening
    # at fine structures (eyes, eyelashes, iris). Dilate Canny mask by 1px so that a
    # color boundary pixel directly adjacent to a Canny pixel is also suppressed.
    kernel1 = np.ones((3, 3), np.uint8)
    canny_dilated = cv2.dilate(canny, kernel1, iterations=1)
    color_edges_new = cv2.bitwise_and(color_edges, cv2.bitwise_not(canny_dilated))

    return cv2.bitwise_or(canny, color_edges_new)

def clean_edges(edges, open_iter=0, close_iter=0, thickness=1):
    """Clean noise while forcing thin, high-resolution line strokes for facial features.

    Changes vs. original:
    - Uses a 2x2 closing kernel instead of 3x3 to bridge gaps without thickening lines.
    - Keeps thickness=1 as a true single-pixel pass (no dilation at all).
    - Minimum component area stays at 8px to protect eyelash / fine-detail strokes.
    """
    kernel3 = np.ones((3, 3), np.uint8)
    # Smaller kernel for closing so broken ends connect without widening strokes
    kernel2 = np.ones((2, 2), np.uint8)
    result = edges.copy()
    
    if open_iter > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel3, iterations=open_iter)
    if close_iter > 0:
        # Use the 2x2 kernel for closing: closes small gaps with minimal widening
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel2, iterations=close_iter)
        
    # Filter out tiny isolated noise blobs, but keep small delicate strokes like eyes (area >= 8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    filtered_mask = np.zeros_like(result)
    
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            filtered_mask[labels == i] = 255
            
    result = filtered_mask
    
    # Only dilate if thickness is explicitly requested to be greater than 1
    if thickness > 1:
        result = cv2.dilate(result, kernel3, iterations=thickness - 1)
        
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
    font_scale=0.9,
    min_font_scale=0.25,
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

        scale = _fit_label_font_scale(
            text,
            font,
            region["bbox"],
            max_scale=font_scale,
            min_scale=min_font_scale,
        )
        thickness = max(1, int(round(scale * 2)))
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


def _fit_label_font_scale(text, font, bbox, max_scale=0.9, min_scale=0.25, padding=4):
    """Choose a text scale that fits inside the region bounding box."""
    _, _, w, h = bbox
    available_w = max(1, w - padding * 2)
    available_h = max(1, h - padding * 2)
    scale = max_scale

    for _ in range(12):
        thickness = max(1, int(round(scale * 2)))
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        text_h = th + base
        if tw <= available_w and text_h <= available_h:
            return max(min_scale, scale)

        width_ratio = available_w / max(1, tw)
        height_ratio = available_h / max(1, text_h)
        scale *= min(width_ratio, height_ratio) * 0.95
        if scale <= min_scale:
            return min_scale

    return max(min_scale, min(scale, max_scale))


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
