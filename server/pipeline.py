import os
import time
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

SERVER_DIR = Path(__file__).resolve().parent
SRC_DIR = SERVER_DIR / "src"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hci_project_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hci_project_cache"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coloringbook_utils import (
    assign_region_color_numbers,
    canny_edges,
    clean_edges,
    color_boundary_edges,
    color_region_edge_preview,
    color_region_preview,
    coloring_line_image,
    combine_with_color_index,
    contour_regions,
    count_unique_colors,
    edge_density,
    estimate_background_color,
    kmeans_quantization_with_labels,
    label_regions,
    laplacian_edges,
    load_image,
    median_cut_quantization,
    posterization,
    quantization_error,
    region_metrics,
    save_color_index_table,
    save_image_rgb,
    segment_connected_components,
    sobel_edges,
    thin_edges,
    timed_call,
    validate_color_count,
    watershed_segmentation,
)


DEFAULT_GENERATED_DIR = SERVER_DIR / "generated"

DEFAULT_SETTINGS = {
    "max_size": 900,
    "canny_low": 80,
    "canny_high": 180,
    "color_edge_delta": 35,
    "object_min_area": 260,
    "object_close_kernel": 5,
    "line_thickness": 1,
    "edge_compare_thickness": 1,
    "min_region_area": 160,
    "simplify_diameter": 9,
    "simplify_sigma_color": 90,
    "simplify_sigma_space": 90,
    "detail_canny_low": 90,
    "detail_canny_high": 200,
    "detail_min_area": 3,
    "detail_max_area": 1200,
    "dark_detail_threshold": 70,
    "dark_detail_min_area": 5,
    "dark_detail_max_area": 900,
}


def object_first_edges(label_map, min_area=260, close_kernel=5, thickness=1):
    """Create closed coloring-book edges from cleaned color-object contours."""
    labels = np.asarray(label_map, dtype=np.int32)
    edge = np.zeros(labels.shape, dtype=np.uint8)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)

    for label_id in np.unique(labels):
        mask = (labels == label_id).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            cv2.drawContours(edge, [contour], -1, 255, thickness, cv2.LINE_8)

    return edge


def detail_expression_edges(
    image,
    object_edges,
    low=100,
    high=220,
    min_area=8,
    max_area=450,
    dark_threshold=70,
    dark_min_area=5,
    dark_max_area=900,
):
    """Extract fine expression/detail strokes without affecting segmentation regions."""
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


def border_connected_region_ids(region_map):
    """Return connected-component ids touching the image border."""
    if region_map.size == 0:
        return set()
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


def main_object_mask_from_regions(region_map, regions, background_ids, source_image=None, bridge_iterations=3):
    """Keep the main subject silhouette so isolated paper/background texture is ignored."""
    if source_image is not None:
        border = np.concatenate([
            source_image[0, :, :],
            source_image[-1, :, :],
            source_image[:, 0, :],
            source_image[:, -1, :],
        ], axis=0)
        bg = np.median(border.reshape(-1, 3), axis=0)
        diff = np.linalg.norm(source_image.astype(np.float32) - bg.astype(np.float32), axis=2)
        gray_src = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
        seed = ((diff > 28) | (gray_src < 120)).astype(np.uint8) * 255
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=2)
        seed = cv2.dilate(seed, np.ones((5, 5), np.uint8), iterations=2)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
        if n_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(mask)
            if contours:
                cv2.drawContours(filled, contours, -1, 255, -1)
                filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=1)
                return filled

    candidate = np.zeros(region_map.shape, dtype=np.uint8)
    for region in regions:
        region_id = int(region["id"])
        if region_id in background_ids:
            continue
        candidate[region_map == region_id] = 255

    if np.count_nonzero(candidate) == 0:
        return candidate

    bridge = cv2.dilate(candidate, np.ones((3, 3), np.uint8), iterations=bridge_iterations)
    bridge = cv2.morphologyEx(bridge, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bridge, 8)
    if n_labels <= 1:
        return bridge

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def draw_paint_by_number_style(
    line_image,
    detail_edges,
    regions,
    region_map,
    background_ids,
    source_image=None,
    min_region_area=160,
    line_gray=185,
    text_gray=155,
    font_scale=0.42,
):
    """Render a white coloring page with thin gray lines and small gray numbers."""
    if line_image.ndim == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = line_image

    object_mask = main_object_mask_from_regions(region_map, regions, background_ids, source_image=source_image)
    keep_near_object = object_mask > 0
    edge_mask = ((gray < 128) | (detail_edges > 0)) & keep_near_object

    canvas = np.full((gray.shape[0], gray.shape[1], 3), 255, dtype=np.uint8)
    canvas[edge_mask] = line_gray

    font = cv2.FONT_HERSHEY_SIMPLEX
    for region in sorted(regions, key=lambda r: r["area"], reverse=True):
        region_id = int(region["id"])
        if region_id in background_ids or region["area"] < min_region_area:
            continue
        region_pixels = region_map == region_id
        object_overlap = np.count_nonzero(region_pixels & keep_near_object)
        if object_overlap == 0:
            continue
        if object_overlap / max(1, int(region["area"])) < 0.45:
            continue

        text = str(region.get("color_id", region_id))
        cx, cy = best_label_point(region_map, region_id, region["bbox"])
        thickness = 1
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        x0, y0, w, h = region["bbox"]
        x = int(np.clip(cx - tw / 2, x0 + 1, max(x0 + 1, x0 + w - tw - 1)))
        y = int(np.clip(cy + th / 2, y0 + th + 1, max(y0 + th + 1, y0 + h - 1)))
        cv2.putText(canvas, text, (x, y), font, font_scale, (text_gray, text_gray, text_gray), thickness, cv2.LINE_AA)
    return canvas


def palette_to_json(palette):
    result = []
    for index, color in enumerate(np.asarray(palette, dtype=np.uint8), start=1):
        rgb = [int(v) for v in color]
        result.append({
            "id": index,
            "rgb": rgb,
            "hex": "#{:02X}{:02X}{:02X}".format(*rgb),
        })
    return result


def generate_coloring_book(image_path, k, output_root=DEFAULT_GENERATED_DIR, result_id=None, settings=None):
    """Generate paint-by-number coloring-book assets from an image file."""
    start = time.perf_counter()
    k = validate_color_count(k)
    config = {**DEFAULT_SETTINGS, **(settings or {})}
    result_id = result_id or uuid4().hex
    output_dir = Path(output_root) / result_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(str(image_path), max_size=config["max_size"])
    (simplified_image, simplify_time) = timed_call(
        cv2.bilateralFilter,
        image,
        config["simplify_diameter"],
        config["simplify_sigma_color"],
        config["simplify_sigma_space"],
    )

    (kmeans_img, kmeans_palette, kmeans_labels), kmeans_time = timed_call(
        kmeans_quantization_with_labels,
        simplified_image,
        k,
    )
    (poster_img, _), poster_time = timed_call(posterization, simplified_image, k)
    (median_img, _), median_time = timed_call(median_cut_quantization, simplified_image, k)

    (sobel_raw, sobel_time) = timed_call(sobel_edges, kmeans_img, 65)
    (lap_raw, lap_time) = timed_call(laplacian_edges, kmeans_img, 25)
    (canny_raw, canny_time) = timed_call(canny_edges, kmeans_img, config["canny_low"], config["canny_high"])

    hybrid_start = time.perf_counter()
    color_raw = color_boundary_edges(kmeans_img, min_delta=config["color_edge_delta"])
    canny_covered = cv2.dilate(canny_raw, np.ones((3, 3), np.uint8), iterations=1)
    color_new = cv2.bitwise_and(color_raw, cv2.bitwise_not(canny_covered))
    hybrid_raw = cv2.bitwise_or(canny_raw, color_new)
    hybrid_time = time.perf_counter() - hybrid_start

    object_start = time.perf_counter()
    object_raw = object_first_edges(
        kmeans_labels,
        min_area=config["object_min_area"],
        close_kernel=config["object_close_kernel"],
        thickness=config["line_thickness"],
    )
    object_time = time.perf_counter() - object_start

    sobel_clean = clean_edges(sobel_raw, open_iter=1, close_iter=1, thickness=1)
    lap_clean = clean_edges(lap_raw, open_iter=1, close_iter=1, thickness=1)
    canny_clean = clean_edges(canny_raw, open_iter=0, close_iter=1, thickness=2)
    hybrid_compare_clean = clean_edges(
        hybrid_raw,
        open_iter=0,
        close_iter=1,
        thickness=config["edge_compare_thickness"],
    )
    object_seg_clean = clean_edges(object_raw, open_iter=0, close_iter=1, thickness=2)
    object_final_clean = thin_edges(object_seg_clean)
    detail_edges = detail_expression_edges(
        image,
        object_final_clean,
        low=config["detail_canny_low"],
        high=config["detail_canny_high"],
        min_area=config["detail_min_area"],
        max_area=config["detail_max_area"],
        dark_threshold=config["dark_detail_threshold"],
        dark_min_area=config["dark_detail_min_area"],
        dark_max_area=config["dark_detail_max_area"],
    )

    sobel_line = coloring_line_image(sobel_clean)
    lap_line = coloring_line_image(lap_clean)
    canny_line = coloring_line_image(canny_clean)
    hybrid_compare_line = coloring_line_image(hybrid_compare_clean)
    segmentation_line_image = coloring_line_image(object_seg_clean)
    object_line = coloring_line_image(object_final_clean)
    final_line_image = object_line

    (region_map, regions), seg_time = timed_call(
        segment_connected_components,
        segmentation_line_image,
        config["min_region_area"],
    )
    background_color = estimate_background_color(kmeans_img)
    regions = assign_region_color_numbers(
        regions,
        region_map,
        kmeans_labels,
        kmeans_palette,
        background_color=background_color,
        background_color_threshold=16,
        merge_background_similar=True,
    )
    background_region_ids = border_connected_region_ids(region_map)
    for region in regions:
        if int(region["id"]) in background_region_ids:
            region["is_background"] = True

    region_preview = color_region_preview(region_map)
    color_edge_preview = color_region_edge_preview(final_line_image, region_map, regions, kmeans_palette, thickness=3)

    colored_by_labels = np.full_like(simplified_image, 255)
    for region in regions:
        color = region.get("color_rgb")
        if color is None:
            continue
        colored_by_labels[region_map == region["id"]] = np.array(color, dtype=np.uint8)
    colored_by_labels[segmentation_line_image < 128] = 0

    line_with_detail = final_line_image.copy()
    line_with_detail[detail_edges > 0] = 0
    colored_by_labels_with_detail = colored_by_labels.copy()
    colored_by_labels_with_detail[detail_edges > 0] = 0

    numbered_coloringbook = draw_paint_by_number_style(
        final_line_image,
        detail_edges,
        regions,
        region_map,
        background_region_ids,
        source_image=simplified_image,
        min_region_area=config["min_region_area"],
    )
    colored_by_labels_numbered = label_regions(
        colored_by_labels_with_detail,
        regions,
        font_scale=0.65,
        region_map=region_map,
        skip_background=True,
    )
    color_index = save_color_index_table(
        kmeans_palette,
        str(output_dir / "palette.png"),
        regions,
        "Color Index",
    )
    numbered_with_index = combine_with_color_index(
        numbered_coloringbook,
        kmeans_palette,
        regions,
        str(output_dir / "numbered_with_color_index.png"),
        "Color Index",
    )

    contour_preview, _ = contour_regions(segmentation_line_image, config["min_region_area"])
    watershed_preview, _ = watershed_segmentation(kmeans_img)

    paths = {
        "original": output_dir / "original.png",
        "simplified": output_dir / "simplified_input.png",
        "kmeans": output_dir / "kmeans.png",
        "posterization": output_dir / "posterization.png",
        "median_cut": output_dir / "median_cut.png",
        "sobel": output_dir / "sobel.png",
        "laplacian": output_dir / "laplacian.png",
        "canny": output_dir / "canny.png",
        "hybrid_color_boundary": output_dir / "hybrid_color_boundary.png",
        "object_first_edges": output_dir / "object_first_edges.png",
        "segmentation_line": output_dir / "segmentation_line.png",
        "detail_edges": output_dir / "detail_edges.png",
        "line_with_detail": output_dir / "line_with_detail.png",
        "line_image": output_dir / "line_image.png",
        "region_preview": output_dir / "region_preview.png",
        "color_edge_preview": output_dir / "color_edge_preview.png",
        "contour_preview": output_dir / "contour_preview.png",
        "watershed_preview": output_dir / "watershed_preview.png",
        "colored_by_labels": output_dir / "colored_by_labels.png",
        "colored_by_labels_numbered": output_dir / "colored_by_labels_numbered.png",
        "coloring": output_dir / "final_numbered_coloringbook.png",
        "palette": output_dir / "palette.png",
        "combined": output_dir / "numbered_with_color_index.png",
    }

    save_image_rgb(str(paths["original"]), image)
    save_image_rgb(str(paths["simplified"]), simplified_image)
    save_image_rgb(str(paths["kmeans"]), kmeans_img)
    save_image_rgb(str(paths["posterization"]), poster_img)
    save_image_rgb(str(paths["median_cut"]), median_img)
    save_image_rgb(str(paths["sobel"]), sobel_line)
    save_image_rgb(str(paths["laplacian"]), lap_line)
    save_image_rgb(str(paths["canny"]), canny_line)
    save_image_rgb(str(paths["hybrid_color_boundary"]), hybrid_compare_line)
    save_image_rgb(str(paths["object_first_edges"]), object_line)
    save_image_rgb(str(paths["segmentation_line"]), segmentation_line_image)
    save_image_rgb(str(paths["detail_edges"]), coloring_line_image(detail_edges))
    save_image_rgb(str(paths["line_with_detail"]), line_with_detail)
    save_image_rgb(str(paths["line_image"]), final_line_image)
    save_image_rgb(str(paths["region_preview"]), region_preview)
    save_image_rgb(str(paths["color_edge_preview"]), color_edge_preview)
    save_image_rgb(str(paths["contour_preview"]), contour_preview)
    save_image_rgb(str(paths["watershed_preview"]), watershed_preview)
    save_image_rgb(str(paths["colored_by_labels"]), colored_by_labels)
    save_image_rgb(str(paths["colored_by_labels_numbered"]), colored_by_labels_numbered)
    save_image_rgb(str(paths["coloring"]), numbered_coloringbook)

    metrics = region_metrics(regions, image.shape, small_area=300)
    total_time = time.perf_counter() - start
    metrics.update({
        "k": k,
        "runtime_sec": total_time,
        "simplify_time_sec": simplify_time,
        "kmeans_time_sec": kmeans_time,
        "posterization_time_sec": poster_time,
        "median_cut_time_sec": median_time,
        "sobel_time_sec": sobel_time,
        "laplacian_time_sec": lap_time,
        "canny_time_sec": canny_time,
        "hybrid_time_sec": hybrid_time,
        "object_time_sec": object_time,
        "segmentation_time_sec": seg_time,
        "color_error": quantization_error(simplified_image, kmeans_img),
        "unique_colors": count_unique_colors(kmeans_img),
        "edge_density": edge_density(object_final_clean),
    })

    return {
        "result_id": result_id,
        "output_dir": str(output_dir),
        "paths": {key: str(value) for key, value in paths.items()},
        "palette": palette_to_json(kmeans_palette),
        "metrics": metrics,
        "image_size": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "assets": {
            "color_index": color_index,
            "numbered_with_index": numbered_with_index,
        },
    }
