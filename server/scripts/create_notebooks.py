import json
import os
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def write_notebook(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / name
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


COMMON_IMPORTS = r"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 공통 함수가 들어 있는 src 폴더를 import 경로에 추가합니다.
sys.path.append("src")
from coloringbook_utils import *

# 결과 저장 폴더(outputs)와 샘플 입력 폴더(data)를 생성합니다.
ensure_dirs()
plt.rcParams["figure.dpi"] = 120

# 직접 사용할 이미지가 있으면 여기에 경로를 넣으세요.
# 예: IMAGE_PATH = "data/my_image.jpg"
IMAGE_PATH = "data/sample_input.png"

# IMAGE_PATH가 None이면 발표/실험용 샘플 이미지가 자동 생성됩니다.
image = load_image(IMAGE_PATH)
show_images([("Original Image", image)], cols=1, figsize=(5, 5))
"""


FINAL_IMPORTS = COMMON_IMPORTS.replace(
    '# 직접 사용할 이미지가 있으면 여기에 경로를 넣으세요.\n'
    '# 예: IMAGE_PATH = "data/my_image.jpg"\n'
    'IMAGE_PATH = "data/sample_input.png"\n'
    '\n'
    '# IMAGE_PATH가 None이면 발표/실험용 샘플 이미지가 자동 생성됩니다.',
    '# 04_final_pipeline은 data/ani.png만 입력 이미지로 사용합니다.\n'
    'IMAGE_PATH = "data/ani.png"\n'
    '\n'
    '# data/ani.png를 기준으로 전체 파이프라인을 실행합니다.',
)


def color_quantization_nb():
    cells = [
        md("""
# 01. Color Quantization

목표는 원본 이미지를 사람이 색칠하기 쉬운 제한된 색상 수로 단순화하는 것입니다.

비교 알고리즘:

- K-Means Clustering: 이미지 픽셀을 K개의 대표 색상 군집으로 묶습니다.
- Posterization: RGB 채널을 균등 구간으로 나누는 빠른 단순화 방식입니다.
- Median Cut: 색상 분포가 넓은 축을 반복 분할해 대표 팔레트를 만듭니다.

최종 기본 알고리즘은 색상 유지 품질과 발표 설명 용이성을 고려해 K-Means로 선택합니다.
"""),
        code(COMMON_IMPORTS),
        md("""
## K-Means, Posterization, Median Cut 실행

아래 셀에서 원하는 색상 개수 `K`를 바꿀 수 있습니다.
"""),
        code(r"""
# K는 최종 팔레트에 남길 대표 색상 개수입니다.
# K가 작으면 색칠은 쉬워지고, K가 크면 원본 색상 보존이 좋아집니다.
K = 10

# K-Means: 픽셀을 K개 군집으로 묶고 각 군집 중심색으로 이미지를 치환합니다.
(kmeans_img, kmeans_palette), kmeans_time = timed_call(kmeans_quantization, image, K)

# Posterization: RGB 채널을 일정한 구간으로 나눠 빠르게 색상을 줄입니다.
(poster_img, poster_palette), poster_time = timed_call(posterization, image, K)

# Median Cut: 색상 분포가 넓은 축을 반복 분할해 대표 색상표를 만듭니다.
(median_img, median_palette), median_time = timed_call(median_cut_quantization, image, K)

# 각 알고리즘 결과를 파일로 저장해 보고서/발표 자료에 바로 사용할 수 있게 합니다.
save_image_rgb("outputs/01_kmeans.png", kmeans_img)
save_image_rgb("outputs/01_posterization.png", poster_img)
save_image_rgb("outputs/01_median_cut.png", median_img)

show_images([
    ("Original", image),
    (f"K-Means K={K} ({kmeans_time:.3f}s)", kmeans_img),
    (f"Posterization K≈{K} ({poster_time:.3f}s)", poster_img),
    (f"Median Cut K={K} ({median_time:.3f}s)", median_img),
], cols=2, figsize=(11, 8), save_path="outputs/01_quantization_compare.png")
"""),
        md("""
## RGB 색상표

색상표는 최종 컬러링북 번호와 연결되는 색상 안내표로 사용할 수 있습니다.
"""),
        code(r"""
plot_palette(kmeans_palette, "K-Means RGB Palette", "outputs/01_kmeans_palette.png")
plot_palette(poster_palette, "Posterization RGB Palette", "outputs/01_poster_palette.png")
plot_palette(median_palette, "Median Cut RGB Palette", "outputs/01_median_palette.png")
"""),
        md("""
## 5색, 10색, 20색 비교

색상 개수가 늘어나면 원본 보존력은 좋아지지만, 색칠해야 할 영역과 인지 복잡도도 함께 증가합니다.
"""),
        code(r"""
# 색상 수가 늘어날 때 시각 품질과 복잡도가 어떻게 변하는지 비교합니다.
k_values = [5, 10, 20]
comparison_items = [("Original", image)]
timing_rows = []

for k in k_values:
    # 같은 입력 이미지에 대해 K만 바꿔 K-Means를 반복 실행합니다.
    (result, palette), runtime = timed_call(kmeans_quantization, image, k)
    save_image_rgb(f"outputs/01_kmeans_k{k}.png", result)
    comparison_items.append((f"K-Means K={k} ({runtime:.3f}s)", result))
    timing_rows.append({"algorithm": "K-Means", "K": k, "runtime_sec": runtime, "palette_size": len(palette)})

show_images(comparison_items, cols=2, figsize=(11, 10), save_path="outputs/01_k_compare.png")
print_table(timing_rows)
"""),
        md("""
## 비교 분석

- 색상 유지 품질: K-Means는 실제 이미지 색상 분포에 맞춰 대표색을 찾기 때문에 세 알고리즘 중 원본 느낌을 가장 잘 유지하는 편입니다.
- 경계 자연스러움: K-Means와 Median Cut은 색상 분포 기반이라 Posterization보다 계단 현상이 적습니다.
- 처리 속도: Posterization이 가장 빠르고, K-Means는 반복 최적화 때문에 중간, Median Cut은 구현 방식에 따라 중간 정도입니다.
- 사용자 관점: K가 작을수록 색칠은 쉽지만 원본 정보가 손실됩니다. 발표용 기본값은 보통 `K=10`이 균형적입니다.
"""),
    ]
    write_notebook("01_color_quantization.ipynb", cells)


def edge_detection_nb():
    cells = [
        md("""
# 02. Edge Detection

목표는 색칠 경계로 사용할 선을 추출하고, 흰 배경과 검은 선 형태의 컬러링북 선화를 만드는 것입니다.

비교 알고리즘:

- Sobel: x/y 방향 밝기 변화량을 계산합니다.
- Laplacian: 2차 미분 기반으로 급격한 변화 지점을 찾습니다.
- Canny: 노이즈 제거, 그래디언트, 비최대 억제, 이중 임계값을 함께 사용하는 대표적인 경계 추출 방식입니다.
- Hybrid Color Boundary: Canny가 놓치는 비슷한 밝기의 색상 경계를 보완합니다.

최종 기본 알고리즘은 Canny에 색상 라벨 경계를 합친 Hybrid 방식입니다.
"""),
        code(COMMON_IMPORTS),
        md("""
## 경계 추출 비교

먼저 K-Means로 색상을 단순화한 뒤 경계를 추출하면 원본의 작은 질감 노이즈가 줄어듭니다.
"""),
        code(r"""
# 경계선 추출 전에 색상을 단순화하면 원본 텍스처 노이즈가 줄어듭니다.
K = 10
quantized, palette, label_map = kmeans_quantization_with_labels(image, K)

# Sobel: x/y 방향 밝기 변화량을 이용해 경계를 찾습니다.
(sobel, sobel_time) = timed_call(sobel_edges, quantized, 65)

# Laplacian: 2차 미분으로 급격한 밝기 변화를 찾습니다.
(laplacian, lap_time) = timed_call(laplacian_edges, quantized, 25)

# Canny: 노이즈 제거, 비최대 억제, 이중 임계값을 사용하는 기본 선택 알고리즘입니다.
(canny, canny_time) = timed_call(canny_edges, quantized, 60, 150)

# Color Boundary: K-Means 대표 색상 라벨이 바뀌는 지점을 경계로 추가합니다.
# 밝기는 비슷하지만 색상이 다른 겹친 영역을 분리하는 데 중요합니다.
(hybrid, hybrid_time) = timed_call(hybrid_canny_color_edges, quantized, 60, 150, label_map)

# Opening은 작은 점 노이즈 제거, Closing은 끊긴 선 연결에 사용합니다.
sobel_clean = clean_edges(sobel, open_iter=1, close_iter=1, thickness=1)
lap_clean = clean_edges(laplacian, open_iter=1, close_iter=1, thickness=1)
canny_clean = clean_edges(canny, open_iter=0, close_iter=1, thickness=2)
hybrid_clean = clean_edges(hybrid, open_iter=0, close_iter=1, thickness=2)

save_image_rgb("outputs/02_sobel_lines.png", coloring_line_image(sobel_clean))
save_image_rgb("outputs/02_laplacian_lines.png", coloring_line_image(lap_clean))
save_image_rgb("outputs/02_canny_lines.png", coloring_line_image(canny_clean))
save_image_rgb("outputs/02_hybrid_color_boundary_lines.png", coloring_line_image(hybrid_clean))

show_images([
    ("Quantized Input", quantized),
    (f"Sobel density={edge_density(sobel_clean):.3f}", coloring_line_image(sobel_clean)),
    (f"Laplacian density={edge_density(lap_clean):.3f}", coloring_line_image(lap_clean)),
    (f"Canny density={edge_density(canny_clean):.3f}", coloring_line_image(canny_clean)),
    (f"Hybrid density={edge_density(hybrid_clean):.3f}", coloring_line_image(hybrid_clean)),
], cols=2, figsize=(11, 10), cmap="gray", save_path="outputs/02_edge_compare.png")
"""),
        md("""
## 선 두께와 Morphology 조절

`thickness`, `open_iter`, `close_iter` 값을 바꾸면 번호 삽입과 색칠 난이도에 맞춰 선을 조절할 수 있습니다.
"""),
        code(r"""
variants = []
for thickness in [1, 2, 3]:
    # thickness가 커질수록 선은 잘 보이지만 색칠 가능한 흰 영역은 줄어듭니다.
    adjusted = clean_edges(hybrid, open_iter=0, close_iter=1, thickness=thickness)
    variants.append((f"Hybrid thickness={thickness}", coloring_line_image(adjusted)))
    save_image_rgb(f"outputs/02_hybrid_thickness_{thickness}.png", coloring_line_image(adjusted))

show_images(variants, cols=3, figsize=(13, 4), cmap="gray", save_path="outputs/02_thickness_compare.png")
"""),
        md("""
## 성능 비교 표

Edge Density는 전체 픽셀 중 경계 픽셀이 차지하는 비율입니다. 너무 낮으면 선이 끊기고, 너무 높으면 색칠 공간이 좁아집니다.
"""),
        code(r"""
edge_rows = [
    {"algorithm": "Sobel", "runtime_sec": sobel_time, "edge_density": edge_density(sobel_clean), "hci_note": "두꺼운 변화 감지, 질감 노이즈 주의"},
    {"algorithm": "Laplacian", "runtime_sec": lap_time, "edge_density": edge_density(lap_clean), "hci_note": "세부 변화 민감, 작은 노이즈 많음"},
    {"algorithm": "Canny", "runtime_sec": canny_time, "edge_density": edge_density(canny_clean), "hci_note": "선명도와 노이즈 균형 우수"},
    {"algorithm": "Hybrid Color Boundary", "runtime_sec": hybrid_time, "edge_density": edge_density(hybrid_clean), "hci_note": "비슷한 밝기의 색상 경계까지 분리"},
]
print_table(edge_rows)
"""),
        md("""
## 비교 분석

- 선 명확도: Canny가 끊김이 적고 외곽선이 비교적 안정적입니다.
- 노이즈: Laplacian은 작은 밝기 변화에도 민감해 노이즈가 많을 수 있습니다.
- 처리 속도: 세 방법 모두 빠르지만 Sobel/Laplacian이 단순하고 Canny가 약간 더 많은 단계를 가집니다.
- 컬러링북 적합성: Canny + Color Boundary + Closing + 적당한 Dilate 조합이 겹친 색상 영역까지 분리해 가장 적합합니다.
"""),
    ]
    write_notebook("02_edge_detection.ipynb", cells)


def segmentation_nb():
    cells = [
        md("""
# 03. Segmentation & Labeling

목표는 검은 선으로 둘러싸인 흰 영역을 색칠 가능한 영역으로 분리하고, 각 영역 중심에 색상 번호를 삽입하는 것입니다.

비교 알고리즘:

- Contour Detection: 외곽선을 추적해 영역 후보를 찾습니다.
- Connected Components Labeling: 연결된 흰 영역을 라벨로 분리합니다.
- Watershed: 거리 변환과 마커를 기반으로 영역을 나누는 비교용 방법입니다.

최종 기본 알고리즘은 Connected Components로 영역을 안정적으로 분리하고, Contour Detection을 보조 시각화에 사용합니다.
"""),
        code(COMMON_IMPORTS),
        md("""
## 선화 생성 후 영역 분리

작은 영역은 색칠하기 어렵고 번호도 읽기 어렵기 때문에 `MIN_AREA`로 제거합니다.
"""),
        code(r"""
# K는 색상 단순화 정도, MIN_AREA는 번호를 넣을 최소 영역 크기입니다.
K = 10
MIN_AREA = 160

# 최종 기본 흐름: K-Means로 단순화한 뒤 Canny 선화를 만듭니다.
quantized, palette, label_map = kmeans_quantization_with_labels(image, K)
edges = clean_edges(hybrid_canny_color_edges(quantized, 60, 150, label_map), close_iter=1, thickness=2)
line_image = coloring_line_image(edges)

# Connected Components는 검은 선으로 분리된 흰 영역을 고유 영역 ID로 구분합니다.
(region_map, regions), cc_time = timed_call(segment_connected_components, line_image, MIN_AREA)

# 컬러링북에 들어가는 숫자는 영역 ID가 아니라 K-Means 팔레트 색상 번호입니다.
# 따라서 같은 색이 떨어진 여러 영역에 있어도 같은 숫자가 표시됩니다.
# 테두리에서 배경 RGB 색을 추정하고, Lab 색상 거리가 가까운 영역은 배경 번호로 병합합니다.
background_color = estimate_background_color(quantized)
regions = assign_region_color_numbers(
    regions,
    region_map,
    label_map,
    palette,
    background_color=background_color,
    background_color_threshold=16,
    merge_background_similar=True,
)
numbered_regions = colorable_regions(regions)
region_preview = color_region_preview(region_map)
color_edge_preview = color_region_edge_preview(line_image, region_map, regions, palette, thickness=4)

# 각 닫힌 영역 내부에서 가장 넓은 지점에 색상 번호를 넣습니다.
# 예시 이미지처럼 같은 색상 번호가 여러 영역에 반복해서 표시됩니다.
numbered = label_regions(line_image, regions, font_scale=0.9, region_map=region_map)
color_index = save_color_index_table(palette, "outputs/03_color_index_table.png", regions, "Segmentation Color Index")
numbered_with_index = combine_with_color_index(
    numbered,
    palette,
    regions,
    "outputs/03_numbered_with_color_index.png",
    "Segmentation Color Index",
)

# Contour와 Watershed는 비교용 결과로 함께 확인합니다.
contour_preview, contours = contour_regions(line_image, MIN_AREA)
watershed_preview, watershed_markers = watershed_segmentation(quantized)

save_image_rgb("outputs/03_line_image.png", line_image)
save_image_rgb("outputs/03_region_preview.png", region_preview)
save_image_rgb("outputs/03_color_edge_preview.png", color_edge_preview)
save_image_rgb("outputs/03_numbered_coloringbook.png", numbered)
save_image_rgb("outputs/03_contour_preview.png", contour_preview)
save_image_rgb("outputs/03_watershed_preview.png", watershed_preview)

show_images([
    ("Line Image", line_image),
    ("Connected Components Regions", region_preview),
    ("Color Edge Preview", color_edge_preview),
    ("Contour Detection Preview", contour_preview),
    ("Watershed Preview", watershed_preview),
    ("Numbered Coloring Book", numbered),
    ("Color Index Table", color_index),
], cols=2, figsize=(12, 11), cmap="gray", save_path="outputs/03_segmentation_compare.png")
"""),
        md("""
## 영역 중심 좌표와 색상 번호 정보

아래 표에서 `region_id`는 분리된 영역의 고유 번호이고, `color_id`는 실제 컬러링북에 표시되는 팔레트 번호입니다. `background_distance`가 작아 `is_background=True`인 영역은 배경색과 비슷하다고 판단되어 배경 번호로 병합됩니다.
"""),
        code(r"""
print(f"Estimated background RGB: {tuple(int(v) for v in background_color)}")

region_rows = []
for region in regions[:30]:
    # 발표 표가 너무 길어지지 않도록 앞 30개 영역만 출력합니다.
    cx, cy = region["centroid"]
    region_rows.append({
        "region_id": region["id"],
        "color_id": region["color_id"],
        "color_rgb": region["color_rgb"],
        "background_distance": region["background_distance"],
        "is_background": region["is_background"],
        "area": region["area"],
        "center_x": round(cx, 1),
        "center_y": round(cy, 1),
        "bbox": region["bbox"],
    })
print_table(region_rows)
print(f"Total segmented regions: {len(regions)}")
print(f"Numbered regions: {len(numbered_regions)}")
"""),
        md("""
## 영역 지표 및 HCI 가독성

평균 영역 크기가 너무 작으면 사용자가 색칠하기 어렵고, 작은 영역 개수가 많으면 번호 인식도 어려워집니다.
"""),
        code(r"""
metrics = region_metrics(numbered_regions, image.shape, small_area=300)
summary_rows = [
    {"metric": "Numbered Regions", "value": metrics["regions"]},
    {"metric": "Background-like Regions", "value": sum(1 for region in regions if region.get("is_background"))},
    {"metric": "Average Region Area", "value": metrics["average_area"]},
    {"metric": "Small Regions (<300 px)", "value": metrics["small_regions"]},
    {"metric": "Region Coverage", "value": metrics["region_coverage"]},
    {"metric": "Runtime", "value": cc_time},
]
print_table(summary_rows)
"""),
        md("""
## 비교 분석

- Connected Components: 닫힌 흰 영역을 직접 분리하므로 색상 번호 삽입 위치를 계산하기 좋습니다.
- Contour Detection: 외곽선 확인과 시각화에 좋지만 내부 구멍, 계층 구조 처리가 추가로 필요할 수 있습니다.
- Watershed: 복잡한 객체 분리에 유용하지만 컬러링북에서는 과분할이 발생하기 쉽습니다.
- 색상 번호화: 영역마다 고유 번호를 붙이는 것이 아니라, 각 영역의 dominant K-Means 색상 번호를 표시합니다.
- 배경 번호 병합: 테두리에서 추정한 배경 RGB와 Lab 색상 거리가 가까운 영역은 가운데에 끼어 있어도 배경 번호로 표시합니다.
- 번호 배치: 각 닫힌 영역 내부에서 가장 넓은 지점을 찾아 색상 번호를 넣습니다.
"""),
    ]
    write_notebook("03_segmentation.ipynb", cells)


def final_pipeline_nb():
    cells = [
        md("""
# 04. Final Coloring Book Pipeline

이 노트북은 전체 시스템을 하나로 연결합니다.

입력:

- `data/ani.png` 입력 이미지
- 원하는 색상 개수 K
- 경계선 Canny 임계값
- 선 두께
- 최소 영역 크기

출력:

1. 원본 이미지
2. K-Means 결과
3. Posterization 결과
4. Sobel 결과
5. Laplacian 결과
6. Canny 결과
7. 영역 분리 결과
8. 색상 번호가 들어간 최종 컬러링북 이미지
9. RGB 색상표
10. 성능 비교 표
"""),
        code(FINAL_IMPORTS),
        md("""
## 전체 파라미터

실험 발표에서는 `K=5, 10, 20`을 비교하고, 최종 결과는 보통 `K=10`부터 조정하는 방식이 좋습니다.
"""),
        code(r"""
# 최종 결과의 기본 색상 수입니다. 보고서에서는 K=5, 10, 20 비교도 함께 사용합니다.
K = 10

# Canny 임계값입니다. 값이 높으면 노이즈는 줄지만 필요한 경계가 사라질 수 있습니다.
CANNY_LOW = 60
CANNY_HIGH = 150

# 선 두께가 두꺼울수록 인쇄 가독성은 좋아지고 색칠 공간은 줄어듭니다.
LINE_THICKNESS = 2

# 너무 작은 영역은 번호가 읽기 어렵기 때문에 제거합니다.
MIN_REGION_AREA = 160
"""),
        md("""
## 최종 파이프라인 실행
"""),
        code(r"""
# 1. 색상 단순화: 최종 기본 알고리즘은 K-Means입니다.
(kmeans_img, kmeans_palette, kmeans_labels), kmeans_time = timed_call(kmeans_quantization_with_labels, image, K)

# Posterization은 속도 비교용 보조 알고리즘입니다.
(poster_img, poster_palette), poster_time = timed_call(posterization, image, K)

# 2. 경계선 추출 알고리즘 비교: 여러 알고리즘을 같은 입력에서 비교합니다.
(sobel_raw, sobel_time) = timed_call(sobel_edges, kmeans_img, 65)
(lap_raw, lap_time) = timed_call(laplacian_edges, kmeans_img, 25)
(canny_raw, canny_time) = timed_call(canny_edges, kmeans_img, CANNY_LOW, CANNY_HIGH)
(hybrid_raw, hybrid_time) = timed_call(hybrid_canny_color_edges, kmeans_img, CANNY_LOW, CANNY_HIGH, kmeans_labels)

# 흰 배경 + 검은 선 형태로 바꿔 컬러링북 출력 형식에 맞춥니다.
sobel_line = coloring_line_image(clean_edges(sobel_raw, open_iter=1, close_iter=1, thickness=1))
lap_line = coloring_line_image(clean_edges(lap_raw, open_iter=1, close_iter=1, thickness=1))
canny_edges_clean = clean_edges(hybrid_raw, open_iter=0, close_iter=1, thickness=LINE_THICKNESS)
canny_line = coloring_line_image(canny_edges_clean)

# 3. 영역 분리 및 번호화: Connected Components로 흰 영역을 분리합니다.
(region_map, regions), seg_time = timed_call(segment_connected_components, canny_line, MIN_REGION_AREA)
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
numbered_regions = colorable_regions(regions)
region_preview = color_region_preview(region_map)
color_edge_preview = color_region_edge_preview(canny_line, region_map, regions, kmeans_palette, thickness=4)
numbered_coloringbook = label_regions(canny_line, regions, font_scale=0.9, region_map=region_map)
color_index = save_color_index_table(kmeans_palette, "outputs/04_color_index_table.png", regions, "Final Color Index")
numbered_with_index = combine_with_color_index(
    numbered_coloringbook,
    kmeans_palette,
    regions,
    "outputs/04_numbered_with_color_index.png",
    "Final Color Index",
)

# 4. 저장: 단계별 결과를 outputs 폴더에 저장합니다.
save_image_rgb("outputs/04_original.png", image)
save_image_rgb("outputs/04_kmeans.png", kmeans_img)
save_image_rgb("outputs/04_posterization.png", poster_img)
save_image_rgb("outputs/04_sobel.png", sobel_line)
save_image_rgb("outputs/04_laplacian.png", lap_line)
save_image_rgb("outputs/04_canny.png", canny_line)
save_image_rgb("outputs/04_region_preview.png", region_preview)
save_image_rgb("outputs/04_color_edge_preview.png", color_edge_preview)
save_image_rgb("outputs/04_final_numbered_coloringbook.png", numbered_coloringbook)

show_images([
    ("1. Original", image),
    ("2. K-Means", kmeans_img),
    ("3. Posterization", poster_img),
    ("4. Sobel", sobel_line),
    ("5. Laplacian", lap_line),
    ("6. Hybrid Canny + Color Boundary", canny_line),
    ("7. Segmentation", region_preview),
    ("8. Color Edge Preview", color_edge_preview),
    ("9. Final Numbered Coloring Book", numbered_coloringbook),
    ("10. Color Index Table", color_index),
], cols=2, figsize=(13, 15), cmap="gray", save_path="outputs/04_final_results_grid.png")
"""),
        md("""
## RGB 색상표
"""),
        code(r"""
plot_palette(kmeans_palette, "Final K-Means RGB Palette", "outputs/04_final_palette.png")
"""),
        md("""
## 성능 비교 표
"""),
        code(r"""
metrics = region_metrics(numbered_regions, image.shape, small_area=300)
performance_rows = [
    {"stage": "K-Means", "runtime_sec": kmeans_time, "edge_density": "", "regions": "", "average_area": "", "small_regions": ""},
    {"stage": "Posterization", "runtime_sec": poster_time, "edge_density": "", "regions": "", "average_area": "", "small_regions": ""},
    {"stage": "Sobel", "runtime_sec": sobel_time, "edge_density": edge_density(255 - sobel_line), "regions": "", "average_area": "", "small_regions": ""},
    {"stage": "Laplacian", "runtime_sec": lap_time, "edge_density": edge_density(255 - lap_line), "regions": "", "average_area": "", "small_regions": ""},
    {"stage": "Canny", "runtime_sec": canny_time, "edge_density": edge_density(canny_raw), "regions": "", "average_area": "", "small_regions": ""},
    {"stage": "Hybrid Canny + Color Boundary", "runtime_sec": hybrid_time, "edge_density": edge_density(canny_edges_clean), "regions": "", "average_area": "", "small_regions": ""},
    {"stage": "Segmentation", "runtime_sec": seg_time, "edge_density": "", "regions": metrics["regions"], "average_area": metrics["average_area"], "small_regions": metrics["small_regions"]},
    {"stage": "Background-like Regions", "runtime_sec": "", "edge_density": "", "regions": sum(1 for region in regions if region.get("is_background")), "average_area": "", "small_regions": ""},
    {"stage": "Total Final", "runtime_sec": kmeans_time + canny_time + seg_time, "edge_density": edge_density(canny_edges_clean), "regions": metrics["regions"], "average_area": metrics["average_area"], "small_regions": metrics["small_regions"]},
]
print_table(performance_rows)
"""),
        md("""
## 색상 개수별 복잡도 변화
"""),
        code(r"""
# K가 증가할수록 색상 표현력, 영역 수, 실행 시간이 어떻게 변하는지 측정합니다.
complexity_rows = complexity_by_k(image, k_values=(5, 10, 20), min_area=MIN_REGION_AREA)
print_table(complexity_rows)

ks = [row["K"] for row in complexity_rows]
regions_by_k = [row["regions"] for row in complexity_rows]
runtime_by_k = [row["runtime_sec"] for row in complexity_rows]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(ks, regions_by_k, marker="o")
axes[0].set_title("K vs Number of Regions")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Regions")
axes[1].plot(ks, runtime_by_k, marker="o", color="tab:red")
axes[1].set_title("K vs Runtime")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Seconds")
plt.tight_layout()
plt.savefig("outputs/04_complexity_by_k.png", dpi=160, bbox_inches="tight")
plt.show()
"""),
        md("""
## 자동 정리: 알고리즘 장단점

### 색상 단순화

- K-Means 장점: 이미지 색상 분포를 반영해 대표색을 찾으므로 색상 유지 품질이 좋습니다.
- K-Means 단점: 반복 계산이 필요해 Posterization보다 느립니다.
- Posterization 장점: 매우 빠르고 구현이 단순합니다.
- Posterization 단점: 실제 이미지 색상 분포와 무관하게 균등 분할하므로 부자연스러운 색상 계단이 생길 수 있습니다.
- Median Cut 장점: 색상 분포 범위를 재귀적으로 나누므로 교육용 설명과 비교 실험에 좋습니다.
- Median Cut 단점: 고품질 최적화는 K-Means보다 약하고 구현 방식에 따라 속도 차이가 큽니다.

### 경계선 추출

- Sobel 장점: 방향별 밝기 변화 설명이 쉽고 빠릅니다.
- Sobel 단점: 선이 두껍거나 질감 변화에 민감할 수 있습니다.
- Laplacian 장점: 모든 방향의 급격한 변화를 한 번에 감지합니다.
- Laplacian 단점: 노이즈와 작은 텍스처까지 경계로 잡는 경향이 있습니다.
- Canny 장점: 노이즈 제거와 이중 임계값을 포함해 선명하고 안정적인 경계를 제공합니다.
- Canny 단점: 밝기가 비슷한 서로 다른 색의 경계는 놓칠 수 있습니다.
- Hybrid Color Boundary 장점: Canny 결과에 색상 라벨 경계를 합쳐 겹친 색상 영역도 분리합니다.
- Hybrid Color Boundary 단점: K-Means 결과가 과분할되면 선이 많아져 복잡도가 올라갈 수 있습니다.

### 영역 분리

- Connected Components 장점: 닫힌 흰 영역을 직접 분리하고, 각 영역에 dominant 색상 번호를 넣기 쉬워 최종 산출물에 적합합니다.
- Connected Components 단점: 선이 끊기면 영역이 합쳐질 수 있어 Morphology 보정이 중요합니다.
- 배경 번호 병합 장점: 외곽에서 추정한 배경색과 비슷한 내부 빈 공간도 같은 배경 번호로 표시되어 색상표와 일관됩니다.
- Contour Detection 장점: 외곽선 시각화와 영역 형태 확인이 쉽습니다.
- Contour Detection 단점: 내부 계층 구조 처리가 필요할 수 있습니다.
- Watershed 장점: 겹친 객체 분리에 강합니다.
- Watershed 단점: 컬러링북에서는 과분할로 인해 사용자가 색칠하기 어려운 작은 영역이 늘 수 있습니다.

## 최종 알고리즘 선택 이유

최종 조합은 `K-Means + Hybrid Canny/Color Boundary + Connected Components/Contour Detection`입니다. K-Means는 원본 색상 보존과 색상 수 제어의 균형이 좋고, Hybrid 경계선은 밝기가 비슷한 색상 사이의 경계까지 보완하며, Connected Components는 색상 번호를 넣을 색칠 영역을 직접 계산하기 좋습니다.

## Trade-off 분석

- K 증가: 원본과 비슷해지지만 영역 수와 번호 수가 늘어 색칠 난이도가 올라갑니다.
- 선 두께 증가: 경계 가독성은 좋아지지만 색칠 공간이 줄고 작은 영역이 사라질 수 있습니다.
- 최소 영역 크기 증가: 번호 가독성은 좋아지지만 세부 묘사가 줄어듭니다.
- Canny 임계값 증가: 노이즈는 줄지만 필요한 경계가 누락될 수 있습니다.

## 발표용 비교 포인트

- 같은 이미지에서 K=5, 10, 20 결과를 나란히 보여주며 복잡도 변화를 설명합니다.
- Sobel, Laplacian, Canny의 Edge Density와 시각적 노이즈를 비교합니다.
- 영역 개수, 평균 영역 크기, 작은 영역 개수를 HCI 지표로 연결합니다.
- 최종 결과는 K-Means 색상표 번호와 영역 번호가 연결되므로, 같은 색상 영역은 같은 숫자로 칠할 수 있습니다.

## HCI 관점 개선점

- 색상 단순화로 사용자가 선택해야 할 색의 수를 줄였습니다.
- 검은 선/흰 배경 형태로 출력해 인쇄와 색칠에 적합하게 만들었습니다.
- 작은 영역 제거로 색칠 스트레스와 번호 혼잡을 줄였습니다.
- 중심 기반 번호 배치와 겹침 회피로 번호 인식 편의성을 높였습니다.
- 성능 표와 복잡도 그래프로 사용자 난이도와 알고리즘 파라미터의 관계를 설명할 수 있습니다.
"""),
    ]
    write_notebook("04_final_pipeline.ipynb", cells)


def main():
    os.makedirs(ROOT / "src", exist_ok=True)
    os.makedirs(ROOT / "data", exist_ok=True)
    os.makedirs(ROOT / "outputs", exist_ok=True)
    color_quantization_nb()
    edge_detection_nb()
    segmentation_nb()
    final_pipeline_nb()


if __name__ == "__main__":
    main()
