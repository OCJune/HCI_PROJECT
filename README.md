# 컬러링북 생성 시스템

Python OpenCV 기반 Jupyter Notebook 프로젝트입니다. 사용자가 이미지를 입력하고 원하는 색상 개수 `K`를 지정하면 원본 이미지를 색칠하기 쉬운 컬러링북 형태로 변환합니다.

## 구성

- `01_color_quantization.ipynb`: K-Means, Posterization, Median Cut 색상 단순화 비교
- `02_edge_detection.ipynb`: Sobel, Laplacian, Canny 경계선 추출 비교
- `03_segmentation.ipynb`: Connected Components, Contour, Watershed 영역 분리 및 번호화
- `04_final_pipeline.ipynb`: 최종 파이프라인, 성능 분석, HCI 평가 정리
- `src/coloringbook_utils.py`: 공통 함수
- `outputs/`: 단계별 결과 이미지 저장 위치
- `data/`: 입력 이미지 저장 위치

## 발표용 주석 요약

### 01. 색상 단순화

- `K-Means`: 원본 이미지의 픽셀 색상을 K개의 대표 색상으로 군집화합니다. 색상 보존력이 좋아 최종 기본 알고리즘으로 사용합니다.
- `Posterization`: RGB 값을 일정 구간으로 나눠 빠르게 색상을 줄입니다. 속도는 빠르지만 색상 경계가 다소 부자연스러울 수 있습니다.
- `Median Cut`: 색상 분포가 넓은 축을 반복적으로 나눠 팔레트를 만듭니다. 알고리즘 설명용 비교 대상으로 적합합니다.
- `K=5, 10, 20 비교`: K가 커질수록 원본과 비슷해지지만 색칠 복잡도와 영역 수가 증가합니다.

### 02. 경계선 추출

- `Sobel`: x/y 방향 밝기 변화량으로 경계를 찾습니다. 빠르고 설명이 쉽지만 선이 두껍거나 노이즈가 생길 수 있습니다.
- `Laplacian`: 2차 미분으로 급격한 변화 지점을 찾습니다. 세부 변화에 민감해 작은 노이즈가 늘 수 있습니다.
- `Canny`: 노이즈 제거와 이중 임계값을 함께 사용해 선명하고 안정적인 선화를 만듭니다. 최종 기본 알고리즘입니다.
- `Hybrid Color Boundary`: Canny가 밝기 차이에 의존해 놓치는 색상 경계를 보완합니다. K-Means 라벨이 바뀌는 지점도 선으로 추가하므로, 겹친 색의 밝기가 비슷해도 경계를 잡을 수 있습니다.
- `Morphology`: Opening은 작은 점 노이즈 제거, Closing은 끊긴 선 연결에 사용합니다.
- `Line Thickness`: 선이 두꺼울수록 가독성은 좋아지지만 색칠 공간은 줄어듭니다.

### 03. 영역 분리 및 번호화

- `Connected Components`: 검은 선으로 나뉜 흰 영역을 각각의 색칠 영역으로 라벨링합니다. 번호 삽입 위치 계산에 사용됩니다.
- `Contour Detection`: 영역 외곽선을 시각적으로 확인하기 좋습니다. Connected Components의 보조 비교용입니다.
- `Watershed`: 객체 분리에는 강하지만 컬러링북에서는 과분할이 발생할 수 있어 비교용으로 사용합니다.
- `MIN_AREA`: 너무 작은 영역은 색칠하기 어렵고 번호도 읽기 어려워 제거합니다.
- `Color Numbering`: 각 영역의 고유 ID를 출력하지 않고, 영역 내부에서 가장 많이 차지하는 K-Means 팔레트 번호를 출력합니다. 그래서 같은 색상은 떨어져 있어도 같은 숫자가 들어갑니다.
- `Background Label Merge`: 이미지 테두리에서 배경 RGB 색상을 추정합니다. 각 영역의 대표 색상과 배경색의 Lab 색상 거리가 가까우면, K-Means 라벨이 달라도 배경 번호로 병합해 라벨링합니다.
- `Labeling`: 영역 중심에 색상 번호를 배치하고, 겹치는 경우 주변 후보 위치로 이동합니다.

### 04. 최종 파이프라인

- 최종 조합은 `K-Means + Hybrid Canny/Color Boundary + Connected Components`입니다.
- 성능 지표는 Runtime, Edge Density, 영역 개수, 평균 영역 크기, 작은 영역 개수입니다.
- HCI 평가는 색칠 난이도, 경계선 가독성, 번호 인식 편의성, K 증가에 따른 복잡도 변화를 중심으로 설명합니다.

## 실행

```bash
pip install -r requirements.txt
jupyter notebook
```

각 노트북의 `IMAGE_PATH`에 원하는 이미지 경로를 넣으면 됩니다. 값을 비워두면 `data/sample_input.png` 샘플 이미지가 자동 생성됩니다.
HCI_PROJECT
