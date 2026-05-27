# 컬러링북 생성 파이프라인 개선기

## 고정 K에서 이미지별 자동 정책까지

처음 목표는 단순했다. 입력 이미지를 받아 edge를 검출하고, 닫힌 영역을 찾아 번호를 넣으면 paint-by-numbers 스타일의 컬러링북을 만들 수 있을 것이라고 생각했다.

하지만 실제 이미지를 넣어보니 단순한 edge detection만으로는 부족했다. 선은 자주 끊겼고, 그림자와 질감은 불필요한 경계로 살아남았으며, 사람이 색칠해야 하는 영역과 단순 표현선이 섞였다. 특히 컬러링북은 결과를 사람이 직접 보고 색칠해야 하기 때문에, 알고리즘이 보기에는 맞는 선이라도 사용자가 보기에는 혼란스러운 선이 될 수 있었다.

그래서 최종 파이프라인은 edge 중심 구조에서 벗어나 다음 방향으로 바뀌었다.

```text
edge detection 중심
-> object-first segmentation
-> 이미지별 K 자동 추천
-> 난이도별 결과 생성
-> segmentation/detail line 분리
-> connected component 기반 색칠 영역 생성
-> filled preview로 검증
-> 서비스용 자동 이미지 정책
```

현재 실제 결과 생성에는 `04_final_pipeline.ipynb`가 아니라 `server/scripts/batch_difficulty_pipeline.py`를 사용한다. 노트북은 실험과 설명용에 가깝고, batch 스크립트는 여러 이미지를 안정적으로 처리하기 위한 실행용 파이프라인이다.

---

## 1. 왜 Object-first 구조로 바꿨는가

초기에는 Canny edge처럼 픽셀 경계선을 먼저 찾고, 그 선을 기준으로 영역을 나누려고 했다.

하지만 이 방식은 몇 가지 문제가 있었다.

- 중요한 경계가 끊기면 색칠 영역도 같이 열린다.
- 그림자, 노이즈, 질감이 모두 edge로 잡힌다.
- 작은 디테일과 실제 색칠 경계가 구분되지 않는다.
- 닫힌 영역이 안정적으로 만들어지지 않는다.

그래서 접근을 바꿨다. 먼저 이미지를 K-Means로 색상 단순화하고, 그 label map에서 색 객체를 찾은 뒤, 객체의 외곽을 segmentation line으로 사용했다.

즉 현재 핵심 원칙은 다음이다.

```text
색칠 영역을 먼저 찾고,
그 영역의 경계를 컬러링북 선으로 사용한다.
```

이 구조를 사용하면 edge detection이 놓치는 부분이 있어도 색 영역 단위로 경계를 만들 수 있다. 또한 같은 색 영역이 하나의 객체처럼 묶이기 때문에, 단순 edge 기반 방식보다 컬러링북의 큰 구조를 안정적으로 만들 수 있다.

---

## 2. 고정 K의 문제

처음에는 K-Means의 색 개수를 고정했다.

```python
K = 30
```

실험 단계에서는 단순해서 좋았지만, 이미지가 달라지면 문제가 생겼다.

예를 들어 `doraemong`처럼 색이 적고 면이 단순한 이미지는 K=30이 너무 컸다. 색상 수가 실제로 많지 않은데도 K를 억지로 늘리면, 알고리즘은 새로운 색을 찾는 대신 JPEG 노이즈, 안티앨리어싱, 배경의 미세한 색 차이를 색 영역처럼 나누기 시작했다.

반대로 `ych`나 `flowers`처럼 디테일이 많은 이미지는 K가 너무 작으면 작은 캐릭터의 눈, 턱선, 장식 같은 부분이 하나의 큰 영역에 묻혀 버렸다.

그래서 K는 고정값이 아니라 이미지마다 자동으로 정해야 했다.

---

## 3. K 자동 추천 방식

K는 원본 이미지가 아니라 K-Means 전에 스무딩된 이미지를 기준으로 계산한다.

현재 흐름은 다음과 같다.

```text
원본 이미지
-> Mean Shift + Bilateral smoothing
-> coarse RGB 색상 그룹 수 계산
-> Lab 색 퍼짐도 계산
-> 최소 K 추천
-> 난이도별 K 생성
```

핵심 함수는 다음이다.

```python
def estimate_min_k_from_smoothed_colors(
    image,
    max_sample=40000,
    rgb_bin_size=16,
):
```

스무딩된 이미지라도 픽셀을 그대로 고유색으로 세면 anti-aliasing 때문에 색 수가 너무 크게 나온다. 그래서 RGB를 coarse bin으로 묶는다.

```python
coarse_rgb = pixels // 16
coarse_unique_colors = len(np.unique(coarse_rgb, axis=0))
```

그리고 Lab 색 공간에서 전체 색 분포가 얼마나 퍼져 있는지도 함께 본다.

```python
lab_spread = mean(distance(pixel_lab, mean_lab))
```

최종 최소 K 추정식은 다음에 가깝다.

```python
estimated = ceil(0.9 * log2(coarse_unique_colors) + lab_spread / 14.0)
min_k = clip(estimated, 3, 40)
```

이렇게 하면 단순히 색 개수만 보는 것이 아니라, 실제 이미지의 색 다양도와 색 분포를 함께 고려할 수 있다.

---

## 4. 난이도별 K 생성

최소 K를 구한 뒤에는 난이도별로 K를 만든다.

기본 정책은 다음과 같다.

```text
easy   = minK
normal = minK + 10
hard   = minK + 20
```

예를 들어 `ych`와 `landscape`는 다음처럼 나왔다.

```text
easy   K = 13
normal K = 23
hard   K = 33
```

하지만 모든 이미지에 이 증가폭을 그대로 적용하면 안 된다. 단순한 캐릭터 그림은 K를 크게 늘릴수록 질감과 배경 노이즈가 영역으로 분리된다.

그래서 현재는 이미지가 `simple` 정책으로 분류되면 더 작은 증가폭을 사용한다.

```text
easy   = minK
normal = minK + 4
hard   = minK + 8
```

`doraemong`은 이 정책에 따라 다음처럼 생성된다.

```text
easy   K = 10
normal K = 14
hard   K = 18
```

이 선택은 도라에몽만을 위한 예외 처리가 아니다. 파일명을 보는 것이 아니라 이미지의 색 다양도와 edge density를 보고 자동으로 결정한다.

---

## 5. 서비스용 이미지 자동 정책

서비스에서는 어떤 이미지가 들어올지 알 수 없다. 따라서 특정 이미지에 맞춘 하드코딩은 의미가 없다.

현재 파이프라인은 입력 이미지마다 다음 값을 계산한다.

- 원본 해상도
- edge density
- blur variance
- Lab 색 퍼짐도
- coarse Lab 색상 그룹 수

이를 바탕으로 이미지를 대략 다음 정책 중 하나로 분류한다.

```text
simple : 색이 적고 면이 단순한 이미지
high   : 작은 디테일과 edge가 많은 이미지
normal : 색 변화는 있지만 구조가 비교적 큰 이미지
soft   : 흐릿하거나 edge가 약한 이미지
```

예를 들어 최근 테스트에서는 다음처럼 분류되었다.

```text
doraemong -> simple
ych       -> high
landscape -> normal
flowers   -> high
```

이 정책은 K뿐 아니라 edge promotion의 민감도, 작업 해상도, morphology 연결 강도에도 영향을 준다.

내가 이 방향을 선택한 이유는 명확하다. 서비스 환경에서는 이미지를 미리 알 수 없기 때문에, 좋은 결과를 위해서는 고정 파라미터보다 이미지 프로파일 기반 정책이 필요하다.

---

## 6. 출력 해상도 정책

초기에는 입력 이미지를 자동으로 줄여서 처리했다. 예를 들어 원본이 `1440 x 900`이어도 작업 중에 `900 x 562` 정도로 줄어들었다.

하지만 컬러링북 결과물은 사람이 확대해서 보거나 출력할 수 있어야 한다. 작은 이미지에서 segmentation을 수행하면 눈, 턱선, 작은 장식 같은 영역이 사라지기 쉽다.

그래서 현재는 다음 정책을 사용한다.

```text
작은 원본  -> 작업 해상도를 일정 수준까지 키움
중간 원본  -> 원본 크기 유지
큰 원본    -> 처리 비용을 고려해 긴 변 상한 적용
최종 출력  -> SEGMENTATION_OUTPUT_SCALE = 2.0
```

현재 주요 결과는 다음과 같다.

```text
doraemong 원본 519x512  -> 작업 1100x1085 -> 출력 2200x2170
ych       원본 1440x900 -> 작업 1440x900  -> 출력 2880x1800
landscape 원본 5184x3456 -> 작업 1800x1200 -> 출력 3600x2400
```

이 방식은 품질과 처리 시간을 모두 고려한 타협이다. 무조건 원본 전체를 쓰면 `flowers`처럼 매우 큰 이미지는 품질은 좋지만 처리 시간이 길고 파일 크기도 커진다. 반대로 너무 줄이면 작은 segmentation 영역이 사라진다.

---

## 7. segmentation line과 detail line 분리

컬러링북에서 가장 중요한 원칙은 다음이다.

```text
segmentation line은 색칠 영역을 나누는 선
detail line은 표현을 위한 선
```

이 둘을 섞으면 사용자는 어떤 선 안쪽을 색칠해야 하는지 헷갈린다. 그래서 현재 렌더링에서는 역할을 분리한다.

```text
segmentation line : 검은색, 색칠 영역의 실제 경계
detail line       : 회색, 표정/장식/작은 묘사용 선
number text       : 연한 회색
```

이전에는 segmentation line과 detail line이 섞이면서, 어떤 선은 색칠 경계처럼 보이지만 실제로는 connected component에 반영되지 않는 문제가 있었다. 현재는 최종 출력에서 보이는 검은 segmentation line이 실제 connected component 기준과 최대한 일치하도록 조정했다.

---

## 8. 끊긴 선과 누락된 디테일 보정

실제 이미지에서는 K-Means label 경계만으로 충분하지 않은 경우가 많았다.

예를 들어 다음 문제가 있었다.

- 도라에몽 한쪽 눈의 눈동자/눈 안쪽 선이 약하게 잡힘
- ych의 호박 눈, 작은 캐릭터 턱선이 segmentation으로 잘 안 잡힘
- landscape에서 작은 흰 구멍 같은 미배정 영역이 보임
- easy 결과에서 segmentation line이 끊겨 색칠 영역이 불안정함

이를 해결하기 위해 여러 edge source를 조합했다.

```text
object_first_edges
source_detail_boundary_edges
dark_detail_region_edges
closed_detail_shape_edges
```

역할은 각각 다르다.

- `object_first_edges`: K-Means label map에서 큰 색 객체 경계를 만든다.
- `source_detail_boundary_edges`: 원본에서 중요한 색/밝기 경계를 보강한다.
- `dark_detail_region_edges`: 눈, 입, 작은 검은 영역처럼 compact한 dark detail을 잡는다.
- `closed_detail_shape_edges`: 거의 닫힌 작은 shape를 morphology로 닫아 segmentation 후보로 승격한다.

이후에는 morphology close를 사용해 작은 선 끊김을 이어준다.

```text
edge 후보 결합
-> morphology close
-> clean_edges
-> connected components
```

이 선택은 사용자가 실제로 색칠할 때 생기는 문제를 줄이기 위한 것이다. 선이 끊기면 컴퓨터 입장에서는 작은 오류일 수 있지만, 사람 입장에서는 어디까지 칠해야 하는지 모르는 문제가 된다.

---

## 9. 단순 이미지에서 질감이 segmentation으로 잡히는 문제

`doraemong`을 보면서 중요한 문제를 확인했다. 원래 색 수가 적은 이미지인데 K를 높이면 색상이 더 풍부해지는 것이 아니라 배경 질감, 압축 노이즈, 안티앨리어싱이 영역으로 분리되었다.

이 문제는 단순히 K를 낮추는 것만으로는 완전히 해결되지 않았다. KMeans가 흰 배경을 여러 near-white label로 쪼개면, 그 label 경계가 segmentation line으로 살아날 수 있었다.

그래서 `simple` 정책에서는 두 가지를 함께 적용했다.

```text
1. 난이도별 K 증가폭 제한
2. 배경과 비슷한 palette label을 segmentation 전에 병합
```

배경 유사 label 병합은 다음 의도를 가진다.

```text
흰 배경의 미세한 색 차이
-> 같은 배경 label로 병합
-> 불필요한 배경 섬 제거
```

이 방식도 도라에몽 전용 처리가 아니다. 색 다양도와 edge density가 낮은 flat artwork에 일반적으로 적용된다.

---

## 10. Connected Components와 filled preview

최종적으로 색칠 영역은 선 이미지의 반전 이미지에서 connected components로 구한다.

```text
segmentation line image
-> invert
-> connected components
-> region_map
-> 번호 배정
```

여기서 중요한 점은 최종 출력의 검은 선과 connected component 계산에 쓰이는 선이 일치해야 한다는 것이다. 사용자가 보는 선과 알고리즘이 생각하는 경계가 다르면, 컬러링북으로서 신뢰하기 어렵다.

그래서 현재는 최종 PNG와 함께 항상 filled preview를 저장한다.

```text
*_segmentation_filled.png
```

filled preview는 segmentation대로 색을 채웠을 때의 모습이다. 이 이미지를 보면 다음을 빠르게 확인할 수 있다.

- 영역이 새는지
- 닫히지 않은 경계가 있는지
- 작은 흰 구멍이 남는지
- 색칠 영역이 지나치게 잘게 쪼개졌는지
- detail line이 segmentation처럼 잘못 작동하는지

최근 확인용 결과는 다음 파일로 정리했다.

```text
server/0527output/three_images_policy_result_sheet.png
```

이 시트에서는 `doraemong`, `ych`, `landscape`의 easy/normal/hard 결과와 filled preview를 함께 볼 수 있다.

---

## 11. 현재 전체 파이프라인

현재 batch 파이프라인은 다음 순서로 동작한다.

```text
입력 이미지
-> 이미지 프로파일 분석
-> 자동 processing policy 선택
-> 작업 해상도 결정
-> Mean Shift + Bilateral smoothing
-> 최소 K 추천
-> 난이도별 K 생성
-> K-Means quantization
-> simple 이미지일 경우 배경 유사 label 병합
-> 출력용 2배 업스케일
-> object-first segmentation edge 생성
-> source/detail/dark/closed edge 보강
-> morphology로 선 연결
-> connected components로 색칠 영역 생성
-> region별 색 번호 배정
-> 검은 segmentation line 렌더링
-> 회색 detail line overlay
-> 번호 배치
-> 최종 컬러링북 PNG 저장
-> segmentation filled preview 저장
-> batch summary CSV 저장
```

실행 예시는 다음과 같다.

```powershell
python server\scripts\batch_difficulty_pipeline.py `
  --data-dir server\data `
  --output-dir server\0527output `
  --max-size 1800
```

특정 이미지만 처리하고 싶을 때는 임시 폴더에 해당 이미지만 넣고 실행하면 된다.

---

## 12. 현재 주요 설정

```python
SEGMENTATION_OUTPUT_SCALE = 2.0

PRE_KMEANS_MEAN_SHIFT_ENABLED = True
MEAN_SHIFT_SPATIAL_RADIUS = 12
MEAN_SHIFT_COLOR_RADIUS = 28

SIMPLIFY_DIAMETER = 9
SIMPLIFY_SIGMA_COLOR = 90
SIMPLIFY_SIGMA_SPACE = 90

DIFFICULTY_MARGINS = {
    "easy": 0,
    "normal": 10,
    "hard": 20,
}

AUTO_MIN_WORKING_LONG_EDGE = 1100
AUTO_DEFAULT_WORKING_LONG_EDGE = 1800
AUTO_MAX_WORKING_LONG_EDGE = 2400

AUTO_SIMPLE_K_UPPER_BOUND = 18
```

단순 이미지의 경우에는 difficulty margin이 별도로 줄어든다.

```text
simple image:
easy   +0
normal +4
hard   +8
```

---

## 13. 결과 예시

최근 세 이미지에 대해 같은 자동 정책을 적용한 결과는 다음과 같다.

```text
doraemong -> simple, K 10/14/18, 출력 2200x2170
ych       -> high,   K 13/23/33, 출력 2880x1800
landscape -> normal, K 13/23/33, 출력 3600x2400
```

결과 파일은 다음 위치에 저장된다.

```text
server/0527output/doraemong_easy_k10.png
server/0527output/doraemong_normal_k14.png
server/0527output/doraemong_hard_k18.png

server/0527output/ych_easy_k13.png
server/0527output/ych_normal_k23.png
server/0527output/ych_hard_k33.png

server/0527output/landscape_easy_k13.png
server/0527output/landscape_normal_k23.png
server/0527output/landscape_hard_k33.png
```

그리고 각 결과에 대응하는 filled preview도 함께 저장된다.

```text
*_segmentation_filled.png
```

---

## 14. 트러블슈팅 정리

### 문제: edge가 끊겨 색칠 영역이 열림

처음에는 edge detection 결과를 그대로 사용해서 선이 자주 끊겼다.

해결은 morphology close와 object-first edge를 함께 사용하는 것이었다. 색 영역 기반 경계를 먼저 만들고, 필요한 source detail만 보강했다.

### 문제: detail line과 segmentation line이 섞임

눈, 입, 장식선 같은 detail line이 색칠 경계처럼 보이면 사용자는 혼란스럽다.

해결은 두 선의 역할을 분리하는 것이었다. segmentation line은 검은색 경계로, detail line은 회색 overlay로 표현했다.

### 문제: K가 높을수록 품질이 좋아지는 것이 아님

단순 이미지에서는 K를 올리면 색 표현이 풍부해지는 것이 아니라 노이즈가 영역으로 분리되었다.

해결은 simple policy를 두고 K 증가폭을 제한하는 것이었다.

### 문제: 작은 디테일이 segmentation으로 잡히지 않음

ych의 호박 눈, 작은 캐릭터 턱선처럼 중요한 작은 영역이 누락되었다.

해결은 dark detail과 closed detail edge를 segmentation 후보로 승격시키는 것이었다.

### 문제: 큰 이미지는 품질이 좋지만 처리 비용이 큼

flowers처럼 큰 원본은 작은 영역까지 잘 잡히지만, 처리 시간이 길고 결과 파일도 커진다.

해결은 서비스용 작업 해상도 상한을 두는 것이었다. 현재는 `--max-size`로 운영 상한을 줄 수 있다.

### 문제: 결과가 실제로 닫힌 영역인지 알기 어려움

라인 이미지만 보면 어떤 영역이 실제로 connected component로 잡혔는지 확인하기 어렵다.

해결은 filled preview를 함께 저장하는 것이었다. segmentation대로 색을 채운 이미지를 보면 누락, leak, 작은 구멍을 빠르게 확인할 수 있다.

---

## 15. 내 의견과 결론

이번 개선에서 가장 중요하게 느낀 점은, 컬러링북 생성은 단순한 edge detection 문제가 아니라는 것이다.

컴퓨터 비전 관점에서는 선을 많이 찾는 것이 좋아 보일 수 있다. 하지만 실제 컬러링북 관점에서는 선이 많다고 좋은 결과가 아니다. 사용자는 어떤 선을 기준으로 색칠해야 하는지 알아야 하고, 영역은 적절히 닫혀 있어야 하며, 번호는 읽을 수 있어야 한다.

그래서 최종적으로 가장 중요한 원칙은 다음이라고 생각한다.

```text
색칠 영역을 나누는 선과 표현 디테일은 역할이 다르다.
```

segmentation line은 실제 색칠 영역과 번호 배치의 기준이 되어야 한다. detail line은 그림의 표정과 분위기를 살리는 보조선이어야 한다. 이 둘이 섞이면 결과물은 복잡해 보일 수는 있어도 사용하기 좋은 컬러링북은 되기 어렵다.

또 하나의 중요한 결론은 고정 파라미터의 한계다. 서비스에서는 입력 이미지가 정해져 있지 않기 때문에, 특정 샘플에 맞춘 튜닝만으로는 안정적인 품질을 만들 수 없다. 그래서 현재 파이프라인은 이미지별로 색 다양도, edge density, 흐림 정도, 해상도를 분석하고 그에 맞게 K, 작업 해상도, detail promotion 강도를 바꾸도록 했다.

결과적으로 파이프라인은 다음 방향으로 정리되었다.

```text
고정 K
-> 이미지별 K

edge 중심
-> object-first segmentation

단일 결과
-> easy/normal/hard 난이도별 결과

라인 결과만 저장
-> filled preview로 segmentation 검증

샘플별 튜닝
-> 서비스용 자동 정책
```

아직 완벽한 알고리즘은 아니다. 어떤 이미지에서는 더 정교한 foreground/background 분리나 semantic segmentation이 필요할 수 있고, 너무 복잡한 이미지에서는 사람이 색칠하기 좋은 수준으로 단순화하는 별도 기준이 필요하다.

하지만 현재 구조는 이전보다 훨씬 서비스에 가까워졌다. 단순한 캐릭터, 복잡한 일러스트, 풍경 이미지가 각각 다른 정책으로 처리되고, 결과도 컬러링북 이미지와 filled preview로 함께 확인할 수 있다. 무엇보다 이제는 "선이 많이 나오는 결과"가 아니라 "사람이 색칠할 수 있는 결과"를 기준으로 파이프라인을 판단하게 되었다.

