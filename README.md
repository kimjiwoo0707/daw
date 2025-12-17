# 아동 음성 분석을 통한 SELSI 기준 표현 언어 발달 단계 분류
---

## Skills
<div align="left">

<!-- Python -->
<img src="https://img.shields.io/badge/Python-FDE68A?style=for-the-badge&logo=python&logoColor=white"/>

<!-- PyTorch -->
<img src="https://img.shields.io/badge/PyTorch-FECACA?style=for-the-badge&logo=pytorch&logoColor=white"/>

<!-- NumPy -->
<img src="https://img.shields.io/badge/NumPy-A7F3D0?style=for-the-badge&logo=numpy&logoColor=white"/>

<!-- scikit-learn -->
<img src="https://img.shields.io/badge/scikit--learn-BFDBFE?style=for-the-badge&logo=scikit-learn&logoColor=white"/>

<!-- Audio Processing -->
<img src="https://img.shields.io/badge/Audio_Processing-FBCFE8?style=for-the-badge&logo=audacity&logoColor=white"/>

<!-- Computer Vision -->
<img src="https://img.shields.io/badge/Computer_Vision-C7D2FE?style=for-the-badge&logo=opencv&logoColor=white"/>

<!-- Vision Transformer -->
<img src="https://img.shields.io/badge/Swin_Transformer-E9D5FF?style=for-the-badge&logo=google&logoColor=white"/>

<!-- Flask -->
<img src="https://img.shields.io/badge/Flask-BAE6FD?style=for-the-badge&logo=flask&logoColor=white"/>

</div>
---

## 📰 연구배경
아동의 언어 발달은 사회적·인지적 성장의 핵심 지표이며, 조기 진단 여부는 이후 학습 능력과 행동 발달에 큰 영향을 미친다. 국내에서는 SELSI가 표준 진단 도구로 널리 활용되지만 문항 기반 평가 특성상 전문가의 해석이 필요해 시간과 비용이 많이 든다는 한계가 있다. 또한 아동의 언어 발달과 관련된 기존 연구는 텍스트 기반 분석이 중심이어서 억양, 피치, 발화 속도 등 음성 신호의 중요한 정보를 충분히 반영하지 못한다. 
본 프로젝트는 이러한 한계를 보완하기 위해, 실제 아동 음성 데이터를 기반으로 SELSI 기준 표현 언어 발달 단계를 자동 분류하는 모델을 구축하는 것을 목표로 한다.

---

## 📰 데이터 구성

*SELSI 기반 표현 언어 발달 단계 분류를 위한 아동 자연 발화 음성 데이터셋* (직접 수집)

### 데이터 수집 기준
SELSI(영유아 언어발달 검사)는 생후 4~35개월 영유아의 언어 능력을 평가하기 위한 국내 표준화 도구이며,  
수용 언어와 표현 언어 두 영역으로 구성되어 있다.  

본 프로젝트에서는 **표현 언어 영역**에 해당하는 항목을 기반으로 데이터를 수집하였다.  
SELSI 문항 중 단편적 기준으로 판단하기 어려운 항목을 제외하고, **총 37개 문항**을 최종 활용하였다.

### 데이터 수집 방식
- SELSI 표현 언어 항목과 개월 수 기준에 맞는 영상·음성을 공개 플랫폼(YouTube, SNS 등)에서 수집  
- 아동 자연 발화 음성만 사용  
- 부모 음성이 포함된 경우 최대한 제거했으나 일부 구간은 완전 제거에 기술적 한계 존재  
- 모든 음성은 Mel-Spectrogram으로 변환하여 비식별화 처리 후 사용
- 데이터는 학술적 연구 목적에 한해 사용하며 상업적 용도는 포함하지 않음

### 클래스 구성 (총 5개 클래스)
ASHA(미국 언어청각협회)의 발달 기준을 참고하여 개월 수 기준으로 5개의 클래스 구간을 설정하였다.

| Class | Age Range (months) | Number of Samples | Avg Length (s) |
|-------|---------------------|-------------------|----------------|
| 0     | 0–11               | 147               | 2.06           |
| 1     | 12–17              | 151               | 1.44           |
| 2     | 18–23              | 150               | 1.46           |
| 3     | 24–29              | 150               | 2.33           |
| 4     | 30–35              | 150               | 2.48           |

총 5개 클래스, **748개 음성 데이터**로 구성된 데이터셋을 구축하였다.  
아동 발달에 따라 평균 발화 길이가 점차 증가하는 경향을 확인할 수 있다.

---
## 📰 데이터 전처리

### 오디오 전처리 단계
1. **파일 형식 및 Sampling Rate 통일**
   - 모든 음성을 wav파일, SR 44kHz로 변환하여 모델 입력 형식 일관성 확보

3. **Noise Reduction (노이즈 제거)**  
   - 배경 소음, 부모 음성, 무음 구간 등을 최대한 제거  
   - 기본적인 필터링 및 에너지 기반 검출 적용

4. **Mel-Spectrogram 변환**  
   - 음성 신호를 2D 이미지 형태로 변환  
   - CNN/Transformer 모델의 입력에 적합하도록 처리  
   
5. **데이터셋 분리**  
   - Train / Test = **9 : 1** 비율로 구성

### 전처리 결과 예시
- Mel-Spectrogram 이미지 형태로 모델 입력 구성
- 클래스별 평균 발화 길이 및 스펙트럼 구조 차이 확인 가능


<img width="149" height="149" alt="image" src="https://github.com/user-attachments/assets/682284fe-fbf0-4f51-ac4d-e9f240f16197" />  

[Mel-Spectrogram 예시]


---
## 📰 모델 구조
### ResNet-34 기반 모델
- 이미지 분류 분야에서 널리 사용되는 대표적인 Residual Network  
- 기울기 소실 문제를 해결하기 위해 **Skip Connection**을 도입한 구조  
- 본 연구에서는 **사전학습(Pretrained) 가중치 적용** 후 학습 진행

### Swin-Transformer 기반 모델
- Vision Transformer(ViT)를 기반으로 한 **Shifted Window Attention** 구조
- 다양한 스케일의 시각 정보를 처리하기 위해 윈도우 단위 Self-Attention 활용
- 입력 특징: Mel-Spectrogram 이미지
- 핵심 구조:
  - Patch Partition으로 입력을 일정 크기 패치로 분할  
  - Linear Embedding을 통해 채널 수 조정
  - **W-MSA (Window-based Multi-Head Self-Attention)**
  - **SW-MSA (Shifted Window MSA)**
  - Patch Merging을 통해 계층적으로 축소된 피처 맵 생성 -> 특징 효과적으로 추출

---

## 모델 성능 평가
### 성능 비교 (ResNet-34 vs Swin-Transformer)

| Model            | Accuracy (%) | Precision (%) |
|------------------|--------------|----------------|
| **ResNet-34**        | 63.29        | 64.76          |
| **Swin-Transformer** | **68.42**    | **69.07**      |

Swin-Transformer는 ResNet-34보다 파라미터 수가 많고 모델이 더 무겁지만,  
Shifted Window 기반 구조를 통해 **다양한 스케일의 패턴을 효과적으로 학습**하는 장점을 가진다.  
이러한 구조적 특성 덕분에 Swin-Transformer는 Accuracy 기준 약 **5% 향상된 성능**을 기록했다.

---

### Swin-Transformer 최종 하이퍼파라미터

| Hyperparameter | Value                |
|----------------|----------------------|
| epoch          | 25                   |
| learning_rate  | 0.0001               |
| batch_size     | 32                   |
| optimizer      | AdamW                |
| scheduler      | CosineAnnealingLR    |


---

## 📰 AI Agent 기반 모델 개선 및 웹 서비스 구현

### AI Agent 기반 반복 학습(Feedback Loop)

본 프로젝트에서는 초기 모델의 데이터 수 제한과 성능 저하 문제를 보완하기 위해  
**AI Agent 기반 반복 학습(Feedback Loop)** 전략을 적용하였다.

- Agent가 모델 출력과 오류 구간을 자동 분석  
- 잘못 분류된 샘플에 대해 추가적인 Feature 보정 가이드 생성  
- 모델이 반영할 수 있도록 반복적으로 파라미터 및 Feature를 조정  
- 작은 데이터에서도 성능을 개선할 수 있는 효과 제공  

이 과정을 통해 Swin-Transformer모델 대비 F1-Score, Precision 약 1%, 6% 증가하였다.

---

### 웹 서비스 구현

최종 모델은 웹 환경에서 실제로 활용할 수 있도록 **Flask 기반 웹 페이지**로 구현하였다.

#### 주요 기능
- 음성 파일 업로드 기능 제공  
- 업로드된 아동 발화 음성을 실시간 전처리(Mel-Spectrogram 변환)  
- Swin-Transformer 기반 모델로 발달 단계 예측  
- 결과를 텍스트로 사용자에게 제공  

#### 기술 스택
- Backend: **Python Flask**  
- Frontend: **HTML / CSS / JavaScript**  

#### 구현 목적
- 모델 성능을 단순 수치로 끝내지 않고, **부모·교사·연구자가 쉽게 사용할 수 있는 형태로 제공하기 위함**  
- 실제 서비스 형태로 동작시키며 모델의 실용성과 활용 가능성을 검증

이 과정을 통해  **AI 모델 개발 → 반복 개선(AI Agent) → 실서비스 구현**으로 이어지는 완전한 End-to-End 워크플로우를 구축하였다.

---

## 📰 프로젝트 전체 플로우차트
<img width="1494" height="548" alt="image" src="https://github.com/user-attachments/assets/4e27e0a8-915f-4bbc-8190-f77799c5b141" />


## 📰 웹 서비스 구현 영상
[![▶ 영상 보기](https://github.com/user-attachments/assets/53f77497-570c-4058-8d7f-477f296373c3)](https://github.com/kimjiwoo0707/daw/raw/1c5840e76b2a0682a87a099ee94c868a3455b7dd/%EC%95%84%EB%8F%99%20%EC%9D%8C%EC%84%B1%20%EB%B6%84%EC%84%9D%20%EC%82%AC%EC%9D%B4%ED%8A%B8%20%EC%86%8C%EA%B0%9C%20%EC%98%81%EC%83%81.mp4)

