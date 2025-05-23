# Conditional DCGAN (CDCGAN)

MNIST 데이터셋을 사용한 조건부 이미지 생성을 위한 Conditional DCGAN 구현 프로젝트입니다.

## GAN의 발전 과정

### GAN (Generative Adversarial Network)
- 생성 모델의 기본 구조
- Generator와 Discriminator의 적대적 학습을 통한 이미지 생성
- Generator: 가짜 이미지 생성
- Discriminator: 진짜/가짜 이미지 구분

### DCGAN (Deep Convolutional GAN)
- GAN에 합성곱 신경망(CNN)을 도입한 모델
- 전치 합성곱(Transposed Convolution)을 사용한 이미지 생성
- 배치 정규화와 LeakyReLU 활성화 함수 도입
- 더 안정적인 학습과 고품질 이미지 생성 가능

### CDCGAN (Conditional DCGAN)
- DCGAN에 조건부 생성 기능을 추가한 모델
- 클래스 레이블을 조건으로 특정 클래스의 이미지 생성 가능
- Generator와 Discriminator 모두에 클래스 정보 주입
- MNIST 데이터셋의 경우 0-9까지의 숫자를 조건부로 생성

## 프로젝트 구조

```
.
├── model.py          # CDCGAN 모델 구현
├── dataset.py        # 데이터셋 로드 및 전처리
├── train.py          # 모델 학습 스크립트
├── test.py           # 모델 테스트 스크립트
├── gradio_test.py    # Gradio 웹 인터페이스
└── requirements.txt  # 프로젝트 의존성
```

## 설치 방법

1. 저장소 클론
```bash
git clone [https://github.com/gyumin4726/CDCGAN]
cd CDCGAN
```

2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습
```bash
python train.py
```

### 모델 테스트
```bash
python test.py
```

### 웹 인터페이스 실행
```bash
python gradio_test.py
```

## 주요 기능

- MNIST 데이터셋을 사용한 조건부 이미지 생성
- PyTorch Lightning 기반 모델 구현
- Gradio를 통한 웹 기반 인터페이스 제공
- 사용자 정의 잠재 벡터를 통한 이미지 생성 제어

## 기술 스택

- PyTorch
- PyTorch Lightning
- Gradio
- MNIST Dataset

## 모델 아키텍처

### Generator
1. **입력 처리**
   - 잠재 벡터(z): 5차원 → 50차원 (Linear)
   - 클래스 레이블(c): 10개 클래스 → 50차원 (Embedding)
   - 결합: 100차원 벡터 → 256×7×7 (Linear)

2. **레이어별 구조**
   ```
   입력: 256×7×7
   ↓
   Layer 1: 256×7×7 → 128×7×7
   - ConvTranspose2d(256→128, kernel=5, stride=1, padding=2)
   - BatchNorm2d + LeakyReLU
   - MaxPool2d(kernel=2, stride=1, padding=1)
   ↓
   Layer 2: 128×7×7 → 64×14×14
   - ConvTranspose2d(128→64, kernel=4, stride=2, padding=1)
   - BatchNorm2d + LeakyReLU
   - AvgPool2d(kernel=2, stride=1, padding=1)
   ↓
   Layer 3: 64×14×14 → 1×28×28
   - ConvTranspose2d(64→1, kernel=4, stride=2, padding=1)
   - Tanh
   ↓
   출력: 1×28×28 (CenterCrop)
   ```

### Discriminator
1. **입력 처리**
   - 이미지: 1×28×28
   - 클래스 레이블: 10개 클래스 → 1×28×28 (Embedding)
   - 결합: 2×28×28 (채널 차원에서 결합)

2. **레이어별 구조**
   ```
   입력: 2×28×28
   ↓
   Layer 1: 2×28×28 → 64×14×14
   - Conv2d(2→64, kernel=4, stride=2, padding=1)
   - LeakyReLU
   - Dropout(0.3)
   ↓
   Layer 2: 64×14×14 → 128×7×7
   - Conv2d(64→128, kernel=4, stride=2, padding=1)
   - LeakyReLU
   - Dropout(0.3)
   ↓
   Layer 3: 128×7×7 → 1
   - Flatten
   - Linear(128×7×7 → 1)
   - Sigmoid
   ↓
   출력: 진위 여부 (0~1 사이의 값)
   ```

## 하이퍼파라미터

- 잠재 벡터 차원: 5
  - Generator의 입력 차원으로 고정
  - 이미지 생성의 다양성을 제어하는 파라미터
  - 50차원으로 투영된 후 클래스 임베딩과 결합
- 배치 크기: 100
- 학습률: 0.0002
- Adam 옵티마이저 (beta1=0.5)
- 에포크 수: 3

## 주의사항

- 학습 시간은 하드웨어 사양에 따라 달라질 수 있습니다.
- 웹 인터페이스는 기본적으로 localhost:7860에서 실행됩니다.
- 메모리 사용량을 고려하여 `dataset.py`의 `batch_size`를 조정할 수 있습니다.

## 문제 해결

### 일반적인 문제
1. 메모리 부족 오류
   - `dataset.py`에서 `batch_size` 값을 줄임
2. Gradio 실행 오류
   - 포트가 이미 사용 중인 경우 `gradio_test.py`에서 포트 번호 변경
   ```python
   # gradio_test.py 마지막 줄을 다음과 같이 수정
   demo.launch(server_port=7861)  # 7860 대신 다른 포트 번호 사용
   ```
