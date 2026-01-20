# 01. RAPIDS 실습 환경 준비하기

## 1. RAPIDS 소개

### RAPIDS란?
RAPIDS는 NVIDIA가 개발한 오픈소스 GPU 가속 데이터 분석 라이브러리다. pandas, scikit-learn과 동일한 API를 제공하여 최소한의 코드 변경으로 GPU 가속 성능을 얻을 수 있다.

### 핵심 용어
- **CUDA**: NVIDIA GPU에서 병렬 연산을 실행하기 위한 플랫폼
- **Compute Capability**: GPU의 기능 수준을 나타내는 버전 번호 (RAPIDS는 7.0 이상 필요)
- **cuDF**: GPU 가속 pandas
- **cuML**: GPU 가속 scikit-learn

### 요구사항 요약

| 항목 | 요구사항 |
|------|----------|
| GPU | Compute Capability 7.0 이상 (RTX 20/30/40, GTX 16 시리즈) |
| 드라이버 | CUDA 12.x: 525.60+, CUDA 13.x: 580.65+ |
| OS | Linux, WSL2 (Windows는 WSL2 필수) |

---

## 2. 설치 방법 선택 가이드

### 어떤 방법을 선택해야 할까?

```
GPU가 있나요?
├─ 없음 → Colab 또는 Kaggle (4장 참고)
└─ 있음 → 환경 격리가 필요한가요?
          ├─ 예 → Docker (5.2 참고)
          └─ 아니오 → Conda (5.1 참고)
```

### 방법별 비교

| 방법 | 장점 | 단점 | 추천 대상 |
|------|------|------|----------|
| **Colab** | 설치 불필요, 무료 GPU | 세션 12시간 제한 | 입문자, 빠른 테스트 |
| **Kaggle** | 설치 불필요, T4 GPU x2 | 주당 사용 시간 제한 | 입문자, 대회 참가 |
| **Docker** | 환경 격리, 재현성 | Docker 학습 필요 | 팀 프로젝트, 프로덕션 |
| **Conda** | 유연한 패키지 관리 | 환경 충돌 가능성 | 개인 연구, 커스터마이징 |

> **입문자 추천 순서**: Colab → Kaggle → Docker → Local Conda

---

## 3. 내 환경 확인하기

### nvidia-smi 실행

터미널에서 `nvidia-smi`를 실행하면 GPU 정보를 확인할 수 있다:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.102.01             Driver Version: 581.57         CUDA Version: 13.0     |
|                                          ^^^^^^^^^^^^             ^^^^^^^^^^^^^^^^^^    |
|                                          (1) 드라이버 버전          (2) CUDA 버전          |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060        On  |   00000000:01:00.0  On |                  N/A |
|      ^^^^^^^^^^^^^^^^^^^^^^^^^          |                        |                      |
|      (3) GPU 모델명                      |                        |                      |
|  0%   50C    P8            N/A  /  115W |    3419MiB /   8188MiB |      5%      Default |
|       ^^^                               |    ^^^^^^^^^^^^^^^^^^  |                      |
|       (4) 온도                           |    (5) VRAM 사용량      |                      |
+-----------------------------------------+------------------------+----------------------+
```

### 확인해야 할 항목

| 번호 | 항목 | 확인 사항 |
|------|------|----------|
| (1) | Driver Version | CUDA 12.x: 525.60+, CUDA 13.x: 580.65+ |
| (2) | CUDA Version | RAPIDS 설치 시 이 버전 이하로 지정 |
| (3) | GPU Name | RTX 20/30/40, GTX 16 시리즈면 지원 |
| (5) | VRAM | 여유 있어야 Out of Memory 방지 |

### GPU 지원 여부 빠른 확인

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

출력 예시:
```
name, compute_cap
NVIDIA GeForce RTX 4060, 8.9
```

Compute Capability **7.0 이상**이면 RAPIDS 사용 가능.

### RAPIDS 지원 CUDA 버전 (2025년 1월 기준)

| CUDA 버전 | 지원 여부 | 최소 드라이버 |
|-----------|----------|--------------|
| CUDA 11.x | 지원 종료 | - |
| CUDA 12.x | 지원 | 525.60.13+ |
| CUDA 13.x | 지원 | 580.65.06+ |

> 최신 정보: https://docs.rapids.ai/install/

---

## 4. 클라우드 환경

### 4.1 Google Colab

Colab은 Google이 제공하는 무료 Jupyter Notebook 환경이다. cuDF가 기본 포함되어 있다.

**GPU 런타임 설정:**
1. 상단 **런타임** → **런타임 유형 변경**
2. **T4 GPU** 선택 후 저장

**RAPIDS 사용:**
```python
# cuDF는 기본 설치됨
import cudf
gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(gdf)
```

**cuML 등 추가 라이브러리 설치:**
```python
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py
```

```python
from cuml.linear_model import LinearRegression
# 이제 cuML 사용 가능
```

### 4.2 Kaggle

Kaggle Notebook에는 RAPIDS가 기본 포함되어 있다.

**GPU 설정:**
1. Notebook 우측 **Settings** → **Accelerator** → **GPU T4 x2** 선택

**RAPIDS 사용:**
```python
import cudf
cudf.__version__
```

> 최신 버전 사용: Settings → Environment → "Always use latest environment"

### 4.3 NVIDIA AI Workbench

NVIDIA AI Workbench는 복잡한 환경 설정을 단일 플랫폼으로 간소화하는 도구다.

- 공식 문서: https://docs.nvidia.com/workbench/
- 요구사항: WSL (Windows), Docker, NVIDIA 계정

> 상세 설정은 공식 문서 참고

---

## 5. 로컬 환경

### 5.1 Conda로 설치 (권장)

#### Miniforge 설치

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

#### RAPIDS 설치

본인 환경에 맞게 `cuda-version`을 수정:

```bash
# RAPIDS 환경 생성 (CUDA 12.5 예시)
conda create -n rapids-24.08 \
    -c rapidsai -c conda-forge -c nvidia \
    rapids=24.08 python=3.11 cuda-version=12.5

# 환경 활성화
conda activate rapids-24.08
```

**옵션 설명:**
- `-c rapidsai -c conda-forge -c nvidia`: 패키지 채널
- `rapids=24.08`: RAPIDS 버전
- `cuda-version=12.5`: nvidia-smi에서 확인한 버전 이하로 지정

#### 설치 확인

```python
python -c "import cudf; print(cudf.__version__)"
```

#### 시각화 라이브러리 추가

```bash
conda install -n rapids-24.08 -c conda-forge matplotlib seaborn
```

### 5.2 Docker로 설치

Docker를 사용하면 환경 충돌 없이 RAPIDS를 사용할 수 있다.

#### 요구사항
- Docker 설치
- NVIDIA Container Toolkit 설치
- NVIDIA GPU 드라이버

#### RAPIDS 컨테이너 실행

```bash
# 이미지 다운로드 및 실행
docker run --gpus all -it -p 8888:8888 \
    nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12
```

#### 로컬 디렉토리 마운트

```bash
docker run --gpus all -it -p 8888:8888 \
    -v $(pwd):/rapids/notebooks/host \
    nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12
```

| 옵션 | 설명 |
|------|------|
| `--gpus all` | 모든 GPU 사용 |
| `-p 8888:8888` | Jupyter Lab 포트 |
| `-v $(pwd):...` | 현재 디렉토리 마운트 |

### 5.3 WSL2 설정 (Windows)

Windows에서 RAPIDS를 사용하려면 WSL2가 필요하다.

#### WSL2 설치

```powershell
# PowerShell (관리자)
wsl --install Ubuntu-22.04
wsl --update
```

#### NVIDIA 드라이버 설치

```bash
# WSL 내부에서
sudo apt update
sudo apt install nvidia-driver-535
```

#### CUDA 설치

```bash
# CUDA for WSL 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get update
sudo apt-get -y install cuda
```

> 상세 가이드: https://learn.microsoft.com/en-us/windows/wsl/install

이후 5.1 Conda 또는 5.2 Docker 방법으로 RAPIDS 설치.

---

## 6. Quick Reference

### 환경 확인

```bash
nvidia-smi                    # GPU 및 드라이버 확인
nvcc --version                # CUDA Toolkit 버전
python -c "import cudf; print(cudf.__version__)"  # RAPIDS 버전
```

### Conda 환경 관리

```bash
conda create -n [이름] [패키지]    # 환경 생성
conda activate [환경명]           # 환경 활성화
conda deactivate                 # 환경 비활성화
conda env list                   # 환경 목록
```

### RAPIDS 설치 (한 줄)

```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia rapids=24.08 python=3.11 cuda-version=12.5
```

---

## 7. FAQ & 문제 해결

### Q1: nvidia-smi 실행 안 됨
**원인**: NVIDIA 드라이버 미설치
**해결**: https://www.nvidia.com/download/index.aspx 에서 드라이버 설치

### Q2: nvcc --version 실행 안 됨
**원인**: CUDA 미설치 또는 PATH 미설정
**해결**: `~/.bashrc`에 추가:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q3: RAPIDS 설치 시 버전 충돌
**원인**: CUDA와 RAPIDS 버전 불일치
**해결**: https://docs.rapids.ai/install/ 에서 호환 버전 확인

### Q4: WSL에서 GPU 인식 안 됨
**원인**: WSL1 사용 또는 드라이버 문제
**해결**:
```bash
wsl --set-version Ubuntu-22.04 2
```

### Q5: Out of Memory 에러
**원인**: GPU VRAM 부족
**해결**:
- 배치 크기 줄이기
- `nvidia-smi`로 다른 프로세스 확인 후 종료

---

## 추가 자료

- [RAPIDS 공식 문서](https://docs.rapids.ai/)
- [RAPIDS GitHub](https://github.com/rapidsai)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
