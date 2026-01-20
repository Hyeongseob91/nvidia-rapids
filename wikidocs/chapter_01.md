# 01. NVIDIA RAPIDS 실습 환경 준비하기

이번 장에서는 RAPIDS를 설치하기 위한 환경을 준비하고 RAPIDS를 설치한다.

RAPIDS는 NVIDIA GPU의 CUDA를 사용해서 실행된다. 따라서 RAPIDS를 사용하기 위해서 필요한 것은 다음과 같다:

1. NVIDIA Volta 이상의 GPU (GeForce 20, Titan V 이상)
2. NVIDIA 드라이버
3. CUDA

### NVIDIA GPU 아키텍처란?

NVIDIA는 GPU를 세대별로 구분하며, 각 세대에 **아키텍처(Architecture)** 이름을 붙인다. 아키텍처는 GPU의 내부 설계 방식을 의미하며, 새로운 아키텍처가 나올수록 성능과 기능이 향상된다.

| 아키텍처 | 출시 연도 | 대표 제품 | Compute Capability |
|----------|----------|----------|-------------------|
| Kepler | 2012 | GTX 600/700 시리즈 | 3.0 - 3.7 |
| Maxwell | 2014 | GTX 900 시리즈 | 5.0 - 5.3 |
| Pascal | 2016 | GTX 10 시리즈 | 6.0 - 6.2 |
| **Volta** | **2017** | **Titan V, Tesla V100** | **7.0** |
| Turing | 2018 | RTX 20 시리즈, GTX 16 시리즈 | 7.5 |
| Ampere | 2020 | RTX 30 시리즈, A100 | 8.0 - 8.6 |
| Ada Lovelace | 2022 | RTX 40 시리즈 | 8.9 |
| Hopper | 2022 | H100 (데이터센터용) | 9.0 |

### Volta 아키텍처란?

**Volta**는 2017년에 발표된 NVIDIA의 GPU 아키텍처로, **딥러닝과 AI 연산에 최적화**된 최초의 아키텍처이다.

**Volta의 핵심 특징:**

- **Tensor Core 최초 도입**: AI/딥러닝 연산을 위한 전용 코어. 행렬 연산을 기존 대비 최대 12배 빠르게 처리
- **Compute Capability 7.0**: RAPIDS가 요구하는 최소 버전
- **HBM2 메모리**: 높은 대역폭의 메모리로 대용량 데이터 처리에 유리
- **NVLink 2.0**: GPU 간 고속 통신 지원

> RAPIDS가 Volta 이상을 요구하는 이유는 **Tensor Core**와 **향상된 CUDA 기능**을 활용하기 때문이다.

### Titan V란?

**NVIDIA Titan V**는 2017년 12월에 출시된 Volta 아키텍처 기반의 소비자용 최상위 GPU이다.

| 항목 | 사양 |
|------|------|
| 아키텍처 | Volta |
| CUDA 코어 | 5,120개 |
| Tensor 코어 | 640개 |
| 메모리 | 12GB HBM2 |
| 메모리 대역폭 | 652.8 GB/s |
| Compute Capability | 7.0 |
| 출시가 | $2,999 |

Titan V는 출시 당시 AI 연구자와 데이터 과학자를 위한 "슈퍼컴퓨터급 GPU"로 마케팅되었다.

### 내 GPU가 RAPIDS를 지원하는지 확인하기

터미널에서 `nvidia-smi` 명령어를 입력하면 다음과 같은 화면이 출력된다:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.102.01             Driver Version: 581.57         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060        On  |   00000000:01:00.0  On |                  N/A |
|  0%   50C    P8            N/A  /  115W |    3419MiB /   8188MiB |      5%      Default |
+-----------------------------------------+------------------------+----------------------+
```

#### 확인해야 할 핵심 정보

| 위치 | 항목 | 예시 값 | 설명 |
|------|------|---------|------|
| 상단 우측 | **CUDA Version** | `13.0` | 지원하는 CUDA 버전. RAPIDS 설치 시 이 버전 이하로 설치해야 함 |
| 상단 우측 | **Driver Version** | `581.57` | NVIDIA 드라이버 버전 |
| 중앙 | **GPU Name** | `NVIDIA GeForce RTX 4060` | GPU 모델명. 이걸로 RAPIDS 지원 여부 확인 |
| 중앙 | **Memory-Usage** | `3419MiB / 8188MiB` | 현재 사용 중인 VRAM / 전체 VRAM |

#### nvidia-smi 출력 해석 가이드

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.102.01             Driver Version: 581.57         CUDA Version: 13.0     |
|                                          ^^^^^^^^^^^^             ^^^^^^^^^^^^^^^^^^    |
|                                          ① 드라이버 버전            ② CUDA 버전          |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|      ^^^^                               |                        |                      |
|      ③ GPU 이름                          |                        |                      |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060        On  |   00000000:01:00.0  On |                  N/A |
|      ^^^^^^^^^^^^^^^^^^^^^^^^^          |                        |                      |
|      ③ 여기서 GPU 모델 확인!             |                        |                      |
|  0%   50C    P8            N/A  /  115W |    3419MiB /   8188MiB |      5%      Default |
|       ^^^                               |    ^^^^^^^^^^^^^^^^^^  |      ^^^             |
|       ④ 온도                             |    ⑤ VRAM 사용량        |      ⑥ GPU 사용률    |
+-----------------------------------------+------------------------+----------------------+
```

**① Driver Version (581.57)**
- NVIDIA 드라이버 버전
- 너무 오래된 버전이면 업데이트 필요

**② CUDA Version (13.0)**
- 이 GPU가 지원하는 **최대** CUDA 버전
- RAPIDS 설치 시 `cuda-version=12.5` 처럼 이 버전 **이하**로 지정

**③ GPU Name (NVIDIA GeForce RTX 4060)**
- GPU 모델명
- RTX 20/30/40 시리즈, GTX 16 시리즈면 RAPIDS 지원 ✅
- GTX 10 시리즈 이하면 RAPIDS 미지원 ❌

**④ Temp (50C)**
- GPU 온도. 80°C 이상이면 쿨링 점검 필요

**⑤ Memory-Usage (3419MiB / 8188MiB)**
- 현재 사용 중인 VRAM / 전체 VRAM
- RAPIDS 작업 시 VRAM이 부족하면 Out of Memory 에러 발생

**⑥ GPU-Util (5%)**
- GPU 사용률. 학습/연산 중에는 높아짐

#### 더 간단하게 확인하기

GPU 이름과 Compute Capability만 확인하고 싶다면:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

출력 예시:
```
name, compute_cap
NVIDIA GeForce RTX 4060, 8.9
```

> **RTX 4060의 Compute Capability는 8.9**로, RAPIDS가 요구하는 7.0 이상을 충족한다. ✅

#### NVIDIA 공식 사이트에서 확인

[CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus)에서 본인 GPU의 Compute Capability를 확인할 수 있다.

**RAPIDS 지원 GPU 예시 (Compute Capability 7.0 이상):**

| 시리즈 | 지원 GPU |
|--------|----------|
| GeForce | RTX 20/30/40 시리즈, GTX 16 시리즈 |
| Titan | Titan V, Titan RTX |
| Quadro | RTX 시리즈 |
| Tesla/데이터센터 | V100, A100, H100 |

> **주의**: GTX 10 시리즈(Pascal, Compute Capability 6.x)는 RAPIDS를 지원하지 않는다.

RAPIDS를 사용하는 방법은 여러 가지가 있다. 이번 장에서는 로컬 환경에 설치하는 방법뿐만 아니라, Colab이나 Kaggle 같은 온라인 플랫폼에서 RAPIDS를 사용하는 방법, 그리고 NVIDIA AI Workbench를 활용하는 방법까지 다양한 방법을 설명한다.

---

## 01-01-01. Windows 10, 11

### 1. NVIDIA 드라이버 설치

명령 프롬프트에서 `nvidia-smi` 실행이 안 된다면 NVIDIA 드라이버를 설치해야 한다.
[NVIDIA 드라이버 다운로드](https://www.nvidia.com/download/index.aspx)에서 그래픽 카드와 운영체제에 맞는 드라이버를 다운로드 받고 설치한다.

### 2. CUDA 설치

먼저 RAPIDS가 지원하는 CUDA 버전을 확인하자: https://docs.rapids.ai/install/

그다음 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)에서 현재 사용하는 운영체제에 맞는 CUDA Toolkit을 다운로드 받고 설치한다.

CUDA 설치가 완료되면 명령 프롬프트에서 아래 명령어를 실행할 수 있다.

```bash
nvcc --version
```

---

## 01-01-02. Ubuntu 20.04 이상

### 1. NVIDIA 드라이버 설치

보유한 NVIDIA GPU에 맞는 권장 드라이버를 자동 설치해보자.

```bash
sudo apt update
sudo ubuntu-drivers autoinstall
```

> `sudo apt update`는 Debian 기반 리눅스 시스템에서 사용하는 명령어로 시스템 패키지 관리자(APT)가 최신 패키지 정보를 가져오도록 업데이트한다. `sudo ubuntu-drivers autoinstall`는 Ubuntu에서 자동으로 하드웨어에 맞는 드라이버를 검색하고 설치하는 명령어이다.

설치가 완료되면 컴퓨터를 재부팅한다.

```bash
sudo reboot
```

이제 `nvidia-smi` 명령어를 실행하여 NVIDIA GPU와 관련된 정보를 확인할 수 있다.

```bash
nvidia-smi
```

### 2. CUDA 설치

먼저 RAPIDS가 지원하는 CUDA 버전을 확인하자.

Ubuntu에서 CUDA를 설치하는 방법은 다양하다. 우리는 runfile을 사용해서 CUDA를 설치한다.

다운 받은 runfile을 실행하고 엔터를 누르다 보면 End User License Agreement가 나온다. 아래와 비슷한 화면이 나올 때 키보드로 `accept`를 입력한다.

```
Do you accept the above EULA? (accept/decline/quit):
```

End User License Agreement(EULA) 동의(accept)하면 아래와 비슷한 화면이 나온다. 이때 CUDA Installer에서 Driver와 Driver 하위 항목을 제외하고 체크 표시를 한다. 그 후 Install을 선택한다.

```
CUDA Installer
- [ ] Driver
  [ ] 그래픽 드라이버 버전
+ [X] CUDA Toolkit 버전
  [X] CUDA Samples 버전
  [X] CUDA Demo Suite 버전
  ...이하 생략...

Options
Install
```

CUDA 설치가 완료되면 아래 명령어로 CUDA 버전을 확인할 수 있다.

```bash
nvcc --version
```

---

## 01-01-03. WSL과 CUDA

NVIDIA Volta™ 이상의 GPU로, 계산 능력이 7.0 이상인 그래픽 카드(16GB 이상의 GPU RAM 권장)가 설치된 윈도우 컴퓨터에서 RAPIDS를 설치하여 실행하려면 가상의 리눅스 환경이 필요하다.

### WSL2 추가 필수 조건

- **OS**: WSL2용 Ubuntu 22.04 인스턴스가 있는 Windows 10 이상
- **WSL 버전**: WSL2 (WSL1은 지원되지 않음)
- **GPU**: Compute Capability 7.0 이상의 GPU (16GB 이상의 GPU RAM 권장). ※ 단일 GPU만 지원됩니다.

### WSL 설치 및 업데이트

먼저 제어판 > Windows 기능 켜기/끄기에서 "Linux용 Windows 하위 시스템"을 켠다.

https://learn.microsoft.com/en-us/windows/wsl/install 참고하여 WSL을 설치하고 업데이트 한다.

```powershell
wsl --install Ubuntu-22.04
wsl --update
```

Microsoft Store에서 Ubuntu 22.04 LTS를 설치해도 된다.

### NVIDIA 드라이버 설치하기

Windows PowerShell(관리자로 실행)에서 WSL에 root로 들어가 암호 설정하고, NVIDIA 드라이버를 설치한 후 윈도우까지 재부팅 한다.

```bash
wsl -d Ubuntu-22.04 --user root
passwd
apt-get update
apt-get upgrade
apt install nvidia-driver-535
reboot
```

### CUDA 설치하기

```bash
wsl -d Ubuntu-22.04
cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

### CUDA-TOOLKIT 설치하기

```bash
sudo apt install nvidia-cuda-toolkit
sudo reboot
nvcc -V
```

### Miniconda 설치하기

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

`~/.bashrc`에 PATH 추가:

```bash
export PATH=/home/user/miniconda3/bin:$PATH
```

```bash
source ~/.bashrc
conda init
source ~/.bashrc
```

### RAPIDS 설치하기

```bash
conda create -n rapids-24.04 -c rapidsai -c conda-forge -c nvidia rapids=24.04 python=3.10 cuda-version=11.5 pytorch
conda activate rapids-24.04
python
>>> import cudf
>>> cudf.__version__
'24.04.01'
>>> import cuml
>>> cuml.__version__
'24.04.00'
>>> quit()
```

---

## 01-02-01. Local에 RAPIDS 설치하기

### 1. 설치 요구사항

RAPIDS를 Local에 설치하기 위한 요구사항은 다음과 같다:

- NVIDIA Volta™ 이상의 GPU로, 계산 능력이 7.0 이상인 GPU
- Ubuntu 20.04 또는 22.04, Rocky Linux 8, 또는 Windows 10, 11 WSL2
- 최신 CUDA 버전과 NVIDIA 드라이버
- 현재 설치된 버전은 `nvidia-smi` 명령어로 확인할 수 있음

### 2. Miniforge 설치

Miniforge는 Anaconda와 Miniconda의 대안으로, Conda 패키지 관리 도구와 Python 환경을 포함한 경량화된 배포판이다. Miniforge는 오픈 소스 소프트웨어만을 기본으로 포함하며, 사용자에게 최소한의 설치 환경을 제공한다.

Miniforge는 conda-forge라는 커뮤니티 기반의 패키지 저장소를 기본으로 사용하여, 다양한 오픈 소스 패키지에 대한 접근을 지원한다.

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 3. RAPIDS 설치

#### 3.1. RAPIDS 빠른 설치

RAPIDS 요구사항을 충족하고 CUDA를 설치했다면 RAPIDS의 설치는 간단하다.

아래 코드는 RAPIDS 24.08 버전을 설치하기 위한 가상 환경을 생성하는 명령어이다. `cuda-version`만 본인 환경에 맞게 수정하자.

```bash
conda create -n rapids-24.08 -c rapidsai -c conda-forge -c nvidia rapids=24.08 python=3.11 cuda-version=12.5
```

**명령어 설명:**
- `conda create -n rapids-24.08`: rapids-24.08라는 이름의 새로운 가상 환경을 생성한다.
- `-c rapidsai -c conda-forge -c nvidia`: 필요한 패키지를 설치하기 위해 사용할 채널을 지정한다.
- `rapids=24.08`: 설치할 RAPIDS의 버전을 지정한다.
- `python=3.11`: 가상 환경에서 사용할 Python의 버전을 설정한다.
- `cuda-version=12.5`: RAPIDS가 사용할 CUDA 버전을 명시한다.

실습을 진행하기 위해서는 RAPIDS를 설치한 가상 환경을 활성화해야 한다.

```bash
conda activate rapids-24.08
```

가상 환경이 활성화되면, 프롬프트에 `(rapids-24.08)`이 표시된다.

#### 3.2. RAPIDS 선택 설치

[RAPIDS Installation Guide](https://docs.rapids.ai/install/)에서 선택 도구를 사용해 선호하는 방법과 패키지, 환경에 맞게 설치할 수도 있다.

### 4. 추가 라이브러리 설치

시각화에 필요한 라이브러리를 설치한다.

```bash
conda install -n rapids-24.08 -c conda-forge matplotlib seaborn
```

---

## 01-02-01-5. Docker로 RAPIDS 사용하기

Docker를 사용하면 복잡한 환경 설정 없이 RAPIDS를 빠르게 시작할 수 있다. NVIDIA가 제공하는 공식 Docker 이미지에는 RAPIDS, CUDA, Python 등이 모두 포함되어 있다.

### 1. 요구사항

- Docker가 설치되어 있어야 한다.
- NVIDIA Container Toolkit이 설치되어 있어야 한다.
- NVIDIA GPU 드라이버가 설치되어 있어야 한다.

### 2. RAPIDS Docker 이미지 실행

NVIDIA NGC에서 제공하는 공식 RAPIDS 이미지를 사용한다.

```bash
# RAPIDS 이미지 다운로드 및 실행
docker pull nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12

# Jupyter Lab과 함께 실행
docker run --gpus all -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12
```

### 3. Docker 실행 옵션 설명

| 옵션 | 설명 |
|------|------|
| `--gpus all` | 모든 GPU를 컨테이너에서 사용할 수 있도록 설정 |
| `-p 8888:8888` | Jupyter Lab 포트 |
| `-p 8787:8787` | Dask Dashboard 포트 |
| `-p 8786:8786` | Dask Scheduler 포트 |

### 4. 로컬 디렉토리 마운트

작업 디렉토리를 컨테이너에 마운트하려면 `-v` 옵션을 사용한다.

```bash
docker run --gpus all -it -p 8888:8888 \
    -v $(pwd):/rapids/notebooks/host \
    nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12
```

> Docker를 사용하면 환경 충돌 없이 깔끔하게 RAPIDS를 사용할 수 있어, 특히 팀 프로젝트나 재현 가능한 환경이 필요한 경우에 유용하다.

---

## 01-02-02. Colab에서 RAPIDS 사용하기

### 1. Colab이란?

Colab은 호스팅된 Jupyter Notebook 서비스이다. Google 계정으로 가입하면 설치, 설정 없이 GPU 및 TPU와 같은 컴퓨팅 자원을 사용할 수 있다. GPU가 없는 독자라면 Google Colab을 사용하여 실습을 따라갈 수 있다.

**Colab의 주요 장점:**
- **설치 및 설정 필요 없음**: 웹 기반 서비스로, 인터넷 연결만 있으면 어디서나 사용 가능
- **무료 컴퓨팅 자원**: 기본적인 GPU 및 TPU 자원을 무료로 제공
- **협업 기능**: 다른 사용자와 쉽게 노트북을 공유하고 동시에 작업 가능
- **다양한 라이브러리 지원**: TensorFlow, PyTorch 등 주요 머신 러닝 라이브러리가 사전 설치되어 있음

2024년 기준, Google Colab에서 무료로 GPU를 사용하는 경우 최대 12시간까지 사용할 수 있다.

### 2. Colab에서 cuDF, CuPy 사용하기

Google I/O'24에서 RAPIDS cuDF가 Google Colab에 통합되었다고 발표했다. 이제 Google Colab에서 코드 변경 없이 pandas를 가속화할 수 있다.

#### 2.1 GPU 런타임 설정

1. 상단 바의 **런타임** → **런타임 유형 변경** 선택
2. **T4 GPU** 선택하고 저장

```python
!nvidia-smi
```

#### 2.2 cuDF 사용하기

```python
import cudf
import cupy

gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(gdf)
print(type(gdf))
```

**출력:**
```
   a  b
0  1  4
1  2  5
2  3  6
<class 'cudf.core.dataframe.DataFrame'>
```

#### 2.3 RAPIDS 추가 설치

CuPy, CuDF 외 다른 RAPIDS API를 Colab에서 사용하려면 RAPIDS를 설치해야 한다.

```python
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py
```

설치가 완료되면 cuML 등 다른 RAPIDS 라이브러리도 사용할 수 있다.

```python
import cupy as cp
import cudf
from cuml.linear_model import LinearRegression
from cuml.metrics import r2_score

# CuPy 배열 생성
n_samples = 1000
n_features = 2

X_cupy = cp.random.rand(n_samples, n_features).astype(cp.float32)
y_cupy = cp.random.rand(n_samples).astype(cp.float32)

# cuDF로 변환
X_cudf = cudf.DataFrame(X_cupy)
y_cudf = cudf.Series(y_cupy)

# cuML 선형 회귀 모델 학습
model = LinearRegression(copy_X=True)
model.fit(X_cudf, y_cudf)

# 예측 및 평가
y_pred = model.predict(X_cudf)
r2 = r2_score(y_cudf, y_pred)
print("R^2 Score:", r2)
```

---

## 01-02-03. Kaggle에서 RAPIDS 사용하기

Kaggle은 데이터 분석 공부를 시작하려는 분들께 추천한다. 이 플랫폼은 현업에서 겪는 다양한 문제의 데이터를 올려 상금을 걸고 경연을 하고, 토론 및 코드를 공유할 수 있는 공간이다.

### 1. Kaggle 가입 및 기본 사용법

#### 1.1. 가입하기
[Kaggle 웹사이트](https://www.kaggle.com)에 접속해 회원가입을 한다. 구글 계정이나 이메일로 쉽게 가입할 수 있다.

#### 1.2. 대회 참가하기
- 가입 후 'Competitions' 섹션에서 진행중인 대회들을 확인할 수 있다.
- Discussion을 통해 다른 참가자와 소통한다.
- Submissions에서 결과물을 제출한다.
- Leaderboard에서 랭킹 및 점수를 확인한다.

#### 1.3. 코드 작성 및 실행
- 'Code' 섹션에서 'New Notebook'을 클릭해 프로그램을 실행할 수 있다.
- **Session options > ACCELERATOR**에서 'GPU T4 ×2'를 선택해 RAPIDS 코드를 실행할 수 있다.

> **주의**: 다른 사람이 공유한 코드를 Fork하여 별다른 수정 없이 Public으로 공유하면 표절로 간주될 수 있다.

### 2. Kaggle에서 RAPIDS 사용하기

RAPIDS가 Kaggle에 기본 포함되어 있다. 최신 버전을 사용하려면 **ENVIRONMENT**에서 'Always use latest environment'로 설정한다.

```python
import cudf
cudf.__version__
```

---

## 01-02-04. NVIDIA AI Workbench에서 RAPIDS 사용하기

### NVIDIA AI Workbench란?

NVIDIA AI Workbench는 각 프로젝트마다 복잡한 개발 환경을 쉽게 설정할 수 있도록 NVIDIA가 제공한 개발 도구이다.

WSL, NVIDIA 드라이버, CUDA 툴킷, conda, RAPIDS 등 복잡한 설정을 단일 플랫폼으로 간소화할 수 있다.

### 필요한 준비물

- WSL (Windows)
- Docker
- NVIDIA 계정 (키 발급)
- Git 계정

### 시작하기

1. [NVIDIA AI Workbench 공식 문서](https://docs.nvidia.com/workbench/)에서 Workbench를 다운로드하고 설치
2. GitHub에서 Workbench 프로젝트를 복제
3. 빌드 완료 후 JupyterLab에서 RAPIDS를 바로 사용

---

## 01-03. 핵심 명령어 Quick Reference

### 1. 설치 확인 명령어

```bash
# 드라이버 및 GPU 상태 확인
nvidia-smi

# GPU 상세 정보 확인
nvidia-smi --query-gpu=name,memory.total --format=csv

# CUDA Toolkit 버전 확인
nvcc --version

# RAPIDS 버전 확인
python -c "import cudf; print(cudf.__version__)"
```

### 2. Conda 환경 관리

```bash
# 가상 환경 생성
conda create -n [이름] [패키지]

# 환경 활성화
conda activate [환경명]

# 환경 비활성화
conda deactivate

# 환경 목록 확인
conda env list
```

### 3. RAPIDS 설치 명령어

```bash
# RAPIDS 설치 (본인 CUDA 버전에 맞게 수정)
conda create -n rapids-24.08 \
    -c rapidsai -c conda-forge -c nvidia \
    rapids=24.08 python=3.11 cuda-version=12.5

# 환경 활성화
conda activate rapids-24.08
```

> **Tip**: 프롬프트에 `(rapids-24.08)` 표시 = 환경 활성화 성공!

---

## 01-04. 자주 발생하는 문제 및 해결

### Q1: nvidia-smi 실행 안 됨

- **원인**: NVIDIA 드라이버 미설치
- **해결**: NVIDIA 드라이버 재설치

### Q2: nvcc --version 실행 안 됨

- **원인**: CUDA 미설치 또는 PATH 미설정
- **해결**: `~/.bashrc`에 CUDA PATH 추가

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q3: RAPIDS 설치 시 버전 충돌

- **원인**: CUDA와 RAPIDS 버전 불일치
- **해결**: https://docs.rapids.ai/install/ 에서 지원 CUDA 버전 확인

### Q4: WSL에서 GPU 인식 안 됨

- **원인**: WSL1 사용 또는 드라이버 문제
- **해결**: WSL 버전 확인 후 WSL2로 업그레이드

```bash
wsl --set-version Ubuntu-22.04 2
```

> **가장 중요한 것!**
> RAPIDS 설치 전 반드시 https://docs.rapids.ai/install/ 에서 지원하는 CUDA 버전을 확인하세요.

### 추가 도움이 필요하다면?

- [RAPIDS GitHub Issues](https://github.com/rapidsai/cudf/issues)
- NVIDIA Developer Forums
- Stack Overflow (rapids 태그)

---

## 01-05. 오늘의 핵심 정리

### 1. RAPIDS = GPU 가속

Pandas/scikit-learn과 동일한 API로 **2배+ 속도 향상**. 기존 코드를 약간만 수정해서 최소 2배 이상의 속도 향상을 경험할 수 있다.

### 2. 다양한 환경 선택지

| GPU 유무 | 추천 환경 |
|----------|----------|
| GPU 있음 | Local / WSL2 |
| GPU 없음 | Colab / Kaggle |

### 3. 설치 전 필수 확인

- RAPIDS 지원 CUDA 버전
- GPU Compute Capability

### 4. 핵심 명령어

- `nvidia-smi`
- `nvcc --version`

### 입문자 추천 순서

**Colab → Kaggle → Local/WSL2**
