# Chapter 01 발표 스크립트: RAPIDS 실습 환경 준비하기

**예상 소요 시간: 12-15분**

---

## [오프닝] RAPIDS 소개 (1분)

> **슬라이드: RAPIDS란?**

안녕하세요, 오늘은 RAPIDS 실습 환경을 어떻게 준비하는지 같이 살펴보겠습니다.

RAPIDS는 NVIDIA에서 만든 GPU 가속 데이터 분석 라이브러리입니다. 쉽게 말해서, 우리가 평소에 쓰는 pandas나 scikit-learn을 GPU에서 돌릴 수 있게 해주는 도구예요.

가장 좋은 점은 API가 거의 동일하다는 겁니다. `import pandas as pd` 대신 `import cudf`로 바꾸면, 나머지 코드는 그대로 써도 됩니다. 그러면서 성능은 GPU 덕분에 훨씬 빨라지죠.

---

## [핵심 용어] 알아야 할 2가지 (1분)

> **슬라이드: 핵심 용어**

RAPIDS를 시작하기 전에 두 가지 용어만 짚고 넘어가겠습니다.

첫 번째는 **CUDA**입니다. CUDA는 NVIDIA GPU에서 연산을 실행할 수 있게 해주는 플랫폼이에요. GPU가 있어도 CUDA가 없으면 GPU 연산을 활용할 수 없습니다.

두 번째는 **Compute Capability**입니다. 이건 GPU의 기능 수준을 나타내는 버전 번호인데요, RAPIDS는 7.0 이상을 요구합니다. RTX 20, 30, 40 시리즈나 GTX 16 시리즈를 쓰고 계시다면 문제없이 사용할 수 있습니다.

---

## [설치 방법] 어떤 걸 선택해야 할까? (2분)

> **슬라이드: 설치 방법 선택 가이드**

RAPIDS를 설치하는 방법이 여러 가지가 있는데, 상황에 따라 적합한 방법이 다릅니다.

**(플로우차트 보면서)**

가장 먼저 확인할 건 GPU 유무입니다.

GPU가 없다면 Colab이나 Kaggle을 추천드립니다. 별도 설치 없이 무료 GPU를 바로 쓸 수 있어요.

GPU가 있다면, 환경 격리가 필요한지 생각해보세요. 팀 프로젝트나 프로덕션 환경처럼 재현성이 중요하면 Docker가 좋고, 개인 연구 목적이라면 Conda로 설치하는 게 편합니다.

**(비교표 보면서)**

정리하면 이렇습니다.

| 방법 | 추천 대상 |
|------|----------|
| Colab, Kaggle | 입문자, GPU 없는 분 |
| Docker | 팀 프로젝트, 프로덕션 환경 |
| Conda | 개인 연구, 로컬 개발 |

처음 시작하시는 분들은 일단 Colab에서 돌려보시고, 나중에 필요하면 로컬에 설치하시는 걸 권장드립니다.

---

## [환경 확인] nvidia-smi 읽는 법 (2분)

> **슬라이드: nvidia-smi**

로컬에 설치하시려면 먼저 본인 환경을 확인해야 합니다.

터미널에서 `nvidia-smi`를 실행해보세요.

**(nvidia-smi 출력 화면 보면서)**

여기서 확인할 건 세 가지입니다.

첫 번째, 오른쪽 상단의 **Driver Version**. CUDA 12.x를 쓰려면 525 이상, CUDA 13.x를 쓰려면 580 이상이 필요합니다.

두 번째, 바로 옆의 **CUDA Version**. 이 숫자가 설치 가능한 최대 CUDA 버전입니다. 예를 들어 여기 13.0이라고 나와 있으면, RAPIDS 설치 시 cuda-version을 12.x나 13.x로 지정하면 됩니다.

세 번째, 중간에 있는 **GPU 이름**. RTX 4060처럼 RTX 20/30/40 시리즈나 GTX 16 시리즈면 RAPIDS를 사용할 수 있습니다.

만약 GTX 10 시리즈 이하라면 Compute Capability가 부족해서 RAPIDS가 지원되지 않습니다. 그런 경우에는 Colab을 사용하시면 됩니다.

---

## [클라우드 환경] Colab & Kaggle (2분)

> **슬라이드: Google Colab**

GPU가 없거나 빠르게 테스트해보고 싶으시면 Colab이 가장 간편합니다.

설정 방법은 간단합니다.

1. 상단 메뉴에서 **런타임** → **런타임 유형 변경**을 클릭합니다.
2. **T4 GPU**를 선택하고 저장하면 끝입니다.

cuDF는 이미 설치되어 있어서 바로 사용할 수 있습니다.

```python
import cudf
gdf = cudf.DataFrame({"a": [1, 2, 3]})
```

cuML 같은 추가 라이브러리가 필요하면 RAPIDS에서 제공하는 설치 스크립트를 실행하시면 됩니다. 문서에 해당 코드가 있으니 참고해주세요.

> **슬라이드: Kaggle**

Kaggle도 방식은 비슷합니다. Notebook 오른쪽 Settings에서 **GPU T4 x2**를 선택하면 됩니다. Colab보다 GPU를 2개 제공해서 조금 더 여유가 있습니다.

---

## [로컬 환경] Conda로 설치하기 (3분)

> **슬라이드: Conda로 설치**

이제 본인 컴퓨터에 직접 설치하는 방법을 알아보겠습니다.

먼저 **Miniforge**를 설치합니다. Anaconda나 Miniconda를 쓰셔도 되지만, Miniforge가 더 가볍고 설치가 빠릅니다.

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

설치가 끝나면 RAPIDS 환경을 생성합니다.

```bash
conda create -n rapids-24.08 \
    -c rapidsai -c conda-forge -c nvidia \
    rapids=24.08 python=3.11 cuda-version=12.5
```

여기서 **cuda-version**은 아까 nvidia-smi에서 확인한 CUDA 버전 이하로 지정해주세요.

설치가 완료되면 환경을 활성화합니다.

```bash
conda activate rapids-24.08
```

터미널 프롬프트 앞에 `(rapids-24.08)`이 표시되면 정상적으로 활성화된 겁니다.

---

## [로컬 환경] Docker로 설치하기 (1분)

> **슬라이드: Docker로 설치**

환경 충돌이 걱정되거나 팀원들과 동일한 환경을 공유해야 한다면 Docker를 추천드립니다.

```bash
docker run --gpus all -it -p 8888:8888 \
    nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12
```

이 명령어 하나로 RAPIDS가 설치된 컨테이너가 실행됩니다. Jupyter Lab도 포함되어 있어서 브라우저에서 `localhost:8888`로 접속하면 바로 작업을 시작할 수 있습니다.

---

## [Windows 사용자] WSL2 설정 (1분)

> **슬라이드: WSL2 설정**

Windows를 쓰시는 분들은 한 가지 주의할 점이 있습니다.

RAPIDS는 Windows에서 직접 실행되지 않습니다. WSL2, 즉 Windows 내부의 Linux 환경에서 실행해야 합니다.

```powershell
wsl --install Ubuntu-22.04
wsl --update
```

이렇게 WSL2를 설치한 다음, 그 안에서 앞서 설명드린 Conda나 Docker 방식으로 RAPIDS를 설치하시면 됩니다.

---

## [마무리] 자주 쓰는 명령어 & 문제 해결 (1분)

> **슬라이드: Quick Reference**

자주 사용하는 명령어를 정리해두었습니다.

```bash
nvidia-smi                    # GPU 및 드라이버 확인
conda activate rapids-24.08   # 환경 활성화
python -c "import cudf; print(cudf.__version__)"  # RAPIDS 버전 확인
```

> **슬라이드: FAQ**

설치 중에 문제가 생기면 문서의 FAQ 섹션을 참고해주세요.

자주 발생하는 문제는 세 가지입니다.

1. **nvidia-smi가 실행되지 않음** → NVIDIA 드라이버 설치 필요
2. **RAPIDS 버전 충돌** → CUDA 버전 호환성 확인
3. **WSL에서 GPU 인식 안 됨** → WSL2로 업그레이드

최신 호환 정보는 공식 문서에서 확인하실 수 있습니다: https://docs.rapids.ai/install/

---

## [클로징]

오늘 내용을 정리하면 이렇습니다.

- GPU가 없으면 → Colab이나 Kaggle
- GPU가 있으면 → Conda 또는 Docker
- Windows라면 → WSL2 설치 후 진행

질문 있으시면 말씀해주세요.

---

## 발표 시간 배분

| 섹션 | 시간 |
|------|------|
| 오프닝 + 핵심 용어 | 2분 |
| 설치 방법 선택 가이드 | 2분 |
| 환경 확인 (nvidia-smi) | 2분 |
| Colab & Kaggle | 2분 |
| Conda 설치 | 3분 |
| Docker + WSL2 | 2분 |
| 마무리 | 1분 |
| **총합** | **14분** |
