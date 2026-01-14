---
sidebar_position: 1
---

# 환경 설정

TensorFlow의 환경 변수, 빌드 구성, 런타임 설정을 설명합니다.

## 주요 환경 변수

### TensorFlow 핵심 환경 변수

| 환경 변수 | 설명 | 기본값 | 예시 |
|-----------|------|--------|------|
| `TF_CPP_MIN_LOG_LEVEL` | 로그 레벨 설정 | `0` | `0`=모두, `1`=INFO 숨김, `2`=WARNING 숨김, `3`=ERROR 숨김 |
| `TF_FORCE_GPU_ALLOW_GROWTH` | GPU 메모리 점진적 할당 | `false` | `true`, `false` |
| `TF_GPU_THREAD_MODE` | GPU 스레드 모드 | `global` | `global`, `gpu_private` |
| `TF_ENABLE_ONEDNN_OPTS` | oneDNN 최적화 활성화 | `1` | `0`, `1` |
| `TF_NUM_INTEROP_THREADS` | op 간 병렬 처리 스레드 수 | 자동 | 숫자 |
| `TF_NUM_INTRAOP_THREADS` | op 내 병렬 처리 스레드 수 | 자동 | 숫자 |

```bash
# 예시: 로그 레벨 설정
export TF_CPP_MIN_LOG_LEVEL=2  # WARNING 이상만 출력

# 예시: GPU 메모리 점진적 할당
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

---

### GPU/CUDA 환경 변수

| 환경 변수 | 설명 | 예시 |
|-----------|------|------|
| `CUDA_VISIBLE_DEVICES` | 보이는 GPU 지정 | `0`, `0,1`, `-1` (GPU 비활성화) |
| `TF_CUDA_PATHS` | CUDA 경로 | `/usr/local/cuda` |
| `TF_CUDNN_PATHS` | cuDNN 경로 | `/usr/local/cuda` |
| `TF_GPU_ALLOCATOR` | GPU 메모리 할당기 | `cuda_malloc_async` |
| `TF_CUDA_HOST_MEM_LIMIT_IN_MB` | 호스트 메모리 제한 | 숫자 |

```bash
# GPU 0번만 사용
export CUDA_VISIBLE_DEVICES=0

# GPU 비활성화
export CUDA_VISIBLE_DEVICES=-1

# 여러 GPU 사용
export CUDA_VISIBLE_DEVICES=0,1,2
```

---

### XLA 컴파일러 환경 변수

| 환경 변수 | 설명 | 예시 |
|-----------|------|------|
| `TF_XLA_FLAGS` | XLA 컴파일러 플래그 | 아래 참조 |
| `XLA_FLAGS` | XLA 추가 플래그 | 아래 참조 |

```bash
# XLA 자동 클러스터링 활성화
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# XLA 디버그 로깅
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_lazy_compilation=false"

# HLO 그래프 덤프
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
```

---

### 결정론적 실행 환경 변수

| 환경 변수 | 설명 | 예시 |
|-----------|------|------|
| `TF_DETERMINISTIC_OPS` | 결정론적 연산 강제 | `1`, `0` |
| `TF_CUDNN_DETERMINISTIC` | cuDNN 결정론적 모드 | `1`, `0` |
| `PYTHONHASHSEED` | Python 해시 시드 | 숫자 |

```bash
# 재현 가능한 학습 설정
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export PYTHONHASHSEED=42
```

---

### 디버그 환경 변수

| 환경 변수 | 설명 | 예시 |
|-----------|------|------|
| `TF_DUMP_GRAPH_PREFIX` | 그래프 덤프 경로 | `/tmp/tf_graphs` |
| `TF_CPP_VMODULE` | 상세 모듈 로깅 | `module_name=level` |
| `TF_CPP_MIN_VLOG_LEVEL` | 상세 로그 레벨 | `0`, `1`, `2` |

```bash
# 그래프 덤프 활성화
export TF_DUMP_GRAPH_PREFIX=/tmp/tf_graphs

# 상세 로깅 활성화
export TF_CPP_MIN_VLOG_LEVEL=1
```

---

## Python에서 설정하기

### GPU 메모리 관리

```python
import tensorflow as tf

# 메모리 증가 허용
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 또는 메모리 제한 설정
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
    )
```

---

### 디바이스 가시성

```python
import tensorflow as tf

# 특정 GPU만 사용
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # GPU 0만

# GPU 비활성화
tf.config.set_visible_devices([], 'GPU')
```

---

### 스레드 설정

```python
import tensorflow as tf

# op 간 병렬 처리 스레드
tf.config.threading.set_inter_op_parallelism_threads(4)

# op 내 병렬 처리 스레드
tf.config.threading.set_intra_op_parallelism_threads(4)
```

---

### 혼합 정밀도 학습

```python
import tensorflow as tf

# 혼합 정밀도 정책 설정
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 원래대로 복원
tf.keras.mixed_precision.set_global_policy('float32')
```

---

## 빌드 구성 (configure.py)

TensorFlow를 소스에서 빌드할 때 사용하는 주요 옵션입니다:

```bash
# 구성 실행
python configure.py
```

### 주요 구성 질문

| 질문 | 설명 | 권장값 |
|------|------|--------|
| Python 경로 | Python 인터프리터 위치 | 자동 감지 |
| CUDA 지원 | GPU 지원 빌드 | Y (GPU 사용 시) |
| CUDA 버전 | CUDA 툴킷 버전 | 시스템에 따라 |
| cuDNN 버전 | cuDNN 라이브러리 버전 | 시스템에 따라 |
| TensorRT 지원 | TensorRT 최적화 | 선택사항 |
| ROCm 지원 | AMD GPU 지원 | AMD GPU 사용 시 |
| 최적화 플래그 | 컴파일러 최적화 | `-march=native` |

---

## Bazel 빌드 옵션 (.bazelrc)

### 주요 빌드 구성

```bash
# CPU 전용 빌드
bazel build --config=opt //tensorflow/tools/pip_package:wheel

# CUDA 빌드
bazel build --config=cuda //tensorflow/tools/pip_package:wheel

# ROCm 빌드
bazel build --config=rocm //tensorflow/tools/pip_package:wheel

# 디버그 빌드
bazel build --config=dbg //tensorflow/tools/pip_package:wheel

# sanitizer 빌드
bazel build --config=asan //tensorflow/tools/pip_package:wheel
```

### 자주 사용하는 플래그

```bash
# 병렬 작업 수 설정
bazel build --jobs=16

# 메모리 제한
bazel build --local_ram_resources=HOST_RAM*.5

# 캐시 디렉토리
bazel build --disk_cache=/path/to/cache

# 원격 캐시
bazel build --remote_cache=grpc://cache.example.com
```

---

## 권장 설정 시나리오

### 개발 환경

```bash
# 로그 표시, 메모리 점진적 할당
export TF_CPP_MIN_LOG_LEVEL=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

```python
import tensorflow as tf

# 즉시 실행 활성화 (기본값)
tf.config.run_functions_eagerly(True)

# 메모리 증가 설정
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

### 프로덕션 환경

```bash
# 최소 로깅
export TF_CPP_MIN_LOG_LEVEL=2

# XLA 최적화
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# GPU 메모리 전체 할당 (더 빠름)
export TF_FORCE_GPU_ALLOW_GROWTH=false
```

```python
import tensorflow as tf

# 그래프 실행 (tf.function 자동 적용됨)
tf.config.run_functions_eagerly(False)

# 혼합 정밀도로 성능 향상
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

---

### 디버깅 환경

```bash
# 모든 로그 표시
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=1

# 그래프 덤프
export TF_DUMP_GRAPH_PREFIX=/tmp/tf_debug

# 결정론적 실행 (재현성)
export TF_DETERMINISTIC_OPS=1
```

```python
import tensorflow as tf

# 숫자 검사 활성화
tf.debugging.enable_check_numerics()

# 즉시 실행 강제
tf.config.run_functions_eagerly(True)

# 장치 배치 로깅
tf.debugging.set_log_device_placement(True)
```

---

### 분산 학습 환경

```bash
# 분산 환경 변수
export TF_CONFIG='{"cluster":{"worker":["host1:port","host2:port"]},"task":{"type":"worker","index":0}}'

# NCCL 설정
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
```

```python
import tensorflow as tf

# 분산 전략 설정
strategy = tf.distribute.MirroredStrategy()

# 또는 멀티 워커
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(...)
```

---

## 환경 변수 검증

```python
import os
import tensorflow as tf

def print_tf_config():
    """TensorFlow 설정 출력"""
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"Eager 실행: {tf.executing_eagerly()}")
    print(f"GPU 장치: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU 장치: {tf.config.list_physical_devices('CPU')}")

    # 환경 변수 확인
    env_vars = [
        'TF_CPP_MIN_LOG_LEVEL',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'CUDA_VISIBLE_DEVICES',
        'TF_XLA_FLAGS',
        'TF_DETERMINISTIC_OPS'
    ]

    print("\n환경 변수:")
    for var in env_vars:
        value = os.environ.get(var, '(설정되지 않음)')
        print(f"  {var}: {value}")

print_tf_config()
```
