---
sidebar_position: 1
---

# 시작 가이드

TensorFlow를 처음 사용하는 분들을 위한 실전 가이드입니다. 실제 코드 예제와 함께 기본적인 사용법을 배웁니다.

## 설치

### 기본 설치

```bash
# GPU 지원 TensorFlow (Ubuntu/Windows)
pip install tensorflow

# CPU 전용 (더 가벼움)
pip install tensorflow-cpu

# 최신 버전으로 업데이트
pip install --upgrade tensorflow
```

### 설치 확인

```python
import tensorflow as tf

print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")
```

---

## 기본 사용법

### 텐서 생성

```python
import tensorflow as tf

# 상수 텐서 생성
hello = tf.constant('안녕하세요, TensorFlow!')
print(hello.numpy())  # b'안녕하세요, TensorFlow!'

# 숫자 연산
result = tf.add(1, 2)
print(result.numpy())  # 3
```

### 다양한 텐서 생성 방법

```python
import tensorflow as tf
import numpy as np

# 상수에서 생성
a = tf.constant([[1, 2], [3, 4]])

# NumPy 배열에서 변환
np_array = np.array([[1, 2], [3, 4]])
b = tf.convert_to_tensor(np_array)

# 특수 텐서
zeros = tf.zeros([3, 3])         # 0으로 채워진 3x3
ones = tf.ones([2, 4])           # 1로 채워진 2x4
random = tf.random.normal([5])   # 정규분포에서 5개 샘플

print(f"zeros shape: {zeros.shape}")  # (3, 3)
print(f"ones shape: {ones.shape}")    # (2, 4)
```

---

## 핵심 패턴

### 패턴 1: tf.function으로 성능 최적화

`@tf.function` 데코레이터를 사용하여 Python 함수를 그래프로 컴파일합니다.

```python
import tensorflow as tf

@tf.function
def format_example(imgs, labels):
    """각 학습 및 테스트 예제를 모델에 맞게 포맷합니다."""
    imgs = tf.reshape(imgs, [-1, 28 * 28])
    imgs = tf.cast(imgs, tf.float32) / 255.0
    labels = tf.one_hot(labels, depth=10, dtype=tf.float32)
    return imgs, labels

@tf.function
def dense_layer(weights, input_tensor, act=tf.nn.relu):
    """단일 밀집 레이어의 순전파를 수행합니다."""
    kernel, bias = weights
    preactivate = tf.matmul(input_tensor, kernel) + bias
    activations = act(preactivate)
    return activations
```

:::tip 성능 팁
`@tf.function`은 첫 호출 시 트레이싱으로 그래프를 생성하고, 이후 호출에서는 최적화된 그래프를 재사용합니다.
:::

---

### 패턴 2: GradientTape로 학습

현대적인 TensorFlow 2.x 학습 패턴입니다.

```python
import tensorflow as tf

# 가중치 초기화
def get_dense_weights(input_dim, output_dim):
    initial_kernel = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=0.1, seed=42)
    kernel = tf.Variable(initial_kernel([input_dim, output_dim]))
    bias = tf.Variable(tf.constant(0.1, shape=[output_dim]))
    return kernel, bias

hidden_weights = get_dense_weights(784, 500)
output_weights = get_dense_weights(500, 10)
variables = hidden_weights + output_weights

# 모델과 손실 정의
@tf.function
def model(x):
    hidden_act = dense_layer(hidden_weights, x)
    logits_act = dense_layer(output_weights, hidden_act, tf.identity)
    return tf.nn.softmax(logits_act)

@tf.function
def loss(probs, labels):
    diff = -labels * tf.math.log(probs)
    return tf.reduce_mean(diff)

# 학습 루프
optimizer = tf.optimizers.Adam(learning_rate=0.025)

for i in range(max_steps):
    x_train, y_train = next(train_batches)

    # GradientTape로 학습 스텝
    with tf.GradientTape() as tape:
        y = model(x_train)
        loss_val = loss(y, y_train)

    grads = tape.gradient(loss_val, variables)
    optimizer.apply_gradients(zip(grads, variables))

    # 평가
    y_pred = model(x_test)
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print(f"스텝 {i}: 정확도 = {accuracy.numpy():.4f}")
```

---

### 패턴 3: tf.data.Dataset으로 데이터 파이프라인

효율적인 데이터 로딩 및 전처리 파이프라인입니다.

```python
import tensorflow as tf

# MNIST 데이터 로드
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

# 셔플링과 배칭이 포함된 학습 데이터셋 생성
train_ds = tf.data.Dataset.from_tensor_slices(mnist_train).shuffle(
    buffer_size=batch_size * max_steps,
    seed=42
).batch(batch_size)

# 전처리 적용
train_ds = train_ds.map(format_example)

# 테스트 데이터셋 생성
test_ds = tf.data.Dataset.from_tensor_slices(mnist_test).repeat().batch(
    len(mnist_test[0]))
test_ds = test_ds.map(format_example)

# 배치 반복
for batch in train_ds:
    x_train, y_train = batch
    # 배치 처리...
```

---

## 사용 사례

### 사용 사례 1: TensorFlow Lite로 이미지 분류

TFLite 모델을 로드하고 이미지를 분류하는 완전한 워크플로우입니다.

```python
import numpy as np
from PIL import Image
from tensorflow.lite.python import lite

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

# TFLite 모델 로드
interpreter = lite.Interpreter(
    model_path='model.tflite',
    num_threads=4
)
interpreter.allocate_tensors()

# 입력/출력 상세 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 모델이 float 입력을 기대하는지 확인
floating_model = input_details[0]['dtype'] == np.float32

# 예상 입력 차원 가져오기
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# 이미지 로드 및 전처리
img = Image.open('image.jpg').resize((width, height))
input_data = np.expand_dims(img, axis=0)

if floating_model:
    input_mean = 127.5
    input_std = 127.5
    input_data = (np.float32(input_data) - input_mean) / input_std

# 추론 실행
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 결과 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

# 상위 5개 예측
top_k = results.argsort()[-5:][::-1]
labels = load_labels('labels.txt')

for i in top_k:
    print(f'{results[i]:.6f}: {labels[i]}')
```

---

### 사용 사례 2: SavedModel 저장 및 로드

시그니처가 있는 모델을 저장하고 복원합니다.

```python
import tensorflow as tf

# 여러 진입점이 있는 모델 정의
class UseMultiplex(tf.Module):

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[5], dtype=tf.bool),
        tf.TensorSpec(shape=[5], dtype=tf.int64),
        tf.TensorSpec(shape=[5], dtype=tf.int64)
    ])
    def use_multiplex(self, cond, a, b):
        return tf.where(cond, a, b)

model = UseMultiplex()

# 명시적 시그니처로 저장
tf.saved_model.save(
    model,
    'saved_model_path',
    signatures=model.use_multiplex.get_concrete_function(
        tf.TensorSpec(shape=[5], dtype=tf.bool),
        tf.TensorSpec(shape=[5], dtype=tf.int64),
        tf.TensorSpec(shape=[5], dtype=tf.int64)
    ))

# 모델 로드 및 사용
restored = tf.saved_model.load('saved_model_path')
cond = tf.constant([True, False, True, False, True], dtype=bool)
a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
result = restored.use_multiplex(cond, a, b)
# 결과: [1, 20, 3, 40, 5]
```

---

### 사용 사례 3: TensorFlow Lite 변환

TensorFlow 모델을 모바일 배포용 TFLite로 변환합니다.

```python
import tensorflow as tf
from tensorflow.lite.python import lite

# 간단한 모델 생성
root = tf.Module()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# 입력 시그니처로 구체적 함수 가져오기
input_data = tf.constant(1.0, shape=[1])
concrete_func = root.f.get_concrete_function(input_data)

# TFLite로 변환
converter = lite.TFLiteConverterV2.from_concrete_functions(
    [concrete_func], root)
tflite_model = converter.convert()

# 모델 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# TFLite 모델 로드 및 실행
interpreter = lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 설정 및 추론 실행
interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
interpreter.invoke()

# 출력 가져오기
output = interpreter.get_tensor(output_details[0]['index'])
print(f"결과: {output}")  # 6.0 (3 * 2 * 1)
```

---

## 에러 처리

```python
import tensorflow as tf

# Shape 불일치 처리
try:
    a = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2,3)
    b = tf.constant([[10, 20], [30, 40], [50, 60]])  # shape (3,2)
    result = a + b  # shape 불일치로 실패
except (tf.errors.InvalidArgumentError, ValueError) as e:
    print(f"Shape 에러: {e}")

# 타입 불일치 처리
try:
    a = tf.constant([1.0, 2.0, 3.0])  # float
    b = tf.constant([10, 20, 30])  # int32
    # 연산 실패 또는 예상치 못한 결과 발생 가능
except TypeError as e:
    print(f"타입 에러: {e}")

# check_numerics로 디버깅
tf.debugging.enable_check_numerics()  # 학습 중 NaN/Inf 감지

# 그래프 모드에서 assertion 사용
tf.debugging.assert_shapes([
    (tensor_a, ('N', 'M')),
    (tensor_b, ('N', 'M')),
])
```

---

## 통합 예제

### Keras와 통합

```python
import tensorflow as tf

# Keras Sequential 모델 구축
keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=4, activation='relu', name='x'),
    tf.keras.layers.Dense(1, activation='relu', name='output'),
])

# Keras 모델을 TFLite로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# 변환된 모델에서 시그니처 접근
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
print(signatures)
```

### 커스텀 연산과 통합

```python
import tensorflow as tf
from tensorflow.python.platform import resource_loader

# 커스텀 op 라이브러리 로드
_custom_op_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("custom_kernel.so"))

# 커스텀 op 사용
custom_op = _custom_op_module.custom_op

def my_custom_function(cond, a, b, name=None):
    """문서화가 포함된 커스텀 op 래퍼."""
    return custom_op(cond=cond, a_values=a, b_values=b, name=name)

# 사용
result = my_custom_function(
    cond=tf.constant([True, False, True]),
    a=tf.constant([1, 2, 3]),
    b=tf.constant([10, 20, 30])
)
```

---

## 모범 사례

실제 TensorFlow 코드에서 발견된 모범 사례입니다:

1. **성능을 위해 `@tf.function` 사용**: 반복 호출되는 함수에 데코레이터를 붙여 그래프 최적화를 활성화합니다.

2. **입력 시그니처 명시적 정의**: 더 나은 모델 직렬화와 TFLite 변환이 가능합니다.

3. **데이터 파이프라인에 `tf.data.Dataset` 사용**: 효율적인 배칭, 셔플링, 프리페칭을 제공합니다.

4. **상태 관리에 AutoTrackable 활용**: 체크포인팅을 위해 변수를 자동으로 추적합니다.

5. **에러를 우아하게 처리**: TensorFlow는 shape 불일치에 `InvalidArgumentError` 같은 특정 예외를 발생시킵니다.

6. **모니터링에 TensorBoard 사용**: 학습 진행 상황을 추적하기 위해 summary writer를 통합합니다.

7. **서빙을 위해 시그니처와 함께 내보내기**: 프로덕션 배포를 위해 항상 명시적 시그니처로 모델을 내보냅니다.
