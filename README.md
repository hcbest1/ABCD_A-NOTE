# ABCD_A-NOTE ( 2024.05.16 ~ )
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import torch
import torch.nn as nn
import torch.nn.functional as F
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import torch
import torch.nn as nn
import torch.nn.functional as F

## Pooling example
input_example = torch.tensor([[[0, 1.0, 2], [3, 4, 5], [6, 7, 8]]])
print(input_example)
# max Pooling
max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=1)
print(max_pooling_layer)
print(max_pooling_layer(input_example))
# average pooling
average_pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=1)
print(average_pooling_layer)
print(average_pooling_layer(input_example))
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
## QUIZ!!
import torch
import torch.nn as nn
import torch.nn.functional as F

input_example = torch.tensor([[[12, 20, 30, 0.0], [20, 12, 2, 0], [0, 70, 5, 2]]])
print(input_example)

max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
print('\nmax pooling:\n', max_pooling_layer(input_example))

average_pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
print('\naverage pooling:\n', average_pooling_layer(input_example))
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import torch
import torch.nn as nn
import torch.nn.functional as F

inputs = torch.Tensor(1, 1, 32, 32)
inputs.shape
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
print(conv1)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
print(conv2)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
out = out.view(out.size(0), -1)
print(out.shape)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
fc = nn.Linear(4096, 10)
out = fc(out)
print(out.shape)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# 이미지 데이터의 크기 정의
height = 28
width = 28
channels = 3 # RGB 이미지
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# 이미지 데이터의 크기 정의
height = 28
width = 28
channels = 3 # RGB 이미지

# 신경망 모델 구성
input_layer = Input(shape=(height, width, channels)) # 입력층의 shape 설정
## Input 함수는 모델의 입력을 정의합니다. 여기서 shape 파라미터는 각 입력 데이터의 형태를 (28, 28, 3)으로 지정하여, 각 이미지가 28X28 픽셀, 3개의 색상 채널을 가지도록 설정합니다.
flatten_layer = Flatten()(input_layer) # 이미지 데이터를 1D 벡터로 펼치기 
# 이는 3차원의 이미지 데이터를 신경망이 처리할 수 있는 형태로 변환하는 데 사용됩니다.
hidden_layer = Dense(64, activation='relu')(flatten_layer) # 숨겨진 계층 
## Dense는 완전 연결 계층을 추가합니다. 이 게층은 64개의 뉴런을 가지며, 각 뉴런은 ReLU 활성화 함수를 사용합니다. ReLU는 입력이 0 이상이면 그대로 출력, 0 이하면 0을 출력하는 비선형 함수로, 모델이 복잡한 패턴을 학습할 수 있게 도와줍니다.
output_layer = Dense(10, activation='softmax')(hidden_layer) # 출력 계층
## 출력 계층 역시 완전 연결 계층입니다. 10개의 뉴런을 사용하며, 각 뉴런은 클래스에 대한 확률을 나타내는 softmax 활성화 함수를 사용합니다. 이 계층은 10개의 서로 다른 클래스(예를 들어 숫자 0부터 9까지)에 대한 예측 확률을 출력합니다.

#모델 생성
model = Model(inputs=input_layer, outputs=output_layer)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
