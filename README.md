## 목차

1. Tokenizer 구축 [링크](https://github.com/a2ran/sent_cognition/blob/main/model_tokenizer.ipynb)
2. Text classification model 구축 [링크](https://github.com/a2ran/sent_cognition/blob/main/text_classification_model.ipynb)
3. Audio classification model 작업 [링크](https://github.com/a2ran/sent_cognition/blob/main/audio_classification_model.ipynb)
4. Multimodal 모델 구축 [링크](https://github.com/a2ran/sent_cognition/blob/main/multimodal_model.ipynb)

## 1. Tokenizer 구축

자연어 처리 모델을 구축해 분류 작업을 진행하기 전, 수집한 코퍼스 (corpus)를 기반으로 토크나이저 (tokenizer)를 새로 정의하고 학습을 진행하고자 한다.

코퍼스 데이터를 기반으로 토크나이저를 새로 정의하여 얻을 수 있는 이득은 다음과 같다.

1. 용량 최적화 : 기존 자연어처리 모델 (koBERT, multilingualBERT, T5 등...)은 해당 모델이 학습한 코퍼스에 최적화 (optimized)한 토큰 사전 (dictionary)를 정의해 맞춤형 학습을 진행한다. 하지만, 방대한 양의 데이터로부터 학습하는 모델 특성상 수많은 가짓수의 단어의 정보를 학습해야 하므로 정의한 토큰의 수가 증가하게 된다. (koBERT의 경우 독립적인 토큰의 개수는 8,002개.) 특히 외국어 정보를 포함하는 multilingual model의 경우 해외 알파벳 토큰까지 추가하므로 정의 토큰의 가짓수가 늘어나 학습량을 늘리는 결과를 초래한다.
2. 문장 내 특징과 패턴 학습 : 학습 코퍼스 내에서 토큰화를 진행해 새로운 토크나이저를 정의하면, 기존 pretrained 토크나이저에 비해 더 많은 단어 토큰을 포함할 수 있게 되고 (단어 개수 threshold을 정의할 시), 이는 모델이 [UNK] 토큰이 아닌 단어 정보다 담긴 토큰을 학습하게 해 더 상세한 특징과 패턴을 학습 가능하게 한다.

이외 여러가지 장점이 있기 때문에, 학습 텍스트에 대해 토크나이저를 정의한 후 학습을 진행한다. 토그나이저의 기본 베이스로 한국어 분류에 강점을 가지는 koBERT을 사용한다.

```
vocab_size = 4096
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size = vocab_size)
```

`batch_iterator()`을 사용해 4096개의 독립적인 토큰 수를 가지는 새로운 토크나이저를 학습한다.

![image](https://github.com/a2ran/sent_cognition/assets/121621858/5a7dbab7-d248-4a5d-ab4c-c9ed5450dcdc)

다음은 토크나이저로 변환한 문장의 토큰 개수를 나타낸 히스토그램이다. 각 문장은 평균 45개, 표준편차 18개의 토큰 수를 보이고 있고, 가장 긴 문장은 169개의 토큰 수를 보유중이다. 해당 토크나이저를 huggingface에 업로드 후, 후에 서술할 text classification model 구축에 사용하였다.

[https://huggingface.co/a2ran/sent_cognition](https://huggingface.co/a2ran/sent_cognition)

## 2. Text Classification Model 구축

**모델 구조도**

1. 입력 데이터
   
```
ids: tensor([   2, 3219, 1568, 3034, 1915, 2902,  812, 3192, 3665, 2836, 3236, 2849,
         649, 1924, 1554, 2868,   10,    3,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1])
mask: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])
targets: 5

[CLS] 면접에 떨어졌다는 소식을 듣고 너무 실망해서 밥맛도 없어. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```

2. 모델 구조

```
입력 데이터
|
----인코더 (TransformerWrapper)---------------------
| 토큰 임베딩 (4096 -> 512)
| 포지셔널 임베딩 (64 -> 512)
| 어텐션 layers (총 18개)
| -- pre-norm Normalization
| -- Position-wise Feed Forward (512 -> 2048 -> 512)
| -- 어텐션 layer (Q, K, V = 512)
| -- 최종 layer normalization
| -- Position-wise Feed Forward (512 -> 4096)
--------------------------------
|
----분류기 (NN_Classifier)---------------------------
| Conv1d layers (총 3개) (1층: 1 -> 128, kernel_size=3, 나머지 128 -> 128, kernel_size=3)
| MaxPooling layer (MaxPool1d: kernel_size=2, stride=2)
| Fully Connected layer
| -- Feed Forward (2045 -> 7)
| -- Dropout (p = 0.1)
| -- 활성화 함수 (Tanh)
| 어텐션 layer
| -- Linear (W: in_features=2045, out_features=2045)
| -- Linear (V: in_features=2045, out_features=1)
-----------------------------------------------------
```

텍스트 분류 (Text Classification) 모델은 Multi-Particle Dynamic System Point of View 트랜스포머 구조를 차용한 Macaron-NET [링크](https://arxiv.org/abs/1906.02762)을 인코더 (Encoder)로 64차원의 입력 벡터를 4096 차원의 고차원 벡터로 보낸 이후, 컨볼루션 레이어를 사용한 Fully Connected Network에 어텐션 (Attention) layer를 추가해 문장 내 토큰간의 관계를 고려한 분류기 구조를 구축하였다.

![image](https://github.com/a2ran/sent_cognition/assets/121621858/13674db4-d3a6-4808-891d-5742d31c48c6)

![image](https://github.com/a2ran/sent_cognition/assets/121621858/d1f1f126-9e56-46b6-86fc-7e9bb6f704e3)

## 3. Audio Classification Model 작업

사용자의 음성을 기반으로 감성 분석 작업을 진행하기 위해, Meta에서 개발한 Wav2Vec2.0 모델의 weight과 bias을 가져와 학습을 진행하고자 한다.

![image](https://github.com/a2ran/sent_cognition/assets/121621858/b1ff3e2d-629b-4db4-9f41-b4fccd626ddf)

self-supervised learning 기법을 사용해 음성의 음향적 특징을 추출해 감성인식을 보조하는 wav2vec2.0은 사용성이 높은 모델이지만, 모델 자체 용량이 무거워 (파라미터 수 3.1M, 총 용량 1.2GB) 멀티모달 구축에는 MFCC 모델을 사용했지만, 모델의 성능을 유지하면서 사용 용량을 줄이는 작업을 꾸준이 하여, 작업한 내역을 남기고자 한다. 현재 용량 축소는 18개의 encoder convlution layer를 3개로 축소하는 방식으로 진행했지만, 추후 작업을 진행한다면 용량을 줄이는 동시에 모델 전반의 weight와 bias을 반영하는 방면으로 작업을 진행할 예정이다.

## 4. Multimodal 모델 구축

최종적으로 구축한 Text_Classification 모델과 음향적 특징을 추출하는 MFCC 모델을 concat해 최종 multimodal framework을 구축했다.

![image](https://github.com/a2ran/sent_cognition/assets/121621858/ed316425-02bc-45f8-af1b-f2a510c2e150)

![image](https://github.com/a2ran/sent_cognition/assets/121621858/dd7e859c-9350-4b0a-9ef0-6078ba65b4a3)


