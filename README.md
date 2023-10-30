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

