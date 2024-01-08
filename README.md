# LSTM Autoencoder

### 과정

1, 필요한 정보를 얻기 위한 시간 격차가 크지 않다면, RNN도 지난 정보를 바탕으로 학습할 수 있다.
2, LSTM은 RNN의 특별한 한 종류로, 긴 의존 기간을 필요로 하는 학습을 수행할 능력을 갖고 있다.
3, 시계열 데이터를 지도학습 문제로 변환 하기 위해서는 예측 대상이 되는 타겟 변수와 예측할 때 사용하는 입력 변수 쌍으로 데이터를 가공해야 한다. 
4, 지도학습이 비지도학습에 비해 학습된 모델의 정확도가 높으나 새로운 이상패턴이 발생하게되면 새로 학습을 진행해야 하기 때문에 비지도 학습 기법 중 LSTM Autoencoder(이하 AE) 라는 알고리즘을 사용하게 되었다.
4, LSTM Autoencoder 학습 시에는 정상(normal) 신호의 데이터로만 모델을 학습한다. 이 모델에 비정상 신호를 입력으로 넣게 되면 정상 분포와 다른 특성의 분포를 나타낼 것이기 때문에 높은 reconstruction error를 보이게 될 것이다.

### 추후 보완할 점

1, 스마트폰이 고장나는 요인(x1, x2, xn)들과 고장 유무(y)을 설정하는 전처리 과정 추가 필요하다.
2, LSTM Autoencoder 학습 시 Normal(y == 0) 데이터만으로 학습할 것이기 때문에 데이터로부터 Normal(y == 0)과 Break(y == 1) 데이터를 분리해야 한다. 

### 구현 환경

본 연구에서 사용된 LSTM 모델은 Jetson AGX Orin Developer Kit을 기반으로 하는 서버에서 구동되었다. 
해당 서버는 2048-core NVIDIA Ampere architecture GPU with 64 Tensor Cores를 탑재하고 있다. 
서버에 사용된 운영체제는 Ubuntu 20.04.6 LTS, 딥러닝 모델 구현에는 Tensorflow의 고수준 API인 Keras 2.9.0과 CUDA 11.4가 사용되었다. 

### 참고 링크

* https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr
* https://pseudo-lab.github.io/Tutorial-Book/chapters/time-series/Ch3-preprocessing.html
* https://velog.io/@jaehyeong/LSTM-Autoencoder-for-Anomaly-Detection
* https://velog.io/@jonghne/LSTM-AE%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%8B%9C%EA%B3%84%EC%97%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B4%EC%83%81-%ED%83%90%EC%A7%80-1-%EA%B0%9C%EC%9A%94
*https://dacon.io/codeshare/5141
