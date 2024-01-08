import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from os.path import join 

# 크기가 N인 시계열 데이터를 N - seq_length 개의 지도학습용 데이터로 변환
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)]
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 데이터 불러오기
ROOT = os.getcwd()
MY_SENSOR_DATA_FILE = 'sensor_data_test2.csv' # 센서 데이터 CSV 파일 경로
PROJECT_PATH = join(ROOT, MY_SENSOR_DATA_FILE) # 프로젝트 경로

# 데이터 전처리
sersor_data =  pd.read_csv(PROJECT_PATH, parse_dates=['dt'], dayfirst=True, infer_datetime_format=True) # 날짜/시간 (Date/Time) 이 포함되어 있을 경우 이를 날짜/시간 형태(DateTime format)에 맞도록 파싱
sersor_data['date'] = pd.to_datetime(sersor_data['dt']) # 시간 정보를 담은 열을 datetime 자료형으로 전환
sersor_data.set_index('date', inplace=True) # set_index를 통해 dt 열 1개를 index로 지정
sersor_data= sersor_data.drop(['dt'], axis='columns')  # Dataframe 내 이전 dt Column 삭제

# 데이터의 범위를 -1과 1사이로 변환 시키는 MinMax scaling 진행 (데이터 정규화)
scaler = MinMaxScaler(feature_range=(-1, 1))
scale_cols = sersor_data.columns
df_scaled = scaler.fit_transform(sersor_data[scale_cols])
df_scaled = pd.DataFrame(df_scaled)

# 훈련 데이터와 테스트 데이터 분리
TEST_SIZE = 8 

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

SEQ_LENGTH = 5 # 시계열 데이터를 모델이 지도학습에 사용할 수 있도록 시퀀스(sequence) 데이터의 시퀀스 길이(sequence length)를 정의

train_feature, train_label = create_sequences(train, SEQ_LENGTH) # 예를 들어 시퀀스 길이가 5인 경우 t 시점을 예측하기 위해 과거 t−1, t−2, t−3, t−4, t−5 시점의 데이터를 활용
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2) 
x_test, y_test = create_sequences(test, SEQ_LENGTH)

# 모형 학습
model = Sequential()
model.add(LSTM(64, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )

model.add(Dense(24))

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint_test.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto') # Keras 모델을 학습하는 동안 일정한 간격으로 모델의 가중치를 저장하고, 최상의 성능을 보인 모델을 선택하는 기능 제공

history = model.fit(x_train, y_train, 
                                    epochs=200, 
                                    batch_size=8,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=[early_stop, checkpoint]) # 모델 학습, ModelCheckpoint 객체를 콜백으로 전달

model.save(filename)
model.load_weights(filename)
pred = model.predict(x_test)

print(pred)
print(pred.shape)