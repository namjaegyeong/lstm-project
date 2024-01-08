import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
from os.path import join  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from tensorflow import keras 
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 시퀀스(sequence) 데이터에 Encoder-Decoder LSTM 아키텍처를 적용하여 구현한 오토인코더

ROOT = os.getcwd()

MY_SENSOR_DATA_FILE = 'sensor_data_test2.csv' # 센서 데이터 CSV 파일 경로
PROJECT_PATH = join(ROOT, MY_SENSOR_DATA_FILE) # 프로젝트 경로

sersor_data =  pd.read_csv(PROJECT_PATH, parse_dates=['dt'], dayfirst=True, infer_datetime_format=True) # 날짜/시간 (Date/Time) 이 포함되어 있을 경우 이를 날짜/시간 형태(DateTime format)에 맞도록 파싱
sersor_data['date'] = pd.to_datetime(sersor_data['dt']) # 시간 정보를 담은 열을 datetime 자료형으로 전환
sersor_data.set_index('date', inplace=True) # set_index를 통해 dt 열 1개를 index로 지정
sersor_data= sersor_data.drop(['dt'], axis='columns') # Dataframe 내 이전 dt Column 삭제

scaler = MinMaxScaler(feature_range=(-1, 1))
scale_cols = sersor_data.columns
df_scaled = sersor_data[scale_cols]

TEST_SIZE = 8
VAILD_SIZE = 4

x_vaild_before, y_vaild_before = train_test_split(df_scaled[-VAILD_SIZE:], test_size=0.2)

# Train / Valid / Test 분리 및 Standardize 적용 
train = pd.DataFrame(scaler.fit_transform(df_scaled[:-TEST_SIZE]))
test = pd.DataFrame(scaler.transform(df_scaled[-TEST_SIZE:-VAILD_SIZE]))
x_vaild_scaled = pd.DataFrame(scaler.transform(x_vaild_before))
y_vaild_scaled = pd.DataFrame(scaler.transform(y_vaild_before))

print(tf.__version__)

# LSTM 모델은 (samples, timesteps, feature)에 해당하는 3d 차원의 shape을 가지므로, 데이터를 시퀀스 형태로 변환한다.
# 1일만 예측할 것이기 때문에 timestamp는 1로 두고 3 차원의 shape 형태로 만들어 주었다.
x_train = np.array(train)
x_train_reshape = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

x_test = np.array(test)
x_test_reshape = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

x_vaild_np = np.array(x_vaild_scaled)
x_vaild_reshape = x_vaild_np.reshape(x_vaild_np.shape[0], 1, x_vaild_np.shape[1])

y_vaild_np = np.array(y_vaild_scaled)
y_vaild_reshape = y_vaild_np.reshape(y_vaild_np.shape[0], 1, y_vaild_np.shape[1])

# 모델의 구조는 Conv1D - Dense층 - LSTM - Dense층으로 encoder 와 decoder가 대칭이 되도록 설계
def conv_auto_model(x):
    n_steps = x.shape[1]
    n_features = x.shape[2]

    keras.backend.clear_session()

    model = keras.Sequential(
        [
            layers.Input(shape=(n_steps, n_features)),
            layers.Conv1D(filters=512, kernel_size=64, padding='same', data_format='channels_last',
                          dilation_rate=1, activation="linear"),
            layers.Dense(128),
            layers.LSTM(
                units=64, activation="relu", name="lstm_1", return_sequences=False
            ),
            layers.Dense(64),
            layers.RepeatVector(n_steps),
            layers.Dense(64),
            layers.LSTM(
                units=64, activation="relu", name="lstm_2", return_sequences=True
            ),
            layers.Dense(128),
            layers.Conv1D(filters=512, kernel_size=64, padding='same', data_format='channels_last',
                          dilation_rate=1, activation="linear"),
            layers.TimeDistributed(layers.Dense(x.shape[2], activation='linear'))
        ]
    )
    return model

# Training LSTM Autoencoder
model = conv_auto_model(x_train_reshape)

# compile
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint_test.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto') # Keras 모델을 학습하는 동안 일정한 간격으로 모델의 가중치를 저장하고, 최상의 성능을 보인 모델을 선택하는 기능 제공

# fit
model_history = model.fit(x_train_reshape, x_train_reshape, 
                                    epochs=200, 
                                    batch_size=8,
                                    # validation_split=0.2, 
                                    validation_data=(x_vaild_reshape, x_vaild_reshape), 
                                    callbacks=[early_stop, checkpoint]) # 모델 학습, ModelCheckpoint 객체를 콜백으로 전달

model.summary()

model.save(filename)

model.load_weights(filename)

# 검증 데이터 입력
predictions_3d = model.predict(x_vaild_reshape)
predictions = predictions_3d.reshape(predictions_3d.shape[0], predictions_3d.shape[2])
x_vaild_reshape_ = x_vaild_reshape.reshape(x_vaild_reshape.shape[0], x_vaild_reshape.shape[2])

# 입력 데이터와 재구성 결과와의 차이인 재구성 손실(reconstruction error) 값 도출
mse = np.mean(np.power(x_vaild_reshape_ - predictions, 2), axis=0)

# Sklearn에서 제공하는 precision_recall_curve 라이브러리를 사용해서 임계값(Threshold) 설정
error_df = pd.DataFrame({'Reconstruction_error' : mse, 'True_class': np.array(y_vaild_before).reshape(-1)})
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])
best_cnt_dic = abs(precision_rt - recall_rt)
threshold_fixed = threshold_rt[np.argmin(best_cnt_dic)]

print('pricision: ', precision_rt[np.argmin(best_cnt_dic)], ', recall: ', recall_rt[np.argmin(best_cnt_dic)])
print('threshold: ', threshold_fixed)

# 정밀도-재현율 그래프를 통해 임계값(Threshold) 확인
# plt.figure(figsize=(8,5))
# plt.plot(threshold_rt, precision_rt[1:], label='Precision')
# plt.plot(threshold_rt, recall_rt[1:], label='Recall')
# plt.xlabel('Threshold'); plt.ylabel('Precision/Recall')
# plt.legend()
# plt.show()

# MSE 값을 precision_recall_curve로 설정한 임계값과 비교하여 정상/이상을 구분, 예측 결과 시각화
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[1], ax.get_xlim()[0], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()