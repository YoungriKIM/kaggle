# 딥러닝 모델 만들어서 돌려보자

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from math import nan
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
from sklearn.model_selection import train_test_split 

# -----------------------------------------------------------------------------------------------------
# 데이터 지정
df_train = pd.read_csv('D:/kaggle/0426_train.csv')
df_test = pd.read_csv('D:/kaggle/0426_test.csv')

print(df_train[:5])
print(df_test[:5])

# survived 나눠서 데이터와 라벨로 지정
x_train = df_train.drop('Survived', axis=1).values
y_train = df_train['Survived'].values
x_test = df_test.values

# split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.1, random_state=519)

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# (80000, 7) (80000,)
# (20000, 7) (20000,)

# 민맥스 쓴 것도 비교!
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
# x_test = scaler.transform(x_test)


#모델 구성
input1 = Input(shape=(x_train.shape[1:]))
dense1 = Dense(180, activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(180)(dropout1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(120)(dropout1)
dense1 = Dense(80)(dense1)
dense1 = Dense(40)(dense1)
output1 = Dense(2, activation='sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

#컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(monitor='loss', patience=8, mode='min', restore_best_weights=True)
lr = ReduceLROnPlateau(factor = 0.25, patience = 4, verbose = 1)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, lr])

# -----------------------------------------------------------------------------------------------------
# 제출물 예측
submission = pd.read_csv('D:/kaggle/sample_submission.csv')

prediction = model.predict(x_test)
prediction = np.argmax(prediction, axis=1)
print(prediction)

submission['Survived'] = prediction

submission.to_csv('D:/kaggle/sub/submission_03.csv', index=False)

print('==== done ====')


# ========================================
# in kaggle

# submission_01
# 0.77653 932등

# submission_02
# ing

# submission_03 # 노 민맥스, 기본 댄스 모델
# 0.78114