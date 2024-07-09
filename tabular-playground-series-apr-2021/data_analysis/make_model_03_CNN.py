# 딥러닝 모델 만들어서 돌려보자


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
df_train = pd.read_csv('D:/kaggle/0427_train.csv')
df_test = pd.read_csv('D:/kaggle/0427_test.csv')

print(df_train[:5])
print(df_test[:5])

# survived 나눠서 데이터와 라벨로 지정
x_train = df_train.drop('Survived', axis=1).values
y_train = df_train['Survived'].values
x_test = df_test.values

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.3, random_state=519)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# (90000, 10) (90000,)
# (10000, 10) (10000,)

# reshape
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])


#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, AveragePooling1D, Activation, Flatten

model = Sequential()
model.add(Conv1D(320, 2, 1, padding='same', input_shape=(x_train.shape[1:])))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv1D(320, 2, 1, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(240, 2, 1, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(240, 2, 1, padding='same'))
model.add(Activation('relu'))
# model.add(MaxPooling1D(2))
# model.add(AveragePooling1D(2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(240))
model.add(Activation('relu'))
model.add(Dense(240))
model.add(Activation('relu'))
model.add(Dense(180))
model.add(Activation('relu'))
model.add(Dense(90))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.summary()


#컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='loss', patience=16, mode='min', restore_best_weights=True)
file_dir = 'D:/kaggle/h5/submission_07.h5'
lr = ReduceLROnPlateau(factor = 0.25, patience = 8, verbose = 1)
mc = ModelCheckpoint(filepath=file_dir, verbose=1, save_best_only=True)

model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, lr, mc])

# -----------------------------------------------------------------------------------------------------
# 제출물 예측
from keras.models import load_model

submission = pd.read_csv('D:/kaggle/sample_submission.csv')

model = load_model(file_dir)

prediction = model.predict(x_test)
prediction = np.argmax(prediction, axis=1)
print(prediction)

submission['Survived'] = prediction

submission.to_csv('D:/kaggle/sub/submission_07.csv', index=False)

print('==== done ====')


# ========================================
# in kaggle

# submission_01
# 0.77653 932등

# submission_02
# ing

# submission_03 # 노 민맥스, 기본 댄스 모델
# 0.78114

# submission_04 # 민맥스, CNN1, 맥스풀
# 9000/9000 [==============================] - 17s 2ms/step - loss: 0.4942 - acc: 0.7683 - val_loss: 0.4956 - val_acc: 0.7685
# 0.78796

# submission_05 # StandardScaler, CNN1
# 9000/9000 [==============================] - 17s 2ms/step - loss: 0.4935 - acc: 0.7677 - val_loss: 0.4957 - val_acc: 0.7675
# 0.77141

# submission_06 # minmax, CNN1, AveragePooling1D
# 7000/7000 [==============================] - 16s 2ms/step - loss: 0.4943 - acc: 0.7685 - val_loss: 0.4940 - val_acc: 0.7693
# 0.78731

# submission_07 > 하다 멈춤
