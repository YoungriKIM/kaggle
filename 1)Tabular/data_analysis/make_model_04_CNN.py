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
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPool1D, AveragePooling1D, Activation, Flatten, Concatenate

model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 2, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 2, padding="same")(x)
    # x = Add()([x, s])
    x= Concatenate(axis=1)([x,s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=2, strides=1)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation="sigmoid", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()


#컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='loss', patience=16, mode='min', restore_best_weights=True)
file_dir = 'D:/kaggle/h5/submission_09.h5'
lr = ReduceLROnPlateau(factor = 0.25, patience = 8, verbose = 1)
mc = ModelCheckpoint(filepath=file_dir, verbose=1, save_best_only=True)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, lr, mc])

# -----------------------------------------------------------------------------------------------------
# 제출물 예측
from keras.models import load_model

submission = pd.read_csv('D:/kaggle/sample_submission.csv')

model = load_model(file_dir)

prediction = model.predict(x_test)
prediction = np.argmax(prediction, axis=1)
print(prediction)

submission['Survived'] = prediction

submission.to_csv('D:/kaggle/sub/submission_09.csv', index=False)

print('==== done ====')


# ========================================
# in kaggle

# submission_01
# 0.77653 932등

# submission_02
# 0.34911 ???????

# submission_03 # 노 민맥스, 기본 댄스 모델
# 0.78114

# submission_04 # 민맥스, CNN1, 맥스풀
# 9000/9000 [==============================] - 17s 2ms/step - loss: 0.4942 - acc: 0.7683 - val_loss: 0.4956 - val_acc: 0.7685
# 0.78796

# submission_05 # StandardScaler, CNN1
# 9000/9000 [==============================] - 17s 2ms/step - loss: 0.4935 - acc: 0.7677 - val_loss: 0.4957 - val_acc: 0.7675
# 학원컴 ing
# 0.77141

# submission_06 # minmax, CNN1, AveragePooling1D
# 7000/7000 [==============================] - 16s 2ms/step - loss: 0.4943 - acc: 0.7685 - val_loss: 0.4940 - val_acc: 0.7693
# 0.78731

# submission_07 > 하다 멈춤

# submission_08
# CNN2
# 학원 ing

# submission_09 keras 기본 모델
# 7000/7000 [==============================] - 49s 7ms/step - loss: 0.4922 - acc: 0.7693 - val_loss: 0.4944 - val_acc: 0.7694
# 0.78320

