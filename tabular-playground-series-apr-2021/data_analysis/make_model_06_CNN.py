# CNN1D


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
x_train = x_train.reshape(x_train.shape[0], 10, 1)
x_val = x_val.reshape(x_val.shape[0], 10, 1)
x_test = x_test.reshape(x_test.shape[0], 10, 1)


# -----------------------------------------------------------------------------------------------------
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, MaxPooling2D, MaxPooling1D, BatchNormalization, Flatten, Input
from tensorflow.keras.layers import Activation, ZeroPadding2D, Concatenate, AveragePooling2D, ZeroPadding1D, AveragePooling1D
from tensorflow.keras.regularizers import l2

input = Input(shape=(10, 1))

conv1_7x7_s2 = Conv1D(64, 2, strides=1, padding='valid', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input)
pool1_helper = BatchNormalization()(conv1_7x7_s2)

conv2_3x3_reduce = Conv1D(64, 1, padding='same', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_helper)
conv2_3x3 = Conv1D(192, 2, padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
pool2_helper = BatchNormalization()(conv2_3x3)
pool2_3x3_s2 = MaxPooling1D(pool_size=2, strides=1, padding='valid', name='pool2/3x3_s2')(pool2_helper)

inception_3a_1x1 = Conv1D(32, 1, padding='same', activation='relu', name='inception_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
inception_3a_3x3_reduce = Conv1D(96, 1, padding='same', activation='relu', name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
inception_3a_3x3 = Conv1D(32, 2, padding='valid', activation='relu', name='inception_3a/3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
inception_3a_5x5_reduce = Conv1D(16, 1, padding='same', activation='relu', name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
inception_3a_5x5 = Conv1D(32, 2, padding='valid', activation='relu', name='inception_3a/5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
inception_3a_pool = MaxPooling1D(pool_size=2, strides=1, padding='same', name='inception_3a/pool')(pool2_3x3_s2)
inception_3a_pool_proj = Conv1D(32, 1, padding='same', activation='relu', name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
inception_3a_output = Concatenate(axis=1, name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

flat = Flatten()(inception_3a_output)
dense1 = Dense(32, activation='relu')(flat)
dense2 = Dense(16, activation='relu')(dense1)
output = Dense(2, activation='sigmoid')(dense2)

model = Model(inputs=input, outputs=output)
model.summary()
# -----------------------------------------------------------------------------------------------------
#컴파일, 훈련
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam

<<<<<<< HEAD
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
=======
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.1), metrics=['acc'])
>>>>>>> 3ea26d3fe05deb1dad4d912ac88003632ad3c404

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='loss', patience=16, mode='min', restore_best_weights=True)
file_dir = 'D:/kaggle/h5/submission_011.h5'
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

submission.to_csv('D:/kaggle/sub/submission_011.csv', index=False)

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

<<<<<<< HEAD
# submission_08 SGD
=======
# submission_08
# CNN2
# 학원 ing
>>>>>>> 3ea26d3fe05deb1dad4d912ac88003632ad3c404

# submission_09 keras 기본 모델
# 7000/7000 [==============================] - 49s 7ms/step - loss: 0.4922 - acc: 0.7693 - val_loss: 0.4944 - val_acc: 0.7694
# 0.78320

# submission_010
<<<<<<< HEAD
# 7000/7000 [==============================] - 78s 11ms/step - loss: 0.6964 - acc: 0.5561 - val_loss: 0.6905 - val_acc: 0.5698 # 그만

# submission_011 Adam # 그만

# submission_011 SGD
# 7000/7000 [==============================] - 26s 4ms/step - loss: 0.4934 - acc: 0.7687 - val_loss: 0.4944 - val_acc: 0.7684

# submission_012 
# 06바탕으로 SGD, RMSprop, Adadelte 돌리기
=======
# 7000/7000 [==============================] - 78s 11ms/step - loss: 0.6964 - acc: 0.5561 - val_loss: 0.6905 - val_acc: 0.5698
# 그만

# submission_011 Adam
# 학원컴

# submission_011 SGD
>>>>>>> 3ea26d3fe05deb1dad4d912ac88003632ad3c404
