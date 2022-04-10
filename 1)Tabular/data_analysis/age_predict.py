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
df_train = pd.read_csv('D:/kaggle/0427_train_age_02.csv')
df_test = pd.read_csv('D:/kaggle/0427_test_age_02.csv')
df_train.drop(['Survived'], axis=1, inplace=True)

####
# df_train.drop(['Pclass'], axis=1, inplace=True)
# df_train.drop(['Sex'], axis=1, inplace=True)
# df_train.drop(['FamilySize'], axis=1, inplace=True)
# df_train.drop(['IsAlone'], axis=1, inplace=True)
####

print(df_train[:5])
print(df_test[:5])

# 나이가 있는 데이터와 아닌 데이터 나누기
x_train_have_age = df_train[df_train['Age_cat']!=8]
x_train_not_have_age = df_train[df_train['Age_cat']==8]
# x_test_have_age = df_test[df_test['Age_cat']!=8]
# x_test_not_have_age = df_test[df_test['Age_cat']==8]

print(x_train_have_age.shape)
print(x_train_not_have_age.shape)
# print(x_test_have_age.shape)
# print(x_test_not_have_age.shape)

x_train = x_train_have_age.drop('Age_cat', axis=1).values
y_train = x_train_have_age['Age_cat'].values
x_pred = x_train_not_have_age.drop('Age_cat', axis=1).values
y_pred = x_train_not_have_age['Age_cat'].values   # 나중에 덮어쓸 용

print(x_train[:5])
print(y_train[:5])
print(x_pred[:5])


# -----------------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_pred = scaler.transform(x_pred)

print(x_train.shape, y_train.shape)
print(x_pred.shape)
# (96708, 4) (96708,)
# (3292, 4)


# split
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.3, random_state=519)

# # vectorize
# from sklearn.preprocessing import OneHotEncoder
# # y_train = y_train.reshape(1, -1)
# # y_try_valain = y_train.reshape(1, -1)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# ohe.fit(y_train)
# y_train = ohe.transform(y_train).toarray()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)

print(x_train.shape, y_train.shape)
print(x_pred.shape)
# (96708, 4) (96708, 8)
# (3292, 4)

# reshape
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
# x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
x_pred = x_pred.reshape(x_pred.shape[0], 1, x_pred.shape[1])

# -----------------------------------------------------------------------------------------------------
#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, AveragePooling1D, Activation, Flatten
from tensorflow.keras.layers import LeakyReLU, Concatenate, MaxPool1D, LSTM

model = Sequential()
# model.add(Conv1D(120, 2, 1, padding='same', input_shape=(x_train.shape[1:])))
model.add(LSTM(90, input_shape=(x_train.shape[1:])))
model.add(Activation('relu'))
# model.add(LeakyReLU())
model.add(Dropout(0.2))
# model.add(Conv1D(80, 2, 1, padding='same'))
# model.add(LSTM(80))
# model.add(Activation('relu'))
# model.add(LeakyReLU())
# model.add(MaxPooling1D(2))
# model.add(AveragePooling1D(2))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(180))
# model.add(Activation('relu'))
# model.add(Dense(90))
# model.add(Activation('relu'))
# model.add(Dense(60))
# model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(LeakyReLU())
model.add(Dense(8))
# model.add(LeakyReLU())
model.add(Activation('softmax'))

'''
#모델 구성
input1 = Input(shape=(x_train.shape[1:]))
dense1 = Dense(360, activation='relu')(input1)
# dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(360, activation='relu')(dense1)
# dropout1 = Dropout(0.2)(dense1)
dense3 = Dense(240)(dense2)
# dropout1 = Dropout(0.2)(dense1)
dense4 = Dense(180)(dense3)
dense5 = Dense(180)(dense4)
dense6 = Dense(90)(dense5)
dense7 = Dense(40)(dense6)
dense8 = Dense(10)(dense7)
dense9 = Dense(10)(dense8)
output = Dense(8, activation='softmax')(dense9)
model = Model(inputs = input1, outputs = output)
'''

'''
model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 1, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 1, padding="same")(x)
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

    # x = AveragePooling1D(pool_size=2, strides=1)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 8)
'''
model.summary()

# -----------------------------------------------------------------------------------------------------
#컴파일, 훈련
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.1), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(monitor='loss', patience=8, mode='min', restore_best_weights=True)
lr = ReduceLROnPlateau(factor = 0.5, patience = 4, verbose = 1)

model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_split=0.3, verbose=1, callbacks=[stop, lr])
