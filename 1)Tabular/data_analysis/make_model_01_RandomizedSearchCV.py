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
df_train = pd.read_csv('D:/kaggle/0427_train.csv')
df_test = pd.read_csv('D:/kaggle/0427_test.csv')

print(df_train[:5])
print(df_test[:5])

# survived 나눠서 데이터와 라벨로 지정
x_train = df_train.drop('Survived', axis=1).values
y_train = df_train['Survived'].values
x_test = df_test.values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.3, random_state=519)

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# (80000, 7) (80000,)
# (20000, 7) (20000,)

# -----------------------------------------------------------------------------------------------------
# 모델 구성

#2. 모델 구성
def build_model(drop=0.2, optimizer='adam'):
    inputs = Input(shape=(x_train.shape[1]), name='input')
    x = Dense(256, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden2')(x)
    x = Dense(64, activation='relu', name='hidden3')(x)
    x = Dense(32, activation='relu', name='hidden4')(x)
    x = Dense(32, activation='relu', name='hidden5')(x)
    outputs = Dense(2, activation='sigmoid', name='outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model
model2 = build_model()

# wrap 적용
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_model, verbose=1)

# 하이퍼파라미터 지정
def create_hyperparameters():
    batches = [16, 32, 8, 5]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.2, 0.3, 0.5]
    return {'batch_size': batches, 'optimizer': optimizers, 'drop': dropouts}
hyper = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search = RandomizedSearchCV(model2, hyper, cv = 3)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='loss', patience=8, mode='min', restore_best_weights=True)
file_dir = 'D:/kaggle/h5/ransear_01.h5'
lr = ReduceLROnPlateau(factor = 0.25, patience = 4, verbose = 1)
mc = ModelCheckpoint(filepath=file_dir, verbose=1, save_best_only=True)

search.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, lr, mc])

print('best_params_: ', search.best_params_)

# -----------------------------------------------------------------------------------------------------
# 제출물 예측
from keras.models import load_model

submission = pd.read_csv('D:/kaggle/sample_submission.csv')

model = load_model(file_dir)
prediction = model.predict(x_test)
prediction = np.argmax(prediction, axis=1)
print(prediction)

submission.to_csv('D:/kaggle/sub/submission_02.csv', index=False)

print('==== done ====')

# ========================================
# in kaggle

# submission_01
# 0.77653 932등

# submission_02 랜덤서치
# 0.34911
# 장난하나?

# submission_03 # 노 민맥스, 기본 댄스 모델
# 0.78114