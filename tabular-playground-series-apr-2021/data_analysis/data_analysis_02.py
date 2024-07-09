from math import nan
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------------------------------
# 데이터 지정
df_train = pd.read_csv('D:/kaggle/train.csv')
df_test = pd.read_csv('D:/kaggle/test.csv')

# 형제자매 + 부모자식 = 가족크기
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함

# test의 Fare null값에 평균값 넣기
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()

# Fare 값의 규모는 큰데 분포가 치우쳐져 있어서 log 씌우기
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

# -----------------------------------------------------------------------------------------------------
# Feature engineering
'''
# null 값을 채우자
# 그 전에 null 값을 확인하자
# train
for col in df_train.columns:
    msg = 'column: {:>12}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)

# column:  PassengerId     Percent of NaN value: 0.00%
# column:     Survived     Percent of NaN value: 0.00%
# column:       Pclass     Percent of NaN value: 0.00%
# column:         Name     Percent of NaN value: 0.00%
# column:          Sex     Percent of NaN value: 0.00%
# column:          Age     Percent of NaN value: 3.29%
# column:        SibSp     Percent of NaN value: 0.00%
# column:        Parch     Percent of NaN value: 0.00%
# column:       Ticket     Percent of NaN value: 4.62%
# column:         Fare     Percent of NaN value: 0.00%
# column:        Cabin     Percent of NaN value: 67.87%
# column:     Embarked     Percent of NaN value: 0.25%
# column:   FamilySize     Percent of NaN value: 0.00%

print('\n')
# test
for col in df_test.columns:
    msg = 'column: {:>12}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)

# column:  PassengerId     Percent of NaN value: 0.00%
# column:       Pclass     Percent of NaN value: 0.00%
# column:         Name     Percent of NaN value: 0.00%
# column:          Sex     Percent of NaN value: 0.00%
# column:          Age     Percent of NaN value: 3.49%
# column:        SibSp     Percent of NaN value: 0.00%
# column:        Parch     Percent of NaN value: 0.00%
# column:       Ticket     Percent of NaN value: 5.18%
# column:         Fare     Percent of NaN value: 0.00%
# column:        Cabin     Percent of NaN value: 70.83%
# column:     Embarked     Percent of NaN value: 0.28%
# column:   FamilySize     Percent of NaN value: 0.00%
'''
# -----------------------------------------------------------------------------------------------------
# age의 null 값을 채워보자

# null 값 개수 확인
print('Age has', sum(df_train['Age'].isnull()), 'Null values')
# Age has 3292 Null values

# df_train['Age'].fillna(101, inplace=True)

# print(df_train.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1))
# 성별과 클래스에 따른 나이를 추출함
#                mean  median
# Sex    Pclass
# female 1       47.0    49.0
#        2       39.3    40.0
#        3       32.0    28.0
# male   1       42.3    44.0
#        2       37.8    37.0
#        3       34.0    31.0

cols = ['Pclass', 'Sex']
age_class_sex = df_train.groupby(cols)['Age'].mean().reset_index()

df_train['Age'] = df_train['Age'].fillna(df_train[cols].reset_index().merge(age_class_sex, how='left', on=cols).set_index('index')['Age'])

print('Age has', sum(df_train['Age'].isnull()), 'Null values')
# Age has 0 Null values
# -----------------------------------------------------------------------------------------------------
# Embarked 

print('Embarked has', sum(df_train['Embarked'].isnull()), 'Null values')
# Embarked has 250 Null values

# 가장 많은 탑승객이 있었던 S로 채워주자
df_train['Embarked'].fillna('S', inplace=True)

# -----------------------------------------------------------------------------------------------------
# Change Age(continuous to categorical)
# age를 카테고리로 나누어서 진행하자

def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7    
    
df_train['Age_cat'] = df_train['Age'].apply(category_age)
df_test['Age_cat'] = df_test['Age'].apply(category_age)

# 이제 중복되는 Age_cat 컬럼과 원래 컬럼 Age 를 제거
df_train.drop(['Age'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)

# -----------------------------------------------------------------------------------------------------
# (string to numerical)
# 문자열을 숫자로

# Embarked
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 잘 되었다면 flase가 나와야 함
print(df_train['Embarked'].isnull().any())
# False

# Sex
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})

# -----------------------------------------------------------------------------------------------------
# 히트맵으로 상관관계 그리기

heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Age_cat']] 

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})
# plt.show()

del heatmap_data


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# 모델을 위한 전처리
# one-hot 인코딩

df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

# -----------------------------------------------------------------------------------------------------
# 필요없는 것들 버리기
print(df_train[:5])
print(df_test[:5])

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin'], axis=1, inplace=True)

# -----------------------------------------------------------------------------------------------------
# 머신러닝 모델 만들기
#importing all the required ML packages
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split 

# survived 나눠서 데이터와 라벨로 지정
X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values

# split
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2021)

# 훈련, 예측
model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
# 총 30000명 중 76.54% 정확도로 생존을 맞춤

# -----------------------------------------------------------------------------------------------------
# 학습된 모델의 Feature importance 를 확인하자
from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)

plt.figure(figsize=(12, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
# plt.show()

# -----------------------------------------------------------------------------------------------------
# 제출물 예측
submission = pd.read_csv('D:/kaggle/sample_submission.csv')

prediction = model.predict(X_test)
submission['Survived'] = prediction

submission.to_csv('D:/kaggle/sub/submission_01.csv', index=False)

print('==== done ====')

# ========================================
# in kaggle

# submission_01
# 0.77653 932등