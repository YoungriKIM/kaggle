# age predict를 위한 저장 파일

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

# -----------------------------------------------------------------------------------------------------
# 데이터 지정
df_train = pd.read_csv('D:/kaggle/train.csv')
df_test = pd.read_csv('D:/kaggle/test.csv')

# 형제자매 + 부모자식 = 가족크기
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함

df_train['IsAlone'] = np.where(df_train['FamilySize'] <= 1, 1, 0)
df_test['IsAlone'] = np.where(df_test['FamilySize'] <= 1, 1, 0)

# test의 Fare null값에 평균값 넣기
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()

# Fare 값의 규모는 큰데 분포가 치우쳐져 있어서 log 씌우기
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

# Calcule SameFirstName
df_train['FirstName'] = df_train['Name'].apply(lambda x:x.split(', ')[0])
df_train['n'] = 1
gb = df_train.groupby('FirstName')
df_names = gb['n'].sum()
df_train['SameFirstName'] = df_train['FirstName'].apply(lambda x:df_names[x])

df_test['FirstName'] = df_test['Name'].apply(lambda x:x.split(', ')[0])
df_test['n'] = 1
gb = df_test.groupby('FirstName')
df_names = gb['n'].sum()
df_test['SameFirstName'] = df_test['FirstName'].apply(lambda x:df_names[x])


# -----------------------------------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------------------------------
# age의 null 값을 채워보자
'''
# null 값 개수 확인
print('Age has', sum(df_train['Age'].isnull()), 'Null values')
# Age has 3292 Null values

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
'''
# -----------------------------------------------------------------------------------------------------

# Change Age(continuous to categorical)
# age를 카테고리로 나누어서 진행하자

df_train['Age'].fillna(120, inplace=True)
df_test['Age'].fillna(120, inplace=True)


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
    elif x < 90:
        return 7
    else:
        return 8    
    
df_train['Age_cat'] = df_train['Age'].apply(category_age)
df_test['Age_cat'] = df_test['Age'].apply(category_age)

# 이제 중복되는 Age_cat 컬럼과 원래 컬럼 Age 를 제거
df_train.drop(['Age'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)

# -----------------------------------------------------------------------------------------------------
# Embarked 

print('Embarked has', sum(df_train['Embarked'].isnull()), 'Null values')
# Embarked has 250 Null values

# 가장 많은 탑승객이 있었던 S로 채워주자
df_train['Embarked'].fillna('S', inplace=True)
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
# 모델을 위한 전처리
# one-hot 인코딩

df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

# -----------------------------------------------------------------------------------------------------
# 필요없는 것들 버리기
print(df_train[:5])
print(df_test[:5])

df_train.drop(['PassengerId', 'Name','FirstName', 'n', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin', 'SameFirstName', 'Embarked_0', 'Embarked_1', 'Embarked_2'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'FirstName', 'n', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin', 'SameFirstName', 'Embarked_0.0', 'Embarked_1.0', 'Embarked_2.0'], axis=1, inplace=True)

# -----------------------------------------------------------------------------------------------------
print(df_train[:5])
print(df_test[:5])
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# .csv 로 저장하자

df_train.to_csv('D:/kaggle/0427_train_age_02.csv', index=False)
df_test.to_csv('D:/kaggle/0427_test_age_02.csv', index=False)

print('==== save done ====')
