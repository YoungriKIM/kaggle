# https://www.kaggle.com/gomes555/tps-apr2021-catboost-run-pseudo-label

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

path = "D:/kaggle/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
pseudo_label = pd.read_csv('D:/kaggle/sub/pseudo_label_02.csv', index_col=0)
submission = pd.read_csv(path+'sample_submission.csv')

# pseudo-label from https://www.kaggle.com/hiro5299834/tps-apr-2021-voting-pseudo-labeling
test['Survived'] = [x for x in pseudo_label.Survived]

# ----------------------------------------------------

# Calcule SameFirstName

train['FirstName'] = train['Name'].apply(lambda x:x.split(', ')[0])
train['n'] = 1
gb = train.groupby('FirstName')
df_names = gb['n'].sum()
train['SameFirstName'] = train['FirstName'].apply(lambda x:df_names[x])

test['FirstName'] = test['Name'].apply(lambda x:x.split(', ')[0])
test['n'] = 1
gb = test.groupby('FirstName')
df_names = gb['n'].sum()
test['SameFirstName'] = test['FirstName'].apply(lambda x:df_names[x])

# To preprocess

data = pd.concat([train, test], axis=0)

# Before fill missing
data['AnyMissing'] = np.where(data.isnull().any(axis=1) == True, 1, 0)

# Family
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = np.where(data['FamilySize'] <= 1, 1, 0)

# Cabin
data['Has_Cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
data['Cabin'] = data['Cabin'].fillna('X').map(lambda x: x[0].strip())
cabin_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
             'F': 6, 'G': 7, 'T': 1, 'X': 8}
data['Cabin'] = data['Cabin'].str[0].fillna('X').replace(cabin_map)

# Embarked
#map_Embarked = train.Embarked.mode().item()
data['Embarked'] = data['Embarked'].fillna("No")
conditions = [
    (data['Embarked']=="S"),
    (data['Embarked']=="Q"),
    (data['Embarked']=="C"),
    (data['Embarked']=="No")
]
choices = [0, 1, 2, -1]
data["Embarked"] = np.select(conditions, choices)
data['Embarked'] = data['Embarked'].astype(int)

# Name
data['SecondName'] = data.Name.str.split(', ', 1, expand=True)[1] # to try
data['IsFirstNameDublicated'] = np.where(data.FirstName.duplicated(), 1, 0)

# Fare
data['Fare'] = data['Fare'].fillna(train['Fare'].median())
# train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# [(0.679, 10.04] < (10.04, 24.46] < (24.46, 33.5] < (33.5, 744.66]]
# From original Titanic:
conditions = [
    (data['Fare'] <= 7.91),
    ((data['Fare'] > 7.91) & (data['Fare'] <= 14.454)),
    ((data['Fare'] > 14.454) & (data['Fare'] <= 31)),
    (data['Fare'] > 31)
]

choices = [0, 1, 2, 3]
data["Fare"] = np.select(conditions, choices)
data['Fare'] = data['Fare'].astype(int)

# Fix Ticket
# data['TicketNum'] = data.Ticket.str.extract(r'(\d+)').\
#                     astype('float64', copy=False) # to_try
data['Ticket'] = data.Ticket.str.replace('\.','', regex=True).\
                    str.replace('(\d+)', '', regex=True).\
                    str.replace(' ', '', regex=True).\
                    replace(r'^\s*$', 'X', regex=True).\
                    fillna('X')

#data['Ticket'] = data['Ticket'].astype('category').cat.codes # to_try

# Age 
conditions = [
    ((data.Sex=="female")&(data.Pclass==1)&(data.Age.isnull())),
    ((data.Sex=="male")&(data.Pclass==1)&(data.Age.isnull())),
    ((data.Sex=="female")&(data.Pclass==2)&(data.Age.isnull())),
    ((data.Sex=="male")&(data.Pclass==2)&(data.Age.isnull())),
    ((data.Sex=="female")&(data.Pclass==3)&(data.Age.isnull())),
    ((data.Sex=="male")&(data.Pclass==3)&(data.Age.isnull()))
]
choices = data[['Age', 'Pclass', 'Sex']].\
            dropna().\
            groupby(['Pclass', 'Sex']).\
            mean()['Age']

data["Age"] = np.select(conditions, choices)

conditions = [
    (data['Age'].le(16)),
    (data['Age'].gt(16) & data['Age'].le(32)),
    (data['Age'].gt(32) & data['Age'].le(48)),
    (data['Age'].gt(48) & data['Age'].le(64)),
    (data['Age'].gt(64))
]
choices = [0, 1, 2, 3, 4]

data["Age"] = np.select(conditions, choices)

# Sex
data['Sex'] = np.where(data['Sex']=='male', 1, 0)

# Drop columns
data = data.drop(['Name', 'n'], axis = 1)

# Transform object to category
#for col in data.columns[data.dtypes=='object'].tolist():
#    data.loc[:,col] = data.loc[:,col].astype('category')

# ----------------------------------------------------

# Splitting into train and test
train = data.iloc[:train.shape[0]]
test = data.iloc[train.shape[0]:].drop(columns=['Survived'])

# ----------------------------------------------------

lab_cols = ['Pclass','Age', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target = 'Survived'

features_selected = ['Pclass', 'Sex', 'Age','Embarked','Parch','SibSp','Fare','Cabin','Ticket','SameFirstName']

X = data.drop(target, axis=1)
X = X[features_selected]
y = data[target]

test = test[features_selected]

# ----------------------------------------------------

def kfold_prediction(X, y, X_test, K, od_wait = 500):
    
    yp = pd.DataFrame()
    trs = []
    acc_trs = []
    
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=314)
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n FOLD {i} ...")
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[test_idx]
        y_val = y.iloc[test_idx]
        
        # https://catboost.ai/docs/concepts/parameter-tuning.html
        params = {'iterations': 10000,
                  'use_best_model':True ,
                  'eval_metric': 'AUC', # 'Accuracy'
                  'loss_function':'Logloss',
                  'od_type':'Iter',
                  'od_wait':od_wait,
                  'depth': 6, # [4, 10]
                  'l2_leaf_reg': 3,
                  # random_strength ??
                  'bootstrap_type': 'Bayesian',
                  'bagging_temperature': 2,
                  'max_bin': 254,
                  'grow_policy': 'SymmetricTree',
                  'cat_features': lab_cols,
                  'verbose': od_wait,
                  'random_seed': 314
         }
        
        #params = {'loss_function':'Logloss',
        #          'eval_metric': 'AUC', # 'Accuracy'
        #          'od_wait':od_wait,
        #          'od_type':'Iter', 
        #          'n_estimators': 10000,
        #          'cat_features': lab_cols,
        #          'verbose': od_wait,
        #          'random_seed': 314
        # }
        
        clf = CatBoostClassifier(**params)
        
        model_fit = clf.fit(X_train,y_train,
                            eval_set=[(X_train, y_train), (X_val, y_val)],
                            use_best_model=True,
                            plot=False)
        
        yp_val = model_fit.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, np.where(yp_val>0.5, 1, 0))
        print(f"- Accuracy before : {acc} ...")
        
        # Moving threshold
        thresholds = np.arange(0.0, 1.0, 0.01)
        accuracy_scores = []
        for thresh in thresholds:
            accuracy_scores.append(
                accuracy_score(y_val, [1 if m>thresh else 0 for m in yp_val]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max() 
        max_accuracy_threshold =  thresholds[accuracies.argmax()] 
        trs = trs + [max_accuracy_threshold]
        
        print("- Max accuracy threshold: "+str(max_accuracy_threshold))
        
        acc = accuracy_score(y_val, 
                             np.where(yp_val>max_accuracy_threshold, 1, 0)) 
        acc_trs = acc_trs + [acc]
        print(f"- Accuracy after: {acc} !")
        
        yp_test = model_fit.predict_proba(X_test)[:, 1]
        yp_fold = pd.DataFrame({
            'fold'+str(i): np.where(yp_test>max_accuracy_threshold, 1, 0)})
        
        yp = pd.concat([yp, yp_fold], axis=1)
    
    return yp, trs, acc_trs

yp, trs, acc = kfold_prediction(X, y, test, 5)

print('Model with train + pseudo train')
print("Final mean and std accuracy: ", np.mean(acc), round(np.std(acc), 5))
print("Final mean and std accuracy with Threshold: ", np.mean(trs), round(np.std(trs), 5))

# ----------------------------------------------------

def vote(r, columns):
    """https://www.kaggle.com/belov38/catboost-lb/"""
    ones = 0
    zeros = 0
    for i in columns:
        if r[i]==0:
            zeros+=1
        else:
            ones+=1
    if ones>zeros:
        return 1
    else:
        return 0

# ----------------------------------------------------

submission_pseudo = pd.DataFrame({
    'PassengerId': test.index,
    'Survived':yp.apply(lambda x:vote(x, yp.columns.tolist()),axis=1)
})

submission_pseudo.to_csv('D:/kaggle/sub/note_02.csv', index = False)