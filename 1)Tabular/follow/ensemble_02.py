# https://www.kaggle.com/kalashnimov/tps-apr-simple-ensemble

# -----------------------------------------------------------
# Load libraries

import numpy as np 
import pandas as pd 

# Read data

sub = pd.read_csv('D:/kaggle/sample_submission.csv')

sub1 = pd.read_csv('D:/kaggle/ensemble2/ensemble_01.csv') 
# sub2 = pd.read_csv('D:/kaggle/ensemble2/ensemble_02.csv') 
sub3 = pd.read_csv('D:/kaggle/ensemble2/ensemble_03.csv') 
sub4 = pd.read_csv('D:/kaggle/ensemble2/ensemble_04.csv')

# Submit
res = (2*sub1['Survived'] + sub3['Survived'] + 2*sub4['Survived'])/5
sub.Survived = np.where(res > 0.5, 1, 0).astype(int)

sub.to_csv("D:/kaggle/sub/submission_04.csv", index = False)
sub['Survived'].mean()

print('==== done ====')