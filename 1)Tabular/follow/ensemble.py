# https://www.kaggle.com/kalashnimov/tps-apr-simple-ensemble

# -----------------------------------------------------------
# Load libraries

import numpy as np 
import pandas as pd 

# Read data

sub = pd.read_csv('D:/kaggle/sample_submission.csv')

sub1 = pd.read_csv('D:/kaggle/ensemble/tps04-sub-006.csv') 
sub2 = pd.read_csv('D:/kaggle/ensemble/AutoWoE_submission_combo.csv') 
sub3 = pd.read_csv('D:/kaggle/ensemble/submission_pseudo_test.csv') 
sub4 = pd.read_csv('D:/kaggle/ensemble/voting_submission.csv')

# Submit
res = (2*sub1['Survived'] + sub2['Survived'] + sub3['Survived'] + 2*sub4['Survived'])/6
sub.Survived = np.where(res > 0.5, 1, 0).astype(int)

sub.to_csv("D:/kaggle/sub/submission_01.csv", index = False)
sub['Survived'].mean()

print('==== done ====')