# 결과물 평균

import numpy as np
import pandas as pd

'''
x = []
for i in range(1,5):
    df = pd.read_csv(f'D:/kaggle/sub/mean/sub_mean ({i}).csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'D:/kaggle/sub/mean/sub_mean ({i}).csv', index_col=0, header=0)
for i in range(10000):
    for j in range(4):
        a = []
        for k in range(1):
            a.append(x[j,i,k].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[k]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')

        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('D:/kaggle/sub/mean/submission_mean_01.csv')
'''

mean = pd.read_csv('D:/kaggle/sub/mean/submission_mean_01.csv', index_col=0, header=0)
submission = pd.read_csv('D:/kaggle/sub/submission_011.csv', index_col=0, header=0)

submission[:] = mean[:]

submission.to_csv('D:/kaggle/sub/mean/submission_mean_02.csv')