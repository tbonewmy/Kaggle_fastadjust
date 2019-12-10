# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:36:53 2019

@author: wmy
"""
import numpy as np 
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
#import tensorflow as tf
#import warnings
#import time
#import sys
#import datetime
#import seaborn as sns
from sklearn.metrics import mean_squared_error
np.random.seed(4590)
import multiprocessing

#test = pd.read_csv('./*.csv')
train = pd.read_csv('/*.csv')
#feat_cols=pd.read_csv('/*.csv')
#feat_cols=feat_cols.iloc[:,0].values.tolist()
num_cores = multiprocessing.cpu_count()
print('core: ',num_cores)
#del train['target']
#---------binary------------
myscore=None
myobj='binary'
target=train['target']

model=lgb.LGBMClassifier(boosting_type='gbdt',random_state=4590,
                    n_jobs=num_cores,
#                        max_depth=-1,
                    bagging_freq= 1, bagging_seed= 11,verbosity=0)    
print('BINARY')
#--------------------------
#target = train['target'] 
#myobj='regression'
#myscore='neg_mean_squared_error'
#param = {'num_leaves': 31,
#         'min_data_in_leaf': 30, 
#         'objective':'regression',
#         'max_depth': 10,
#         'learning_rate': 0.01,
#         "min_child_samples": 20,
#         "boosting": "gbdt",
#         "feature_fraction": 1,
#         "bagging_fraction": 0.9 ,
#         "metric": 'rmse',
#         "lambda_l1": 0.1,
#         "lambda_l2":0.1,
#         "verbosity": -1,
#         "nthread": num_cores-2,
#         "random_state": 4590}
#model=lgb.LGBMModel(boosting_type='gbdt',random_state=4590,
#                    n_jobs=num_cores,objective=myobj,
##                        max_depth=-1,
#                    bagging_freq= 1, bagging_seed= 11,verbosity=0)
#model.set_params(**param)
#print('regression!')

#space  = {
#        'max_depth':Integer(6, 15),
#          'num_leaves':Integer(6, 100),
#          'min_child_samples':Integer(10, 150),
#          'min_data_in_leaf':Integer(10, 150), 
#          'learning_rate':Categorical([0.001,0.01,0.1]),
#          'lambda_l1':Real(0,1),
#          'lambda_l2':Real(0,1),
#          'feature_fraction':Real(0.001,1),
#          'bagging_fraction':Real(0.001,1),
#          'n_estimators':Integer(500, 2500)
#         }
if __name__ == "__main__":
    feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=0)
    feat_cols = [c for c in train.columns if c not in ['card_id', 'first_active_month','target','outliers',
#                                                       'feature_1','feature_2','feature_3'
                                                        ]]
#    cate_cols = ['feature_1','feature_2','feature_3']
#    for c in cate_cols:
#        train[c]=train[c].astype('category')
 
    feat_selector.fit(train[feat_cols], train['target'])
    selected = train.iloc[:, feat_selector.support_]
selected.to_csv("/*.csv", index=False)
print('boruta')

