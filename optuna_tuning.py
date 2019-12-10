# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:16:36 2019

@author: wmy
"""
import numpy as np 
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import multiprocessing
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
#from gpfs.home.mw15m.custom.dist_cate import dist_cate

train = pd.read_csv('*.csv')
num_cores = multiprocessing.cpu_count()
print('core: ',num_cores)
#---------binary------------
myscore='auc'
myobj='binary'
target=train['target']
#features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]
#dc=dist_cate()#draw=True,save_path='./*.png')
#train=dc.fit_transform(train[features],target)
#outlier=train['outliers']
print('BINARY')
#--------------------------
#target = train['target'] 
#outlier=train['outliers']
#myobj='regression'
#myscore='rmse'
#print('regression!')
#-------------------------------------
from numba import jit
# fast roc_auc computation from: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True

feat_cols = [c for c in train.columns if c not in ['card_id', 'first_active_month',
                                                       'target','outliers',
#                                                       'feature_1','feature_2','feature_3'
                                                       ]]
#feat_cols = ['feature_1','feature_2','feature_3']
#cate_cols = ['feature_1','feature_2','feature_3']
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial,train,target,outliers,
#              catefeat,
              features,fold,obj):
    oof = np.zeros(len(train))
    param = {'objective': obj, "bagging_freq": 1,"bagging_seed": 11,'verbosity': -1,"boost_from_average": "false","n_jobs": num_cores-2,'metric': myscore,
         'boosting_type': trial.suggest_categorical('boosting', [
                                                                 'gbdt'
#                                                                         'dart', 
#                                                                         'goss'
                                                                 ]),
         'max_depth':trial.suggest_int('max_depth', 2, 14),
         'num_leaves': trial.suggest_int('num_leaves', 2, 50),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.0001,0.001,0.01,0.1]),
         'learning_rate': trial.suggest_uniform('learning_rate', 0.0, 1.0),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 500),
         'lambda_l1':trial.suggest_uniform('lambda_l1', 0.0, 1.0),
         'lambda_l2':trial.suggest_uniform('lambda_l2', 0.0, 1.0),
         'bagging_fraction':trial.suggest_uniform('bagging_fraction', 0.0, 1.0),
         'feature_fraction':trial.suggest_uniform('feature_fraction', 0.0, 1.0)
         }
    for fold_, (trn_idx, val_idx) in enumerate(fold.split(train,outliers)): 
        print(fold_)
        trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                       label=target.iloc[trn_idx],
#                          categorical_feature=catefeat
                      )
        val_data = lgb.Dataset(train.iloc[val_idx][features],
                       label=target.iloc[val_idx],
#                           categorical_feature=catefeat
                      )
        

    
#        if param['boosting_type'] == 'dart':
#            param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
#            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
#        if param['boosting_type'] == 'goss':
#            param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
#            param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])
    
        num_round = 1000000
        gbm = lgb.train(param,
                        trn_data,
                        num_round,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=5000,                        
                        feval=eval_auc,
                        early_stopping_rounds = 3000
                        )
        oof[val_idx] = gbm.predict(train.iloc[val_idx][features], num_iteration=gbm.best_iteration)
        if obj=='binary':
            score=roc_auc_score(target,oof)
        else:           
            score=-(mean_squared_error(oof, target)**0.5)
    return score


if __name__ == '__main__':
    study = optuna.create_study()
    optuna.logging.set_verbosity(10)
    study.optimize(lambda trial: objective(trial,train,target,target,
#                                           cate_cols,
                                           feat_cols,folds,myobj),n_jobs=1,timeout =(60*3)*60)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    print('features')