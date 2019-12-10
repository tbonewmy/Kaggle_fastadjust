# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 12:49:14 2019

@author: wmy
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
#import warnings
from sklearn.linear_model import LinearRegression
import time
from sklearn.neighbors import LocalOutlierFactor
#import sys
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
#import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from scipy.stats import boxcox
from scipy.special import inv_boxcox
np.random.seed(4590)
import gc

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

#*-split data
def leftover(data):
    fini=data[~data[notcommon].isnull().all(axis=1)]
    lef=data.loc[data[notcommon].isnull().all(axis=1),hist_columns]
    return fini,lef
finished1_h,left1_h=leftover(historical_transactions)
finished1_n,left1_n=leftover(new_transactions)



def timefeature(df):
    truedate = pd.to_datetime(df['purchase_date'])
    df['purchase_date']=truedate
    df['year'] = truedate.dt.year
    df['weekofyear'] = truedate.dt.weekofyear
    df['month'] = truedate.dt.month
    df['dayofweek'] = truedate.dt.dayofweek
    df['weekend'] = (truedate.dt.weekday >=5).astype(int)
    df['hour'] = truedate.dt.hour
    df['month_diff'] = ((pd.to_datetime(datetime.date.today()) - truedate).dt.days)//30
    df['month_diff'] += df['month_lag']
    return df
  

def aggregate_transactions(dataset):
    agg_func = {
    'category_1': ['sum', 'mean'],
    'category_2_1.0': ['mean','sum'],
    'category_2_2.0': ['mean','sum'],
    'category_2_3.0': ['mean','sum'],
    'category_2_4.0': ['mean','sum'],
    'category_2_5.0': ['mean','sum'],
    'category_2_N': ['mean','sum'],
    'category_3_A': ['mean','sum'],
    'category_3_B': ['mean','sum'],
    'category_3_C': ['mean','sum'],
    'category_3_N': ['mean','sum'],
    'state_id': ['nunique'],
    'city_id': ['nunique'],
    'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
    'installments': ['sum', 'mean', 'max', 'min', 'std'],
    'month': ['mean', 'max', 'min', 'std','nunique'],
    'purchase_date': ['min', 'max'],
    'month_lag': ['nunique','mean', 'max', 'min', 'std'],
    'month_diff': ['mean','max', 'min', 'std'],
    'weekend' : ['sum', 'mean']
    }
    for c in merchant_uniqcol:
        if c=='merchant_group_id':
            agg_func[c]=['nunique']
        elif c[:11]=='most_recent' or c=='category_4':
            agg_func[c]=['sum', 'mean']
        else:
            agg_func[c]=['sum', 'mean', 'max', 'min', 'std']
    for col in ['hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
        agg_func[col] = ['nunique'] 
    for col in ['category_1','category_2','category_3','category_4']:
        dataset[col+'_mean'] = dataset.groupby([col])['purchase_amount'].transform('mean')
        agg_func[col+'_mean'] = ['mean','max', 'min', 'std']      
    agg_history = dataset.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)  
    df = dataset.groupby('card_id').size().reset_index(name='transactions_count')
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    timedate_span=agg_history['purchase_date_max'] - agg_history['purchase_date_min']
    agg_history['purchase_date_span'] = timedate_span.dt.days
    agg_history['purchase_date_avg_span'] = agg_history['purchase_date_span']/agg_history['transactions_count']
    agg_history['purchase_date_uptonow_max'] = (pd.to_datetime(datetime.date.today()) - agg_history['purchase_date_max']).dt.days
    agg_history['purchase_date_uptonow_min'] = (pd.to_datetime(datetime.date.today()) - agg_history['purchase_date_min']).dt.days
    agg_history['purchase_date_max'] = agg_history['purchase_date_max'].astype(np.int64) * 1e-9
    agg_history['purchase_date_min'] = agg_history['purchase_date_min'].astype(np.int64) * 1e-9
#----new feature
    agg_history['avg_purchase_amount_per_act_month']=agg_history['purchase_amount_sum']/agg_history['month_lag_nunique']
    agg_history['card_active_ratio']=agg_history['month_lag_nunique']/(timedate_span.dt.days//30 + 1)
    agg_history['merchant_active_ratio3']=agg_history['active_months_lag3_mean']/3 
    agg_history['merchant_active_ratio6']=agg_history['active_months_lag6_mean']/6
    agg_history['merchant_active_ratio12']=agg_history['active_months_lag12_mean']/12
    agg_history['card_merch_act_ratio3']=agg_history['card_active_ratio']/agg_history['merchant_active_ratio3']
    agg_history['card_merch_act_ratio6']=agg_history['card_active_ratio']/agg_history['merchant_active_ratio6']
    agg_history['card_merch_act_ratio12']=agg_history['card_active_ratio']/agg_history['merchant_active_ratio12']
#    agg_history[['merchant_profit3_mean','merchant_profit6_mean','merchant_profit12_mean']]=agg_history[['merchant_profit3_mean','merchant_profit6_mean','merchant_profit12_mean']].replace(0,1e-15)
    agg_history['card_merch_gain_ratio3']=agg_history['avg_purchase_amount_per_act_month']/(agg_history['merchant_profit3_mean']+1e-15)
    agg_history['card_merch_gain_ratio6']=agg_history['avg_purchase_amount_per_act_month']/(agg_history['merchant_profit6_mean']+1e-15)
    agg_history['card_merch_gain_ratio12']=agg_history['avg_purchase_amount_per_act_month']/(agg_history['merchant_profit12_mean']+1e-15)
    agg_history['card_merch_biasgain_ratio3']=agg_history['purchase_amount_mean']/(agg_history['merchant_profit3_mean']+1e-15)
    agg_history['card_merch_biasgain_ratio6']=agg_history['purchase_amount_mean']/(agg_history['merchant_profit6_mean']+1e-15)
    agg_history['card_merch_biasgain_ratio12']=agg_history['purchase_amount_mean']/(agg_history['merchant_profit12_mean']+1e-15)
    agg_history['energy_ratio3']=agg_history['month_lag_nunique']/agg_history['active_months_lag3_mean']
    agg_history['energy_ratio6']=agg_history['month_lag_nunique']/agg_history['active_months_lag6_mean']
    agg_history['energy_ratio12']=agg_history['month_lag_nunique']/agg_history['active_months_lag12_mean']
#    agg_history=agg_history.fillna(0)
    return agg_history
	
	
def convert2float32(data):
    inf_list=['numerical_1','numerical_2','active_months_lag12','avg_sales_lag12','avg_purchases_lag12','merchant_profit3','merchant_profit6','merchant_profit12']#,'avg_purchases_lag3','active_months_lag12', 'avg_purchases_lag6','avg_purchases_lag12']
    data[inf_list]=data[inf_list].astype('float32')
    return data


def aggregate_per_month(dataset):
    grouped = dataset.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)
#    intermediate_group=intermediate_group.fillna(0)
    intermediate_group.drop(['month_lag'],axis=1,inplace=True)
    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
#    final_group.drop(['month_lag_mean','month_lag_std'],axis=1,inplace=True)
#    final_group=final_group.fillna(0)
    return final_group


def successive_aggregates(df, field1, field2):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
#    u.fillna(0,inplace=True)
    return u
	
	
def addFeatures(data):
    additional_fields = successive_aggregates(data, 'category_1', 'purchase_amount')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_2', 'purchase_amount'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_3', 'purchase_amount'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'installments', 'purchase_amount'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'city_id', 'purchase_amount'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_1', 'installments'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_2', 'installments'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_3', 'installments'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_2', 'merchant_profit3'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_2', 'merchant_profit6'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_2', 'merchant_profit12'),
                                                on = 'card_id', how='left')    
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_4', 'merchant_profit3'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_4', 'merchant_profit6'),
                                                on = 'card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(data, 'category_4', 'merchant_profit12'),
                                                on = 'card_id', how='left') 
    return additional_fields


def shift_division(group):
    group=(group.shift(1)/group).dropna()
    return group
def shift_minues(group):
    group=(group.shift(1)-group).dropna()
    return group
purchase_ratio=monthly_purchase.groupby('card_id')['purchase_amount'].apply(shift_division)


#::::::::::::: features_* mix:https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
def featmix(mix_f):
    for f in ['feature_1','feature_2','feature_3']:
        order_label = train.groupby([f])[mix_f].mean()
        train[f+'_'+mix_f] = train[f].map(order_label)
        test[f+'_'+mix_f] = test[f].map(order_label)
featmix('outliers')
featmix('hist_purchase_amount_mean')
