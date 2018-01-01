from __future__ import print_function
from __future__ import division
import warnings; warnings.filterwarnings('ignore')

import time
import random
from math import log

import numpy as np
import pandas as pd

random.seed(125)
np.random.seed(137)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout

from numerapi.numerapi import NumerAPI
from sklearn.externals import joblib

def import_data_sets():
    ## import data
    train = pd.read_csv('/input/numerai_training_data.csv', index_col=0).drop('data_type', axis=1)
    df = pd.read_csv('/input/numerai_tournament_data.csv', index_col=0)
    valid = df.loc[df['data_type']=='validation'].drop('data_type', axis=1)
    test = df.loc[df['data_type']=='test'].drop('data_type', axis=1)
    live = df.loc[df['data_type']=='live'].drop('data_type', axis=1)
    
    ## extract feature columns from target and meta data
    feature_cols = [f for f in train.columns if "feature" in f]    
    x_train = train[feature_cols]
    x_val = valid[feature_cols]
    x_test = test[feature_cols]
    x_live = live[feature_cols]
    y_train = train['target']
    y_val = valid['target']

    ## get eras
    train_eras = train['era'].values
    val_eras = valid['era'].values
    
    return(x_train, y_train, train_eras, x_val, y_val, val_eras, x_test, x_live, feature_cols)

def calc_consistency(labels, preds, eras):
    """ Calculate the consistency score.

    Args:
        labels: (np array) The correct class ids
        preds:  (np array) The predicted probabilities for class 1
        eras:   (np array) The era each sample belongs to
    """
    unique_eras = np.unique(eras)
    better_than_random_era_count = 0
    for era in unique_eras:
        this_era_filter = [eras == era]
        logloss = log_loss(labels[this_era_filter], preds[this_era_filter])
        
        print(logloss)
        
        if logloss < -np.log(0.5):
            better_than_random_era_count += 1

    consistency = better_than_random_era_count / float(len(unique_eras)) * 100
    print(consistency)
     
    
def get_spacekitty_sgd_pipe():
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rbf', RBFSampler(random_state=73)),
        ('classify', SGDClassifier(loss='log', penalty='l2'))
    ])

    N_COMPONENTS = [7]
    GAMMA = [0.1]
    ALPHA = [0.001]

    param_grid = [{
            'rbf__n_components': N_COMPONENTS,
            'rbf__gamma': GAMMA,
            'classify__alpha': ALPHA
        }]

    sgd_grid = GridSearchCV(pipe, cv=5, n_jobs=1,
                           param_grid=param_grid, verbose=2, scoring='neg_log_loss')
    
    return(sgd_grid)

def get_spacekitty_xgb_pipe():
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classify', XGBClassifier(eval_metric='logloss', n_jobs=-1, silent=False))
    ])

    N_EST = [60]
    LR = [0.01]
    DEPTH = [10]

    param_grid = [{
            'classify__n_estimators': N_EST,
            'classify__learning_rate': LR,
            'classify__max_depth': DEPTH        
        }]

    xgb_grid = GridSearchCV(pipe, cv=5, n_jobs=1,
                            param_grid=param_grid, verbose=2, scoring='neg_log_loss')
    
    return(xgb_grid)

def get_spacekitty_rf_pipe():
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('reduce_dim', PCA()),
        ('classify', RandomForestClassifier(n_jobs=-1, verbose=2, random_state=73))
    ])

    N_DIM = [10]
    N_EST = [50]
    MAX_DEPTH = [5]

    param_grid = [{
            'reduce_dim__n_components': N_DIM,
            'classify__n_estimators': N_EST,
            'classify__max_depth': MAX_DEPTH
        }]

    rf_grid = GridSearchCV(pipe, cv=5, n_jobs=1,
                            param_grid=param_grid, verbose=2, scoring='neg_log_loss')
    
    return(rf_grid)

def get_spacekitty_nn_model(input_size):
    model = Sequential()
    model.add(Dense(1048, activation='relu', input_dim=input_size))
    model.add(Dense(1048, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

def apply_vote(row):
    mean = np.mean(row)    
    if(mean > .5):
        return np.max(row)
    else:
        return np.min(row)


    
    