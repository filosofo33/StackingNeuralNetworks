from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
import matplotlib.ticker as tkr
import matplotlib.dates as mdates
import datetime
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from decimal import Decimal
from cleanf import *

class NanDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops rows with NaN values in specified column"""
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.drop('sku', axis=1)
        return X.dropna(subset=[self.column])

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects specified columns from dataframe"""
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.feature_names].values

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Transformer that handles outliers based on percentile"""
    def __init__(self, percentile):
        self.percentile = percentile
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X

def normalize_minmax(X):
    """Normalize features to 0-1 range"""
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(X)

def prepare_matrix(df, is_training=True):
    """Convert dataframe to numpy matrices"""
    X = df.drop('target', axis=1).values
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    
    if is_training:
        y = df['target'].values
        y = y.reshape(-1, 1).astype(np.float32)
    else:
        y = []
        
    return X, y

def split_data(df, scaler=None):
    """Split data into features and labels and optionally scale"""
    X, y = prepare_matrix(df, is_training=True)
    return X, y, scaler

def split_test_data(df, scaler=None):
    """Prepare test data"""
    X, _ = prepare_matrix(df, is_training=False)
    return X, scaler

def print_full_dataframe(df):
    """Print full dataframe without truncation"""
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def save_predictions(predictions, output_file, input_file, ids):
    """Save predictions to CSV file"""
    print(predictions.shape)
    np.set_printoptions(suppress=True)

    # Add IDs column
    results = np.c_[ids, predictions]
    np.savetxt(output_file, results, fmt="%d,%10.8f", delimiter=",")

    # Calculate log loss if validation data available
    df_val = pd.read_csv(input_file)
    df_val = df_val.dropna()
    df_val = clean_dataframe(df_val, remove_id_era=True)

    X_val, y_val = split_data(df_val)[0:2]
    n_rows = y_val.shape[0]

    pred_val = predictions[:n_rows, 1]
    print(f"Log loss for {output_file}: {log_loss(y_val, pred_val)}")

    # Add header
    with open(output_file, 'r') as f:
        data = f.read()
    with open(output_file, 'w') as f:
        f.write("id,probability\n" + data)

    return True

def prepare_input_data(csv_prefix, df):
    """Prepare and merge input data"""
    df.reset_index(drop=True, inplace=True)

    correlation = df.corr()
    print_full_dataframe(correlation["target"].sort_values(ascending=False))
    print(df.info())
    print(df.describe())

    # Merge prediction files
    df_pred1 = pd.read_csv(f'{csv_prefix}1.csv')
    df_pred2 = pd.read_csv(f'{csv_prefix}2.csv') 
    df_pred3 = pd.read_csv(f'{csv_prefix}3.csv')

    df = df_pred1.merge(df_pred2, on='id')
    df = df.merge(df_pred3, on='id')
    df = df.merge(df, on='id')

    print(df.info())
    print(df.describe())

    df = clean_dataframe(df, remove_id_era=False)

    print(df.shape)
    correlation = df.corr()
    print_full_dataframe(correlation["target"].sort_values(ascending=False))
    print("Dataset ready!")
    
    return df