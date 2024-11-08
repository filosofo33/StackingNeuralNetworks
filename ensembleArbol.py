from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
import matplotlib.ticker as tkr
import matplotlib.dates as mdates
import datetime
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
import sklearn
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import heapq
from decimal import Decimal
from sklearn.model_selection import train_test_split
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

def clean_features(df, remove_id_era=False):
    """
    Clean and preprocess features in the dataframe
    Args:
        df: Input dataframe
        remove_id_era: Whether to remove ID and era columns
    Returns:
        Cleaned dataframe
    """
    if remove_id_era:
        df = df.drop(['id', 'era'], axis=1)
    
    df = df.drop('data_type', axis=1)
    
    # Remove redundant features
    redundant_features = ['feature15', 'feature25', 'feature22', 'feature41', 
                         'feature45', 'feature7', 'feature16', 'feature38',
                         'feature29', 'feature40', 'feature14', 'feature43',
                         'feature36', 'feature20', 'feature28', 'feature17',
                         'feature27', 'feature26', 'feature6', 'feature2',
                         'feature13', 'feature48', 'feature46', 'feature39',
                         'feature44', 'feature42', 'feature34', 'feature18',
                         'feature8', 'feature10', 'feature30']
    
    df = df.drop(redundant_features, axis=1)

    # Add engineered features
    df['feature23_squared'] = df['feature23']**2
    df['feature49_squared'] = df['feature49']**2 
    df['feature12_squared'] = df['feature12']**2

    return df

class NanDropper(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.drop('sku', axis=1)
        return X.dropna(subset=[self.column])

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.feature_names].values

class OutlierHandler(BaseEstimator, TransformerMixin):
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
    X = df.drop('target', axis=1).values.astype(np.float32)
    
    if is_training:
        y = df['target'].values.reshape(-1, 1).astype(np.float32)
    else:
        y = []
        
    return X, y

def split_data(df, scaler=None):
    """Split data into features and target"""
    X, y = prepare_matrix(df, is_training=True)
    return X, y, scaler

def split_test_data(df, scaler=None):
    """Prepare test data"""
    X, _ = prepare_matrix(df, is_training=False)
    return X, scaler

def print_full_df(df):
    """Print full dataframe"""
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def save_predictions(predictions, ids, output_file, input_file):
    """Save predictions to CSV file and calculate logloss"""
    print(f"Predictions shape: {predictions.shape}")
    
    # Combine IDs with predictions
    results = np.column_stack([ids, predictions])
    np.savetxt(output_file, results, fmt="%s,%10.8f", delimiter=",")

    # Calculate logloss on validation data
    val_df = pd.read_csv(input_file)
    val_df = val_df.dropna()
    val_df = clean_features(val_df, remove_id_era=True)
    
    X_val, y_val = split_data(val_df)[:2]
    val_size = y_val.shape[0]
    val_preds = predictions[:val_size]

    logloss = sklearn.metrics.log_loss(y_val, val_preds)
    print(f"Log loss for {output_file}: {logloss:.6f}")

    # Add header to output file
    with open(output_file, 'r') as f:
        data = f.read()
    with open(output_file, 'w') as f:
        f.write("id,probability\n" + data)

    return True

def prepare_input_data(csv_prefix, df):
    """Prepare and merge input data"""
    df.reset_index(drop=True, inplace=True)

    # Print correlations
    correlations = df.corr()
    print_full_df(correlations["target"].sort_values(ascending=False))
    
    # Load and merge prediction files
    pred1 = pd.read_csv(f'{csv_prefix}1.csv')
    pred2 = pd.read_csv(f'{csv_prefix}2.csv') 
    merged = pred1.merge(pred2, on='id')
    df = merged.merge(df, on='id')
    
    df = clean_features(df, remove_id_era=False)
    
    print(f"Final dataset shape: {df.shape}")
    correlations = df.corr()
    print_full_df(correlations["target"].sort_values(ascending=False))
    print("Dataset preparation complete!")
    
    return df

# Load data
tournament_data = pd.read_csv('numerai_tournament_data.csv')
training_data = pd.read_csv('numerai_training_data.csv')

# Prepare datasets
tournament_data = prepare_input_data("predictsnumer", tournament_data)
training_data = prepare_input_data("predictsnumerOld", training_data)

# Get IDs
tournament_ids = tournament_data['id'].values.reshape(-1, 1)
training_ids = training_data['id'].values.reshape(-1, 1)

# Clean and prepare features
tournament_data = tournament_data.dropna()
tournament_data = tournament_data.drop(['era', 'id'], axis=1)
X_train, y_train, _ = split_data(tournament_data)

# Prepare test data
tournament_test = tournament_data.copy()
X_test, _ = split_test_data(tournament_test)

training_data = training_data.drop(['era', 'id'], axis=1) 
X_train_old, y_train_old = split_data(training_data)[:2]

# Train Random Forest model
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=350,
    max_leaf_nodes=7,
    n_jobs=-1,
    oob_score=True
)
rf_model.fit(X_train, y_train)
print(f"Out-of-bag score: {rf_model.oob_score_}")

# Generate predictions
tournament_preds = rf_model.predict_proba(X_test)[:, 1]
training_preds = rf_model.predict_proba(X_train_old)[:, 1]

# Save predictions
save_predictions(tournament_preds, tournament_ids, "predictsnumer.csv", "numerai_tournament_data.csv")
save_predictions(training_preds, training_ids, "predictsnumerOld.csv", "numerai_training_data.csv")
