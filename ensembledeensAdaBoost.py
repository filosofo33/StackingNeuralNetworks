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
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import csv
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier

def clean_dataframe(df, remove_id_era=False):
    """Clean and preprocess the dataframe by removing redundant features and engineering new ones"""
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
    """Normalize features to 0-1 range using MinMaxScaler"""
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return minmax_scaler.fit_transform(X)

def prepare_matrix(df, is_training=True):
    """Convert dataframe to numpy matrices for model input"""
    X = df.drop('target', axis=1).values
    X = X.astype(np.float32)
    
    if is_training:
        y = df['target'].values
        y = y.reshape(-1, 1).astype(np.float32)
        return X, y
    return X, None

def split_data(df, scaler=None):
    """Split data into features and labels, optionally scaling"""
    X, y = prepare_matrix(df, is_training=True)
    return X, y, scaler

def prepare_test_data(df, scaler=None):
    """Prepare test data, optionally applying existing scaler"""
    X, _ = prepare_matrix(df, is_training=False)
    return X, scaler

def print_full_dataframe(df):
    """Print full dataframe without truncation"""
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def save_predictions(predictions, output_file, input_file, ids):
    """Save model predictions to CSV file"""
    print(f"Predictions shape: {predictions.shape}")
    np.set_printoptions(suppress=True)
    
    # Combine IDs with predictions
    results = np.c_[ids, predictions]
    np.savetxt(output_file, results, fmt="%s,%10.8f", delimiter=",")

    # Calculate log loss on validation data if available
    df_validation = pd.read_csv(input_file)
    df_validation = df_validation.dropna()
    df_validation = clean_dataframe(df_validation, remove_id_era=True)

    X_val, y_val = split_data(df_validation)[:2]
    n_rows = y_val.shape[0]
    
    pred_probas = predictions[:n_rows]
    print(f"Log loss for {output_file}: {sklearn.metrics.log_loss(y_val, pred_probas)}")

    # Add header to output file
    with open(output_file, 'r') as f:
        content = f.read()
    with open(output_file, 'w') as f:
        f.write("id,probability\n" + content)

def prepare_input_data(predictions_prefix, df):
    """Prepare and merge input data with predictions"""
    df.reset_index(drop=True, inplace=True)

    # Print correlation with target
    correlation = df.corr()
    print_full_dataframe(correlation["target"].sort_values(ascending=False))
    
    # Load and merge prediction files
    pred1 = pd.read_csv(f'{predictions_prefix}1.csv')
    pred2 = pd.read_csv(f'{predictions_prefix}2.csv')
    merged = pred1.merge(pred2, on='id')
    df = merged.merge(df, on='id')

    df = clean_dataframe(df, remove_id_era=False)
    print(f"Final dataset shape: {df.shape}")
    
    correlation = df.corr()
    print_full_dataframe(correlation["target"].sort_values(ascending=False))
    print("Dataset preparation complete!")
    
    return df

# Load data
tournament_data = pd.read_csv('numerai_tournament_data.csv')
training_data = pd.read_csv('numerai_training_data.csv')

# Prepare datasets
tournament_data = prepare_input_data("predictsnumer", tournament_data)
training_data = prepare_input_data("predictsnumerOld", training_data)

# Split validation set
validation_size = 0.05
tournament_data = tournament_data.dropna()

# Stratified sampling across eras
sample_fn = lambda x: x.loc[np.random.choice(x.index, int(x.shape[0]*validation_size), replace=False)]
validation_data = tournament_data.groupby('era', as_index=False).apply(sample_fn)
train_data = tournament_data[~tournament_data.id.isin(validation_data.id)]

# Remove era and id columns
train_data = train_data.drop(['era', 'id'], axis=1)
validation_data = validation_data.drop(['era', 'id'], axis=1)

# Prepare train/validation sets
X_train, y_train, _ = split_data(train_data)
X_val, y_val, _ = split_data(validation_data)

# Get IDs for predictions
tournament_ids = tournament_data['id'].values.reshape(-1, 1)
training_ids = training_data['id'].values.reshape(-1, 1)

# Prepare test sets
tournament_data = tournament_data.drop(['era', 'id'], axis=1)
training_data = training_data.drop(['era', 'id'], axis=1)

X_tournament, _ = prepare_matrix(tournament_data, is_training=False)
X_training, y_training = prepare_matrix(training_data)

# Train AdaBoost model
print("Training AdaBoost model...")
max_depth = 2
base_estimator = DecisionTreeClassifier(max_depth=max_depth)

# Find optimal number of estimators
ada_boost = AdaBoostClassifier(
    base_estimator,
    n_estimators=100,
    algorithm="SAMME.R",
    learning_rate=0.1
)
ada_boost.fit(X_train, y_train)

# Get staged predictions to find optimal number of estimators
staged_predictions = ada_boost.staged_predict_proba(X_training)
errors = [mean_squared_error(y_training, pred[:,1]) for pred in staged_predictions]

optimal_estimators = np.argmin(errors)
print(f"Optimal number of estimators: {optimal_estimators}")

# Retrain with optimal estimators
final_model = AdaBoostClassifier(
    base_estimator,
    n_estimators=optimal_estimators,
    algorithm="SAMME.R",
    learning_rate=0.1
)
final_model.fit(X_train, y_train)

# Generate and save predictions
tournament_predictions = final_model.predict_proba(X_tournament)[:,1]
save_predictions(tournament_predictions, "predictsnumer.csv", "numerai_tournament_data.csv", tournament_ids)

training_predictions = final_model.predict_proba(X_training)[:,1]
save_predictions(training_predictions, "predictsnumerOld.csv", "numerai_training_data.csv", training_ids)