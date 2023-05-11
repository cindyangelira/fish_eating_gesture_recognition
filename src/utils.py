import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import math
import pickle5 as pickle
import joblib



class FeaturePreprocessing:
    """Preprocesses the data by adding rolling and lag features"""
    def __init__(self, window_size=5, lag_size=2):
        self.window_size = window_size
        self.lag_size = lag_size

    def preprocess_by_id(self, df):
        id = df['pond_code'].unique()
        X_train_all = []
        X_test_all = []
        y_train_all = []
        y_test_all = []
        for i in id:
            df2 = df[df['pond_code'] == i]
            X_train, X_test, y_train, y_test = self.preprocess(df2)
            X_train_all.append(X_train)
            X_test_all.append(X_test)
            y_train_all.append(y_train)
            y_test_all.append(y_test)
        X_train = pd.concat(X_train_all).fillna(0)
        X_test = pd.concat(X_test_all).fillna(0)
        y_train = pd.concat(y_train_all).fillna(0)
        y_test = pd.concat(y_test_all).fillna(0)
        return X_train, X_test, y_train, y_test

    def preprocess(self, data):
        data = self._add_rolling_features(data)
        data = self._add_lag_features(data)
        data = self._one_hot_encode(data)
        # drop rows with NaN values
        data['magn'] = (data['x']**2 + data['y']**2 + data['z']**2).apply(lambda x: math.sqrt(x))
        data = data.dropna()

        # split data into train and test
        X = data.drop(columns=['label', 'date', 'timestamp','x','y','z'], axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    def _add_rolling_features(self, data):
        data['x_rolling_mean'] = data['x'].rolling(window=self.window_size).mean()
        data['x_rolling_std'] = data['x'].rolling(window=self.window_size).std()
        data['x_rolling_min'] = data['x'].rolling(window=self.window_size).min()
        data['x_rolling_max'] = data['x'].rolling(window=self.window_size).max()
        data['y_rolling_mean'] = data['y'].rolling(window=self.window_size).mean()
        data['y_rolling_std'] = data['y'].rolling(window=self.window_size).std()
        data['y_rolling_min'] = data['y'].rolling(window=self.window_size).min()
        data['y_rolling_max'] = data['y'].rolling(window=self.window_size).max()
        data['z_rolling_mean'] = data['z'].rolling(window=self.window_size).mean()
        data['z_rolling_std'] = data['z'].rolling(window=self.window_size).std()
        data['z_rolling_min'] = data['z'].rolling(window=self.window_size).min()
        data['z_rolling_max'] = data['z'].rolling(window=self.window_size).max()
        return data
    
    def _add_lag_features(self, data):
        for i in range(1, self.lag_size+1):
            data[f'x_lag_{i}'] = data['x'].shift(i)
            data[f'y_lag_{i}'] = data['y'].shift(i)
            data[f'z_lag_{i}'] = data['z'].shift(i)
        return data

    def _one_hot_encode(self, data):
        #data['label'] = data['label'].astype('category')
        data = pd.get_dummies(data, columns=['pond_code'])
        return data


class TimeSeriesClassifier:
    """Classifies time series data using Random Forest, XGBoost and Stacked Classifier"""
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb = XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
        self.sclf = StackingClassifier(
            estimators=[('rf', self.rf), ('xgb', self.xgb)],
            final_estimator=LogisticRegression()
        )

    def fit(self, X_train, y_train, model='rf'):
        if model == 'rf':
            self.rf.fit(X_train, y_train)
        elif model == 'xgb':
            self.xgb.fit(X_train, y_train)
        elif model == 'stacked':
            self.sclf.fit(X_train, y_train)
        else:
            raise ValueError("Invalid model type")

    def predict(self, X_test, model='rf'):
        if model == 'rf':
            y_pred = self.rf.predict(X_test)
        elif model == 'xgb':
            y_pred = self.xgb.predict(X_test)
        elif model == 'stacked':
            y_pred = self.sclf.predict(X_test)
        else:
            raise ValueError("Invalid model type")
        return y_pred
    
    def predict_proba(self, X_test, model='rf'):
        if model == 'rf':
            y_pred = self.rf.predict_proba(X_test)
        elif model == 'xgb':
            y_pred = self.xgb.predict_proba(X_test)
        elif model == 'stacked':
            y_pred = self.sclf.predict_proba(X_test)
        else:
            raise ValueError("Invalid model type")
        return y_pred

    def evaluate(self, X, y, model='rf'):
        if model == 'rf':
            clf = self.rf
        elif model == 'xgb':
            clf = self.xgb
        elif model == 'stacked':
            clf = self.sclf
        else:
            raise ValueError("Invalid model type")
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("ROC AUC Score:", roc_auc_score(y, y_pred))

    def plot_roc_curve(self, X, y, model='rf'):
        if model == 'rf':
            clf = self.rf
        elif model == 'xgb':
            clf = self.xgb
        elif model == 'stacked':
            clf = self.sclf
        else:
            raise ValueError("Invalid model type")
        y_score = clf.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(12,10))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    def save_model(self, model_path, model='rf'):
        if model == 'rf':
            clf = self.rf
        elif model == 'xgb':
            clf = self.xgb
        elif model == 'stacked':
            clf = self.sclf
        else:
            raise ValueError("Invalid model type")

        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

        # Set the loaded model to the appropriate instance variable
        if isinstance(clf, RandomForestClassifier):
            self.rf = clf
        elif isinstance(clf, XGBClassifier):
            self.xgb = clf
        elif isinstance(clf, StackingClassifier):
            self.sclf = clf
        else:
            raise ValueError("Invalid model type")