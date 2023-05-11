import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import fftpack
import math
from sklearn.datasets import make_classification
from src.utils import FeaturePreprocessing, TimeSeriesClassifier

# Set page config
st.set_page_config(
    page_title="EATING POSTURE RECOGNITION OF FISH USING ACCELEROMETER SENSOR DATA WITH XGBOOST",
    page_icon="📊",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
 # data
st.title('EATING POSTURE RECOGNITION OF FISH USING ACCELEROMETER SENSOR DATA WITH XGBOOST')
st.write('This is a ML project to classify whether the fish is currently in eating position or not.')
st.write('The data is collected from accelerometer sensor attached to the fish.')
st.write('The data is collected from 3 different ponds, each pond has different fish species.')

#load data

st.header('Data')
st.subheader('Signal By Pond')

# read transaction data
def read_data(path):
    files = [file for file in os.listdir(path) if file.endswith('.xlsx')]
    dfs = []
    for file in files:
        file_path = os.path.join(path, file)
        pond_name = os.path.splitext(file)[0]
        df = pd.read_excel(file_path)
        df['pond_code'] = pond_name
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data['date'] = (data['timestamp']/1000).apply(datetime.fromtimestamp)
    data.sort_values(by=['pond_code', 'date'], inplace=True)
    return data

df = read_data('data/')
df.sort_values(by='timestamp', inplace=True)
c1 = st.empty()
c1.dataframe(df)
st.write(f'The data collected at {df.date.min()} to {df.date.max()}.')

# create sidebar to add filter on pond_code
st.sidebar.header('Filter Pond Code')
id = df.pond_code.unique().tolist()
pond_code = st.sidebar.selectbox('Pond Code', id[0:] + [None])
if pond_code is None:
    c1.dataframe(df)
else:
    df1 = df[df.pond_code == pond_code]
    c1.dataframe(df1)

st.subheader('EDA')
ponds = df['pond_code'].unique()
for pond in ponds:
    pond_df = df[df['pond_code'] == pond]
    labels = pond_df['label'].unique()
    fig = plt.figure(figsize=(20, 5))
    for i, label in enumerate(labels):
        label_df = pond_df[pond_df['label'] == label]
        x = label_df['x']
        y = label_df['y']
        z = label_df['z']
        time = label_df['date']
        ax = fig.add_subplot(1, len(labels), i+1)
        ax.plot(time, x, label='x')
        ax.plot(time, y, label='y')
        ax.plot(time, z, label='z')
        if label == 0:
            ax.set_title("not eating")
        elif label == 1:
            ax.set_title("eating")
        ax.legend()
    fig.suptitle(f'Signal Value of Pond {pond}')
    # add suptitle
    st.pyplot(fig)

st.subheader('Eating Duration by Pond')
# eating duration by pond
groups = df.groupby(['pond_code', 'label']).agg({'date': lambda x: x.max() - x.min(), 
                                                'timestamp': lambda x: len(x)}).reset_index().rename(columns={'date': 'duration', 'timestamp': 'count'})
groups['duration'] = groups['duration'].apply(lambda x: x.seconds)
fig = plt.figure(figsize=(20, 5))
# bar plot to fig
sns.barplot(x='pond_code', y='duration', hue='label', data=groups)
plt.title('Duration of Eating and Not Eating by Pond')
st.pyplot()

st.header("XGBOOST MODEL")
prep = FeaturePreprocessing(lag_size=20)
X_train, X_test, y_train, y_test = prep.preprocess_by_id(df)
model = TimeSeriesClassifier()
model.fit(X_train, y_train, model='xgb')
model.evaluate(X_test, y_test)
