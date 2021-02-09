#!/usr/bin/env python
# coding: utf-8

# In[16]:


import argparse
import os
from collections import Counter
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
#import umap
import phik #https://phik.readthedocs.io/en/latest/ - Correlation with mixed categorical and continuous data
sns.set(style="ticks")


np.random.seed(42)
run = Run.get_context()

parser = argparse.ArgumentParser("prep")
parser.add_argument("--data", type=str, help="data")
parser.add_argument("--X_train", type=str, help="X_train")
parser.add_argument("--X_test", type=str, help="X_test")
parser.add_argument("--y_train", type=str, help="y_train")
parser.add_argument("--y_test", type=str, help="y_test")
parser.add_argument("--scaler", type=str, help="scaler")

args = parser.parse_args()

data = run.input_datasets["raw_data"].to_pandas_dataframe()


#Transform categorical (ordinal) values to numerical
encoder = OrdinalEncoder()
va = np.array(data['Vehicle_Age'], ndmin=2).T
data['Vehicle_Age'] = encoder.fit_transform(va).astype(int)


for col in ['Driving_License', 'Gender', 'Vehicle_Damage', 'Vehicle_Age']:
    data[col] = data[col].astype('category')

    
#Transform categorical values to one-hot values
categories = ['Driving_License', 'Gender', 'Vehicle_Damage']
df_onehot = pd.get_dummies(data, columns=categories, drop_first=True)# 


# ## Split into X (features) and y (target)

y = df_onehot['Response']
X = df_onehot.drop(['Response'], axis=1)


# ## Over-/undersample dataset to balance target (y)

smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)


# ## Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state=42)


# ## Scale dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Save train, test and scaler
os.makedirs(args.X_train, exist_ok=True)
os.makedirs(args.X_test, exist_ok=True)
os.makedirs(args.y_train, exist_ok=True)
os.makedirs(args.y_test, exist_ok=True)

np.savetxt(f"{args.X_train}/X_train.csv", X_train, fmt="%f")
np.savetxt(f"{args.X_test}/X_test.csv", X_test, fmt="%f")
np.savetxt(f"{args.y_train}/y_train.csv", y_train, fmt="%f")
np.savetxt(f"{args.y_test}/y_test.csv", y_test, fmt="%f")

if not os.path.isdir(args.scaler):
    os.mkdir(args.scaler)
joblib.dump(scaler, f"{args.scaler}/scaler.pkl")


service_test = X.iloc[1,:]
output_dir = './output/'

os.makedirs(output_dir, exist_ok=True)
joblib.dump(scaler, f'{output_dir}scaler.pkl') 
np.savetxt(f"{output_dir}service_test.csv", service_test, fmt="%f")

### ## Create visualizations
#pairplt = sns.pairplot(data, hue='Response')
#run.log_image("pairplot", plot=pairplt)
##
#interval_cols = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
#phik_overview = X.phik_matrix(interval_cols=interval_cols)
###phik_overview
##
#corr_map = sns.heatmap(phik_overview, annot=True)
#corr_map.figure.set_size_inches(14,10)
#run.log_image("Correlations", plot=corr_map)
###plt.show()
