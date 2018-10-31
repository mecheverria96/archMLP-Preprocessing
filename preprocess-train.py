# 
# ArcMLP - Preprocessing Component
#
# Data cleaning and preprocessing methods to get data ready for
# training ML model.
#

# Importing libraries
import pandas as pd
import numpy as np
# scikit learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# === Preprocessing and Data Cleaning Methods ===

# Getting dataset (proof concept - 'paysim-transactions.csv')
df = pd.read_csv('paysim-transactions.csv')

# Cleaning data

# Only keeping entries that are of type Transfer and Cash_out given
# that are the only ones that contain fraud transactions
df = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
df = df.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

# Taking care of cases were the old balance and new balance are 0 but
# the amount is not
df.loc[(df.oldbalanceDest == 0) & (df.newbalanceDest == 0) & (df.amount != 0), \
    ['oldbalanceDest', 'newbalanceDest']] = -1
df.loc[(df.oldbalanceOrg == 0) & (df.newbalanceOrig == 0) & (df.amount != 0), \
    ['oldbalanceOrg', 'newbalanceOrig']] = np.nan
# Getting rid of NA values
df = df.dropna()
# Separate features and labels
X = df.drop('isFraud', 1)
y = df['isFraud']
# Binary encode type variable
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
# convert dtype('O') to dtype(int)
X.type = X.type.astype(int) 
# Adding two features 
X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest
# Splitting train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
# Training the model 
svm_classifier = SVC(kernel='linear') 
svm_classifier.fit(X_train, y_train)  
y_pred = svm_classifier.predict(X_test)  

