# 
# ArcMLP - Preprocessing Component
#
# Data cleaning and preprocessing methods to get data ready for
# training ML model.
#

# Importing libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# === Preprocessing and Data Cleaning Methods ===
def read_file(source):
    """
    Read csv file and convert to dataframe

    Parameters
    - source: (string) source where file with dataset is stored

    Return
    - df: (pandas dataframe) dataframe
    """
    # Getting dataset (proof concept - 'paysim-transactions.csv')
    df = pd.read_csv(source)
    
    return df

def inspect_data(df, n=5):
    """
    Get first 'n' rows from dataframe, types of each feature,
    and sum of na values for each feature

    Parameters
    - df: (pandas dataframe) dataframe
    - n (default n=5): (int) num of rows to inspect 

    Return
    - df_sample, df_types, df_na: (tuple)
        df_sample: (pandas dataframe) dataframe with first n rows
        df_types: (pandas Series) series with features and respective types
        df_na: (pandas Series) series with number of na values in each feature
    """
    # Getting first n rows of dataframe
    df_sample = df.head(n)
    # Getting type of each feature
    df_types = df.dtypes
    # Getting na values
    df_na = df.isna().sum()

    return df_sample, df_types, df_na

def remove_variables(df, variables):
    """
    Drop given variables from the dataframe and 
    returns the new dataframe

    Parameters
    - df: (pandas dataframe) dataframe
    - variables: (list of strings) list of variables to remove

    Return
    - df: (pandas dataframe) dataframe
    """
    if len(variables) > 0:
        df = df.drop(variables, axis = 1)

    return df

def filter_data(df, condition):
    """
    Filter data by keeping only the values that satisfy
    the conditions given

    Parameters
    - df: (pandas dataframe) dataframe
    - conditions: (string) string that contains conditions to 
                  be used to filter dataframe. 
                  Ex: if the user only wants to keep entries whose col1
                  value is either 1 or -1, the argument should be 
                  'col1 == 1 | col1 == -1'

    Return
    - df: (pandas dataframe) dataframe
    """
    copy_df = df.copy()
    copy_df = df.loc[df.eval(condition)]

    return copy_df

def split_features_labels(df, label_name):
    """
    Removes label column fromn dataframe

    Parameters
    - df: (pandas dataframe) dataframe
    - label_name: (string) name of variable that contains
                  column with labels

    Return
    - X, Y: (tuple)
        X: (pandas dataframe) dataframe without label column
        Y: (pandas Series) series with labels
    """
    X = df.drop(label_name, 1)
    Y = df[label_name]

    return X, Y

def add_features(df, new_features):
    """
    Add features to dataframe by perfoming any operation
    with the already existent variables

    Parameters
    - df: (pandas dataframe) dataframe
    - new_features: (dictionary) dictionary that contains
                    as keys a string that represents the name of the 
                    new variable, and as values a string with an operation
                    to be evaluated whose result will be the entries
                    of the new feature.
                    Ex: if we want the new feature to be the sum of 
                    two other features ('col1' and 'col2'), the dictionary
                    argument will be {'col3': 'df.col1 + df.col2'}

    Return
    - copy_df: (pandas dataframe) dataframe with new feature(s)
    """
    copy_df = df.copy()
    for key, value in new_features.items():
        copy_df[key] = eval(value)

    return copy_df

def one_hot_encoding(X):
    """
    Perfom One Hot Encoding for categorical variables

    Parameters
    - X: (pandas dataframe) dataframe with features
    
    Return 
    - res: (pandas dataframe) dataframe with one hot enconding
    """
    X_cat = pd.get_dummies(X.select_dtypes(include=['object']))
    X_num = X.select_dtypes(exclude=['object'])
    res = pd.concat([X_num, X_cat], axis=1, sort=False)
     
    return res

def impute_values(X_train, X_test, col_names, missing_val, impute_strategy, impute_val=None):
    """
    Impute missing values with either a constant value or 
    statistics using the scikit learn SimpleImputer

    Parameters
    - X_train: (pandas dataframe) dataframe with 1 - test_percentage of entries
    - X_test: (pandas dataframe) dataframe with test_percentage of entries
    - missing_val: (number, string, np.nan (default) or None) 
                          placeholder for the missing values
    - impute_strategy: ()
    
    Return 
    - res: (pandas dataframe) dataframe with one hot enconding
    """
    X_train_res = X_train.copy()
    X_test_res = X_test.copy()

    for col in col_names:
        imp = SimpleImputer(missing_values=missing_val, strategy=impute_strategy, fill_value=impute_val)
        imp.fit(X_train[[col]])
        X_train_res[col] = imp.transform(X_train[[col]])
        X_test_res[col] = imp.transform(X_test[[col]])

    return X_train_res, X_test_res

def split_train_test(X, Y, test_percentage=0.3):
    """
    Split X and Y into train and test sets

    Parameters
    - X: (pandas dataframe) dataframe with features
    - Y: (pandas Series) series with labels
    - test_percentage = (int) percentage of entries to be 
                        used for the test set
    
    Return 
    - X_train, X_test, Y_train, Y_test: (tuple)
        X_train: (pandas dataframe) dataframe with 1 - test_percentage of entries
        X_test: (pandas dataframe) dataframe with test_percentage of entries
        Y_train: (pandas Series) series with 1 - test_percentage of entries
        Y_test: (pandas Series) series with test_percentage of entries
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_percentage)

    return X_train, X_test, Y_train, Y_test
