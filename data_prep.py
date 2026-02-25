import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso


def readData(fileName):
    # grab data -- change path for your own file
    df1 = pd.read_csv(fileName)

    # print first few rows -- could also use df.head()
    #print (df1.iloc[:3])
    # and get summary stats on variables
    #print (df1.describe())

    df2 = df1.dropna()
    # and get summary stats on variables to check to see if any variables had missingness
    #print (df2.describe())
    # no missingness!
    # OR use df1.isna().sum().sum()

    # create set of variables to use / exclude Y
    vars = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    x = df2.loc[:, vars].values

    # also create Y while we're at it for use later on in regressions
    y = df2.loc[:, 'y'].values

    # normalize x
    #x_norm = StandardScaler().fit_transform(x)
    return x, y, df1, df2

def train_test_split_data(df2):
    # just an example of running a model with SOME of the columns
    IVs = ['x1','x2','x3','x4','x5','x6','x7','x8','x9']

    # create train / test split using dataframe
    x_train, x_test, y_train, y_test = train_test_split(
        df2.loc[:, IVs], 
        df2.loc[:, 'y'], 
        test_size=0.2, #80% training data, 20% test data
        random_state=13
    )
    return x_train, x_test, y_train, y_test
