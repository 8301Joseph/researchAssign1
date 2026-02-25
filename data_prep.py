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
    print (df1.iloc[:3])
    # and get summary stats on variables
    print (df1.describe())

    df2 = df1.dropna()
    # and get summary stats on variables to check to see if any variables had missingness
    print (df2.describe())
    # no missingness!
    # OR use df1.isna().sum().sum()

    # create set of variables to use / exclude Y
    vars = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    x = df2.loc[:, vars].values

    # also create Y while we're at it for use later on in regressions
    y = df2.loc[:, 'y'].values

    # normalize x
    x_norm = StandardScaler().fit_transform(x)
    return x_norm, x, y, df1, df2

x_norm, x, y, df1, df2 = readData("data/assign1_25.csv")

# try pca -- imagine theory suggests you look for a 1d latent representation for x2,x3,x7

def ignore():
    #ignore this part
    dim = 1
    vars2 = ['x2', 'x3', 'x7']
    temp = df2.loc[:, vars2].values
    pca2 = PCA(n_components=1)

    # create 1 dimensional representation
    latent_vars = pca2.fit_transform(temp)
    # check to see if this is right

    print ("Variance explained by each latent variable in PCA: ", pca2.explained_variance_ratio_)
    print ("\n")

    # create new dataframe with the latent variables from pca1
    df2['pca1'] = latent_vars[:,0]
    # add the latent variables to x_norm
    x_norm = np.append(x_norm,latent_vars,1)


# just an example of running a model with SOME of the columns
IVs = ['x1', 'x2', 'x5', 'pca1']

def train_test_split(df2):

    # create train / test split using dataframe
    x_train, x_test, y_train, y_test = train_test_split(
        df2.loc[:, IVs], 
        df2.loc[:, 'y'], 
        test_size=0.2, #80% training data, 20% test data
        random_state=13
    )
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(df2)


# make sure results make sense
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)