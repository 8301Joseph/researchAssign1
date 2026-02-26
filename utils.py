from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV


def trainModels(x_train, x_test, y_train, y_test):
    #normalise
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)



    # try both linear and polynomial of different degrees
    linear_model = LinearRegression()

    # create polynomial features
    p2_features = PolynomialFeatures(degree=2)
    p2_train = p2_features.fit_transform(x_train)
    p2_test = p2_features.transform(x_test)

    p3_features = PolynomialFeatures(degree=3)
    p3_train = p3_features.fit_transform(x_train)
    p3_test = p3_features.transform(x_test)

    # now do estimation of models
    lin_1 = linear_model.fit(x_train, y_train)
    '''p2_1 = p2_model.fit(p2_train, y_train)
    p3_1 = p3_model.fit(p3_train, y_train)

    # predict values for test sets'''
    lin1_predict = lin_1.predict(x_test)
    '''p2_predict = p2_1.predict(p2_test)
    p3_predict = p3_1.predict(p3_test)

    print (f"Linear RMSE: {mean_squared_error(lin1_predict, y_test)**(.5)}")
    print(f"P2 RMSE: {mean_squared_error(p2_predict, y_test)**(.5)}")
    print(f"P3 RMSE: {mean_squared_error(p3_predict, y_test)**(.5)}")'''

    return p2_train, p2_test, p3_train, p3_test, p2_features, lin1_predict


# try different lambdas with regularization
# remember: alpha here is lambda in most treatments

def optimiseRegularization(x_train, y_train):
    """
    Select lambda (alpha) using cross-validation on TRAIN only.
    Returns: fitted LassoCV model, optimal alpha
    """
    lasso = LassoCV(cv=5, max_iter=200000)
    lasso.fit(x_train, y_train)
    return lasso, lasso.alpha_


def refit(x, y):
    #refit degree 2 f(x) on entire training set
    lasso = LassoCV(cv=5, max_iter=200000)
    lasso.fit(x, y)
    return lasso, lasso.alpha_






