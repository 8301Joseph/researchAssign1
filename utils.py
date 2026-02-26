from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


def trainModels(x_train, x_test, y_train, y_test):
    #normalise
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)



    # try both linear and polynomial of different degrees
    linear_model = LinearRegression()
    p2_model = LinearRegression()
    p3_model = LinearRegression()

    # create polynomial features
    p2_features = PolynomialFeatures(degree=2)
    p2_train = p2_features.fit_transform(x_train)
    p2_test = p2_features.transform(x_test)

    p3_features = PolynomialFeatures(degree=3)
    p3_train = p3_features.fit_transform(x_train)
    p3_test = p3_features.transform(x_test)

    # now do estimation of models
    lin_1 = linear_model.fit(x_train, y_train)
    p2_1 = p2_model.fit(p2_train, y_train)
    p3_1 = p3_model.fit(p3_train, y_train)

    # predict values for test sets
    lin1_predict = lin_1.predict(x_test)
    p2_predict = p2_1.predict(p2_test)
    p3_predict = p3_1.predict(p3_test)

    print(f"Linear MSE: {mean_squared_error(lin1_predict, y_test)}")
    print (f"Linear RMSE: {mean_squared_error(lin1_predict, y_test)**(.5)}")
    print(f"P2 MSE: {mean_squared_error(p2_predict, y_test)}")
    print(f"P3 MSE: {mean_squared_error(p3_predict, y_test)}")

    return p2_train, p2_test, p3_train, p3_test


# try different lambdas with regularization
# remember: alpha here is lambda in most treatments

def optimiseRegularization(train, test, y_test, y_train):
    lambdas = (.1, .5, 1, 2.5, 5, 7.5, 10, 20, 50, 100, 200)
    for i in lambdas:
        lasso_reg = Lasso(alpha = i, max_iter=200000)
        lasso1 = lasso_reg.fit(train, y_train)
        lasso1_predict = lasso1.predict(test)
        print (f"{i}: {mean_squared_error(y_test, lasso1_predict)**(.5)}")