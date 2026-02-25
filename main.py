from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from linear_model import train_test_split, readData

x_norm, x, y, df1, df2 = readData("data/assign1_25.csv")
x_train, x_test, y_train, y_test = train_test_split(df2)

# try both linear and polynomial of different degrees
linear_model = LinearRegression(normalize=True)
p2_model = LinearRegression(normalize=True)
p3_model = LinearRegression(normalize=True)

# create polynomial features
p2_features = PolynomialFeatures(degree=2)
p2_train = p2_features.fit_transform(x_train)
p2_test = p2_features.fit_transform(x_test)

p3_features = PolynomialFeatures(degree=3)
p3_train = p3_features.fit_transform(x_train)
p3_test = p3_features.fit_transform(x_test)

# now do estimation of models
lin_1 = linear_model.fit(x_train, y_train)
p2_1 = p2_model.fit(p2_train, y_train)
p3_1 = p3_model.fit(p3_train, y_train)

# predict values for test sets
lin1_predict = lin_1.predict(x_test)
p2_predict = p2_1.predict(p2_test)
p3_predict = p3_1.predict(p3_test)

print(mean_squared_error(lin1_predict, y_test))
print(mean_squared_error(p2_predict, y_test))
print(mean_squared_error(p3_predict, y_test))

# use k-fold with regularization
# remember: alpha here is lambda in most treatments
from sklearn.linear_model import LassoCV

#Lasso Cross validation
lasso_1 = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], random_state=0).fit(x_train, y_train)

# results
print(lasso_1.score(x_train, y_train))
print(lasso_1.score(x_test, y_test))


# or you can do by hand
lambdas = (.1, .5, 1, 2.5, 5, 7.5, 10, 20, 50, 100, 200)

for i in lambdas:
    lasso_reg = Lasso(alpha = i, max_iter=10000)
    lasso1 = lasso_reg.fit(p3_train, y_train)
    lasso1_predict = lasso1.predict(p3_test)
    print (mean_squared_error(y_test, lasso1_predict)**(.5))