from data_prep import train_test_split_data, readData
from utils import trainModels, optimiseRegularization, refit
from sklearn.metrics import mean_squared_error

def regularization(p2_train, p2_test, p3_train, p3_test, y_train, y_test):

    print("\nP2 (degree 2) LassoCV")
    p2_lasso, p2_alpha = optimiseRegularization(p2_train, y_train)
    p2_pred = p2_lasso.predict(p2_test)
    p2_rmse = mean_squared_error(y_test, p2_pred) ** 0.5
    print(f"Optimal lambda (alpha): {p2_alpha:.3f}")
    print(f"Test RMSE: {p2_rmse:.3f}")

    print("\nP3 (degree 3) LassoCV")
    p3_lasso, p3_alpha = optimiseRegularization(p3_train, y_train)
    p3_pred = p3_lasso.predict(p3_test)
    p3_rmse = mean_squared_error(y_test, p3_pred) ** 0.5
    print(f"Optimal lambda (alpha): {p3_alpha:.3f}")
    print(f"Test RMSE: {p3_rmse:.3f}")

def linearModel(y_test, lin1_predict):
    print("\nLinear Model")
    print(f"Test RMSE: {mean_squared_error(y_test, lin1_predict) ** 0.5:.3f}")


if __name__ == "__main__":
    x, y, df1, df2 = readData("data/assign1_25.csv")
    x_train, x_test, y_train, y_test = train_test_split_data(df2) #split data
    p2_train, p2_test, p3_train, p3_test, p2_features, lin1_predict = trainModels(x_train, x_test, y_train, y_test)

    linearModel(y_test, lin1_predict)
    regularization(p2_train, p2_test, p3_train, p3_test, y_train, y_test)

    refit(x, y)


#Linear RMSE: 350.094
#P2 min RMSE: 215.621 (lambda = 19.807)
#P3 min RMSE: 280.546 (lambda = 23.271)

''' Best DGP is 
y=f(x)+e 
where 
f(x) is a degree 2 Lasso-selected polynomial model.'''