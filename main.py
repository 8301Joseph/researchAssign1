from data_prep import train_test_split_data, readData
from utils import trainModels, optimiseRegularization, refit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


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

def printFinaleqn():
    # ----- REFIT DEGREE 2 LASSO ON FULL CLEAN SAMPLE -----

    # Full cleaned sample already available as df2
    X_full = df2.drop(columns=["y"]).values
    y_full = df2["y"].values

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    X_full_p2 = p2_features.fit_transform(X_full_scaled)

    final_model, final_alpha = refit(X_full_p2, y_full)

    feature_names = p2_features.get_feature_names_out(df2.drop(columns=["y"]).columns)
    coefs = final_model.coef_
    intercept = final_model.intercept_

    print("\nFINAL DEGREE 2 LASSO MODEL (Full Sample)")
    print(f"Optimal lambda: {final_alpha:.3f}")

    equation = f"y = {intercept:.6f}"
    for name, coef in zip(feature_names, coefs):
        if abs(coef) > 1e-6:
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.4f}*{name}"

    print("\nFinal Equation:")
    print(equation)





if __name__ == "__main__":
    x, y, df1, df2 = readData("data/assign1_25.csv")
    x_train, x_test, y_train, y_test = train_test_split_data(df2) #split data
    p2_train, p2_test, p3_train, p3_test, p2_features, lin1_predict = trainModels(x_train, x_test, y_train, y_test)

    linearModel(y_test, lin1_predict)
    regularization(p2_train, p2_test, p3_train, p3_test, y_train, y_test)

    printFinaleqn()



#Linear RMSE: 350.094
#P2 min RMSE: 215.621 (lambda = 19.807)
#P3 min RMSE: 280.546 (lambda = 23.271)

''' Best DGP is 
y=f(x)+e 
where 
f(x) is a degree 2 Lasso-selected polynomial model.'''


"y = -192.633145 + 16.3014*x1 + 104.1009*x2 + 92.4932*x3 + "
"513.5162*x6 - 8.4396*x2^2 - 11.8651*x2 x3 - 219.1545*x6^2 "
"- 11.3830*x6 x7 - 16.1697*x7 x9 - 4.6263*x8 x9"