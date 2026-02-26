from data_prep import train_test_split_data, readData
from utils import trainModels, optimiseRegularization, optimiseRegularizationV2
from utils import optimiseRegularization, trainModels
from data_prep import readData, train_test_split_data


def regularization():
    #p2 optimisation
    print("")
    print("P2 RMSE trials for lambdas")
    optimiseRegularization(p2_train, p2_test, y_test, y_train)

    #p3 optimisation
    print("")
    print("P3 RMSE trials for lambdas")
    optimiseRegularization(p3_train, p3_test, y_test, y_train)


if __name__ == "__main__":
    x, y, df1, df2 = readData("data/assign1_25.csv")
    x_train, x_test, y_train, y_test = train_test_split_data(df2)

    p2_train, p2_test, p3_train, p3_test, p2_features = trainModels(x_train, x_test, y_train, y_test)

    #regularization()


#Linear RMSE: 350.09
#P2 min RMSE: 214.28 (lambda = 7.5)
#P3 min RMSE: 274.81 (lambda = 50)

#continue exploring P2 with lambda between 6 and 10

    optimalLambda, minRmse, best_model = optimiseRegularizationV2(p2_train, p2_test, y_test, y_train)
    print(f"Optimal lambda value: {optimalLambda:.2f} which provides an RMSE of {minRmse:.2f}")
    print("\nModel equation:")

    feature_names = p2_features.get_feature_names_out()
    coefs = best_model.coef_
    intercept = best_model.intercept_

    terms = [
        f"{coef:.2f}*{feature_names[i]}"
        for i, coef in enumerate(coefs)
        if abs(coef) > 1e-6
    ]

    equation = " + ".join(terms)
    print(f"f(x) = {intercept:.2f} + {equation}")

#best model

# f(x) = -165.77 + 360.94*x0 + 57.70*x1 + 77.24*x2 + 547.16*x5 
# + -11.67*x7 + 14.66*x8 + 25.40*x0^2 + -14.82*x1^2 + -35.58*x1 x2 
# + -29.24*x1 x3 + 4.97*x1 x7 + -14.25*x2^2 + 6.64*x2 x3 + -40.52*x2 x5 
# + 5.28*x2 x6 + -15.19*x2 x7 + -1.97*x3^2 + 13.10*x3 x5 + 6.06*x3 x6 
# + -3.23*x3 x7 + -4.89*x3 x8 + -1.04*x4 x5 + 2.20*x4 x8 + -242.75*x5^2 
# + -12.67*x5 x6 + 17.52*x5 x8 + -29.51*x6 x8 + -31.12*x7 x8 + -3.68*x8^2