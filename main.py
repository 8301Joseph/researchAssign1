from data_prep import train_test_split_data, readData
from utils import trainModels, optimiseRegularization
from utils import optimiseRegularization, trainModels
from data_prep import readData, train_test_split_data



if __name__ == "__main__":
    x, y, df1, df2 = readData("data/assign1_25.csv")
    x_train, x_test, y_train, y_test = train_test_split_data(df2)

    p2_train, p2_test, p3_train, p3_test = trainModels(x_train, x_test, y_train, y_test)

    #p2 optimisation
    print("")
    print("P2 RMSE trials for lambdas")
    optimiseRegularization(p2_train, p2_test, y_test, y_train)

    #p3 optimisation
    print("")
    print("P3 RMSE trials for lambdas")
    optimiseRegularization(p3_train, p3_test, y_test, y_train)
