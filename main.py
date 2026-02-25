from data_prep import train_test_split_data, readData
from utils import trainModels, optimiseRegularization

if __name__ == "__main__":
    x, y, df1, df2 = readData("data/assign1_25.csv")
    x_train, x_test, y_train, y_test = train_test_split_data(df2)

    trainModels(x_train, x_test, y_train, y_test)
