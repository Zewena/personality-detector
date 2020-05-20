from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def split_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)

        return x_train, x_test, y_train, y_test
