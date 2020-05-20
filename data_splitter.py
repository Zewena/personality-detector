from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, text_attr, trait_attrs, test_size=0.2):
        self.text_attr = text_attr
        self.trait_attrs = trait_attrs
        self.test_size = test_size

    def split_data(self, data):
        x = sequence.pad_sequences(data[self.text_attr], padding='post')
        y = data[self.trait_attrs].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        return x_train, x_test, y_train, y_test
