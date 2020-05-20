from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Dropout
from tensorflow.keras import Model
from tensorflow import concat


class TextModel(Model):
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=5,
                 dropout_rate=0.1,
                 name="text_model"
                 ):
        super(TextModel, self).__init__(name=name)

        self.embedding = Embedding(vocabulary_size, embedding_dimensions)

        self.cnn_layer1 = Conv1D(filters=cnn_filters,
                                 kernel_size=2,
                                 padding="valid",
                                 activation="relu")

        self.cnn_layer2 = Conv1D(filters=cnn_filters,
                                 kernel_size=3,
                                 padding="valid",
                                 activation="relu")

        self.cnn_layer3 = Conv1D(filters=cnn_filters,
                                 kernel_size=4,
                                 padding="valid",
                                 activation="relu")

        self.pool = GlobalMaxPool1D()

        self.dense_1 = Dense(units=dnn_units, activation="relu")

        self.dropout = Dropout(rate=dropout_rate)

        self.last_dense = Dense(units=model_output_classes, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output
