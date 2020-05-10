from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
import configuration as config


class DeepModel:
    @staticmethod
    def build(params):
        model = Sequential()

        model.add(Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH))

        model.add(Dense(512, input_shape=(config.MAX_SEQ_LENGTH, config.EMBEDDING_DIM), activation='relu'))

        model.add(Dropout(params.dropout_rate))

        model.add(Flatten())

        model.add(Dense(len(config.dims), activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
