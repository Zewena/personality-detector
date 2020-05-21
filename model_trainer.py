import logging

from tensorflow.keras.callbacks import ModelCheckpoint, History

from essay_dataset import EssayDataset
from text_model import TextModel
import configuration as config


class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger()
        self.model = None
        self.path_to_save_model = "checkpoint/text.model.hdf5"

    def get_model(self, vocabulary_size):
        model = TextModel(vocabulary_size=vocabulary_size,
                          embedding_dimensions=config.EMB_DIM,
                          cnn_filters=config.CNN_FILTERS,
                          dnn_units=config.DNN_UNITS,
                          dropout_rate=config.DROPOUT_RATE)

        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model

    def train(self):
        ds = EssayDataset()
        x_train, y_train, x_test, y_test, vocab = ds.load_data()

        model = self.get_model(vocabulary_size=len(vocab))

        history = History()

        checkpoint = ModelCheckpoint(
            filepath=self.path_to_save_model,
            verbose=1,
            save_best_only=True,
            save_weights_only=True)

        callbacks_list = [history, checkpoint]

        model.fit(x_train, y_train,
                  epochs=config.NB_EPOCHS,
                  verbose=2,
                  validation_split=0.1,
                  callbacks=callbacks_list)
