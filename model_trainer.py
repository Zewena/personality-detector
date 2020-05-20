import logging

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History

from essay_dataset import EssayDataset
from text_model import TextModel
import configuration as config


class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger()
        self.model = None
        self.path_to_save_model = "checkpoint/text-model.ckpt"

    def train(self):
        ds = EssayDataset()
        x_train, y_train, x_test, y_test, vocab = ds.load_data()

        text_model = TextModel(vocabulary_size=len(vocab),
                               embedding_dimensions=config.EMB_DIM,
                               cnn_filters=config.CNN_FILTERS,
                               dnn_units=config.DNN_UNITS,
                               dropout_rate=config.DROPOUT_RATE)

        text_model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        history = History()

        checkpoint = ModelCheckpoint(
                filepath=self.path_to_save_model,
                verbose=1,
                save_best_only=True)

        callbacks_list = [history, checkpoint]

        text_model.fit(x_train, y_train,
                       epochs=config.NB_EPOCHS,
                       verbose=2,
                       validation_split=0.1,
                       callbacks=callbacks_list)



