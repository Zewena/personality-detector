from text_model import TextModel
import configuration as config

if __name__ == "__main__":
    model = TextModel(vocabulary_size=30522,
                      embedding_dimensions=config.EMB_DIM,
                      cnn_filters=config.CNN_FILTERS,
                      dnn_units=config.DNN_UNITS,
                      dropout_rate=config.DROPOUT_RATE)

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    weights = model.load_weights(filepath='checkpoint/text.model.hdf5')
