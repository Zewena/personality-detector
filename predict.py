from keras.models import load_model
from attention import AttLayer
from main import Preprocessing
import configuration as config
from keras.preprocessing import sequence


if __name__ == "__main__":
    preprocessed = Preprocessing()
    t = preprocessed.load_data(config.dims)

    input_text = "I hate you!"

    result = t.texts_to_sequences([input_text])

    model = load_model('checkpoint/best_weights.hdf5', custom_objects={'AttLayer': AttLayer})

    prediction = model.predict(sequence.pad_sequences(result, maxlen=config.MAX_SEQ_LENGTH))

    print(config.dims)
    print("=" * 60)
    print(prediction)
