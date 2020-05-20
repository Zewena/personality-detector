from tensorflow.keras.preprocessing import sequence


class SequencePadder:
    def __init__(self, text_attr, trait_attrs):
        self.text_attr = text_attr
        self.trait_attrs = trait_attrs

    def pad_sequences(self, data):
        x = sequence.pad_sequences(data[self.text_attr], padding='post')
        y = data[self.trait_attrs].values

        return x, y
