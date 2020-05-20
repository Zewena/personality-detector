from tensorflow.keras.preprocessing import sequence


class SequencePadder:
    def __init__(self, text_attr):
        self.text_attr = text_attr

    def pad_sequences(self, data):
        q = data.copy()

        q[self.text_attr] = self.handle_pad(q[self.text_attr])

        return q

    def handle_pad(self, sentences):
        sequences = sequence.pad_sequences(sentences, maxlen=400, padding='post')

        return sequences.tolist()
