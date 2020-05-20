from bert import bert_tokenization
from tensorflow_hub import KerasLayer


class BertTokenizer:
    def __init__(self):
        self.handle = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
        self.vocab = None

    def get_tokenizer(self):
        bert_layer = KerasLayer(handle=self.handle, trainable=False)
        vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

        return bert_tokenization.FullTokenizer(vocabulary_file, to_lower_case)

    def tokenize(self, sentence):
        tokenizer = self.get_tokenizer()
        tokenized_sentence = tokenizer.tokenize(sentence)

        self.vocab = tokenizer.vocab

        return tokenizer.convert_tokens_to_ids(tokens=tokenized_sentence)