class TokenConverter:
    def __init__(self, tokenizer, text_attr):
        self.tokenizer = tokenizer
        self.text_attr = text_attr

    def convert_tokens_to_ids(self, data):
        preprocessed_data = data.copy()

        preprocessed_data[self.text_attr] = preprocessed_data[self.text_attr].apply(self.convert)

        return preprocessed_data

    def convert(self, sentence):
        return self.tokenizer.tokenize(sentence)
