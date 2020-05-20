from csv_data_loader import CSVDataLoader
from data_splitter import DataSplitter
from bert_tokenizer import BertTokenizer
from token_converter import TokenConverter
from sequence_padder import SequencePadder


class EssayDataset:
    def __init__(self):
        self.file_path = 'data/essays2007.csv'
        self.text_attr = 'text'
        self.trait_attrs = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']

    def load_data(self):
        loader = CSVDataLoader(file_path=self.file_path, text_attr=self.text_attr, trait_attrs=self.trait_attrs)

        bert_tokenizer = BertTokenizer()

        token_converter = TokenConverter(tokenizer=bert_tokenizer, text_attr=self.text_attr)

        vectorized_data = token_converter.convert_tokens_to_ids(data=loader.load_data())

        sequence_padder = SequencePadder(text_attr=self.text_attr, trait_attrs=self.trait_attrs)

        x, y = sequence_padder.pad_sequences(data=vectorized_data)

        x_train, x_test, y_train, y_test = DataSplitter().split_data(x=x, y=y)

        return x_train, y_train, x_test, y_test, bert_tokenizer.vocab
