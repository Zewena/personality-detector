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

        self.train_data = None
        self.test_data = None

    def load_data(self):
        loader = CSVDataLoader(file_path=self.file_path, text_attr=self.text_attr, trait_attrs=self.trait_attrs)
        token_converter = TokenConverter(tokenizer=BertTokenizer(), text_attr=self.text_attr)

        vectorized_data, vocab = token_converter.convert_tokens_to_ids(data=loader.load_data())

        sequence_padder = SequencePadder(text_attr=self.text_attr)

        # padded_df = sequence_padder.pad_sequences(data=vectorized_data)

        splitter = DataSplitter(text_attr=self.text_attr, trait_attrs=self.trait_attrs)

        x_train, x_test, y_train, y_test = splitter.split_data(data=vectorized_data)

        return x_train, y_train, x_test, y_test, vocab
