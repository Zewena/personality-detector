import pandas as pd


class CSVDataLoader:
    def __init__(self, file_path, text_attr, trait_attrs):
        self.file_path = file_path
        self.text_attr = text_attr
        self.trait_attrs = trait_attrs

    def load_data(self):
        df = pd.read_csv(filepath_or_buffer=self.file_path)

        return df[[self.text_attr, *self.trait_attrs]]
