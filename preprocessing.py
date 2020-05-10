import numpy
import pandas as pd

from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing import sequence
import configuration as config


class Preprocessing:
    # handle attributes and attribute
    def preprocess(self, docs, labels, attribute):
        t = Tokenizer(num_words=20000, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
        t.fit_on_texts(docs)

        encoded_docs = t.texts_to_sequences(docs)

        print("BEFORE Pruning:")

        self.get_statistics(encoded_docs, labels, attribute)

        idx2word = {v: k for k, v in t.word_index.items()}

        # stopwords
        stopwrd = set(stopwords.words('english'))

        # handle abbreviation
        def abbreviation_handler(text):
            ln = text.lower()
            # case replacement
            ln = ln.replace(r"'t", " not")
            ln = ln.replace(r"'s", " is")
            ln = ln.replace(r"'ll", " will")
            ln = ln.replace(r"'ve", " have")
            ln = ln.replace(r"'re", " are")
            ln = ln.replace(r"'m", " am")

            # delete single '
            ln = ln.replace(r"'", " ")

            return ln

        # handle stopwords
        def stopwords_handler(text):
            words = text.split()
            new_words = [w for w in words if w not in stopwrd]

            return ' '.join(new_words)

        # get post-tokenized docs
        def sequence_to_text(list_of_sequences):
            tokenized_list = []

            for text in list_of_sequences:
                new_text = ''

                for num in text:
                    new_text += idx2word[num] + ' '

                new_text = abbreviation_handler(new_text)
                new_text = stopwords_handler(new_text)

                tokenized_list.append(new_text)

            return tokenized_list

        return sequence_to_text(encoded_docs)

    # handle single and multiple attributes
    def get_statistics(self, encoded_docs, Ys, attributes):

        # explore encoded docs: find sequence length distribution
        result = [len(x) for x in encoded_docs]

        print("Min=%d, Mean=%d, Max=%d" % (numpy.min(result), numpy.mean(result), numpy.max(result)))

        data_max_seq_length = numpy.max(result) + 1

        # explore Y for each attribute
        if type(attributes) == list:
            for idx, attribute in enumerate(attributes):
                Y = Ys[:, idx]
                self.get_single_statistics(Y, attribute)
        elif type(attributes) == str:
            self.get_single_statistics(Ys, attributes)

        return data_max_seq_length

    # get statistics of a single attribute
    def get_single_statistics(self, Y, attribute):

        # find majority distribution
        ct = Counter(Y)
        majorityDistribution = max([ct[k] * 100 / float(Y.shape[0]) for k in ct])

        print("Total majority is {0} for {1}.".format(majorityDistribution, attribute))

    def load_data(self, attribute):
        # read the data
        df = pd.read_csv(config.FILENAME)

        print("The size of data is {0}".format(df.shape[0]))

        docs = df[config.column_to_read].astype(str).values.tolist()
        labels = df[attribute].values  # attribute is either string or a list of strings
        # preprocess data before feeding into tokenizer
        docs = self.preprocess(docs, labels, attribute)

        # tokenize the data
        t = Tokenizer(num_words=config.MAX_NUM_WORDS)
        t.fit_on_texts(docs)
        encoded_docs = t.texts_to_sequences(docs)
        print("Real Vocab Size: %d" % (len(t.word_index) + 1))
        print("Truncated Vocab Size: %d" % config.MAX_NUM_WORDS)

        self.word_index = t.word_index

        # perform Bag of Words
        print("AFTER Pruning:")
        data_max_seq_length = self.get_statistics(encoded_docs, labels,
                                                  attribute)  # can be used to replace MAX_SEQ_LENGTH
        self.X = sequence.pad_sequences(encoded_docs,
                                        maxlen=config.MAX_SEQ_LENGTH)  # use either a fixed max-length or the real max-length from data

        # for use with categorical_crossentropy
        self.onehot_Y = labels

        return t
