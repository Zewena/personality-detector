dims = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']

FILENAME = 'data/essays2007.csv'
column_to_read = 'text'

MAX_NUM_WORDS = 10000
MAX_SEQ_LENGTH = 400
MAX_SENTS = 30
EMBEDDING_DIM = 100


class Params:
    n_epoch = 50
    batch_size = 32
    dropout_rate = 0.5