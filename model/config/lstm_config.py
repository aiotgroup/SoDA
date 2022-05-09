class LSTMConfig(object):
    def __init__(self):
        self.seq_len = 224
        self.acc_axis = 3
        self.gyr_axis = 3
        self.num_layers = 2
        self.hidden_dim = 256
        self.dropout = 0.1
        self.batch_first = True
        self.bidirectional = True
