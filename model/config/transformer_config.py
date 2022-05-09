class TransformerConfig(object):
    def __init__(self):
        self.seq_len = 224
        self.patch_size = 16
        self.acc_axis = 3
        self.gyr_axis = 3
        self.d_model = 768
        self.num_layers = 12
        self.n_head = 12
        self.expansion_factor = 4
        self.dropout = 0.1
        self.pooling = False
