class MLPMixerConfig(object):
    def __init__(self):
        self.seq_len = 224
        self.patch_size = 16
        self.acc_axis = 3
        self.gyr_axis = 3
        self.hidden_dim = 256
        self.num_layers = 6
        self.expansion_factor = 4
        self.dropout = 0.1
