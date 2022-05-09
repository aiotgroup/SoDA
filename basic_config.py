import os
from model import (
    ResNet1D,
    MLPMixer,
    ViT,
    LSTM1D,
    SpanClassifier,
)


class BasicConfig(object):
    def __init__(self):
        # model/head/strategy
        self.model_name = 'resnet101'
        self.head_name = 'span_cls'
        self.strategy_name = 'span_cls'

        # 数据预处理脚本config
        self.datasource_path = os.path.join('/data/wuxilei/watch_action_recognizer', 'transform_source')
        self.dataset_path = os.path.join('/data/wuxilei/watch_action_recognizer')
        self.preprocess_method = "upsampling"
        self.preprocess_strategy = 'shuffle_0'
        self.seq_len = 224
        self.is_normalize = True
        self.train_test_ratio = [0.8, 0.2]

        # 训练超参
        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.num_epoch = 1000
        self.opt_method = 'adamw'
        self.lr_rate = 1e-4
        self.lr_rate_adjust_epoch = 20
        self.lr_rate_adjust_factor = 0.5
        self.weight_decay = 1e-4
        self.save_epoch = 10
        self.eval_epoch = 1
        self.patience = 10
        self.check_point_path = os.path.join('/data/wuxilei/watch_action_recognizer/log',
                                             '%s-%s-%d-%s-%s' %
                                             (self.preprocess_method,
                                              self.preprocess_strategy,
                                              self.seq_len,
                                              "normalize" if self.is_normalize else "none",
                                              self.model_name))
        self.use_gpu = True
        self.gpu_device = "0"

        # 测试超参
        self.test_batch_size = 64
        self.model_path = os.path.join(self.check_point_path,
                                       '%s-%s-final' % (self.model_mapping(self.model_name),
                                                        self.head_mapping(self.head_name)))

        self.n_classes = 18

    def model_mapping(self, model_name: str):
        if model_name.startswith('resnet'):
            return ResNet1D.__name__
        elif model_name.startswith('mixer'):
            return MLPMixer.__name__
        elif model_name.startswith('vit'):
            return ViT.__name__
        elif model_name.startswith('lstm'):
            return LSTM1D.__name__

    def head_mapping(self, head_name):
        if head_name == 'span_cls':
            return SpanClassifier.__name__
