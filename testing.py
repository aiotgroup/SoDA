import logging
import os

import scipy.io as scio
import torch
from torch.utils.data.dataloader import DataLoader

import utils as init_utils
from data_process import SensorDataset
from pipeline import Tester

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    basic_config = init_utils.init_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = basic_config.gpu_device

    test_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                         '%s-%s-%d' %
                                          (basic_config.preprocess_method,
                                           basic_config.preprocess_strategy,
                                           basic_config.seq_len),
                                         'test.mat'))
    test_dataset = SensorDataset(test_mat)

    strategy = init_utils.init_strategy(basic_config)
    strategy.load_state_dict(torch.load(basic_config.model_path))

    tester = Tester(strategy,
                    eval_data_loader=DataLoader(test_dataset, batch_size=basic_config.test_batch_size, shuffle=False),
                    n_classes=basic_config.n_classes,
                    output_path=basic_config.check_point_path,
                    use_gpu=True)
    tester.testing()
