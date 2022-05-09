import os
import logging

import scipy.io as scio
from torch.utils.data.dataloader import DataLoader

import utils as init_utils
from data_process import SensorDataset
from pipeline import Trainer

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    basic_config = init_utils.init_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = basic_config.gpu_device

    train_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                          '%s-%s-%d' %
                                          (basic_config.preprocess_method,
                                           basic_config.preprocess_strategy,
                                           basic_config.seq_len),
                                          'train.mat'))
    eval_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                         '%s-%s-%d' %
                                          (basic_config.preprocess_method,
                                           basic_config.preprocess_strategy,
                                           basic_config.seq_len),
                                         'test.mat'))
    train_dataset = SensorDataset(train_mat)
    eval_dataset = SensorDataset(eval_mat)

    strategy = init_utils.init_strategy(basic_config)

    trainer = Trainer(
        strategy=strategy,
        train_data_loader=DataLoader(train_dataset, batch_size=basic_config.train_batch_size, shuffle=True,
                                     drop_last=True),
        eval_data_loader=DataLoader(eval_dataset, batch_size=basic_config.eval_batch_size, shuffle=False),
        num_epoch=basic_config.num_epoch,
        opt_method=basic_config.opt_method,
        lr_rate=basic_config.lr_rate,
        lr_rate_adjust_epoch=basic_config.lr_rate_adjust_epoch,
        lr_rate_adjust_factor=basic_config.lr_rate_adjust_factor,
        weight_decay=basic_config.weight_decay,
        save_epoch=basic_config.save_epoch,
        eval_epoch=basic_config.eval_epoch,
        patience=basic_config.patience,
        check_point_path=basic_config.check_point_path,
        use_gpu=basic_config.use_gpu,
    )

    trainer.training()
