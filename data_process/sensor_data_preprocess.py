import argparse
import logging
import os
import random
import time

import numpy
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F

from sensor_data import SensorData

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="数据预处理：在序列轴上进行上采样到固定长度，依据不同策略切分数据成训练/测试集")

parser.add_argument("-i", '--input', dest="input_path", required=True, type=str,
                    help="数据源路径")

parser.add_argument("-o", '--output', dest="output_path", required=True, type=str,
                    help="数据经过转换后输出路径")

parser.add_argument("-m", '--method', dest="method", required=True, type=str,
                    help="预处理数据方法：upsampling：上采用到对应长度；padding：填充零到对应长度。")

parser.add_argument("-s", '--strategy', dest="strategy", required=True, type=str,
                    help="制作数据策略：normal_i(0-4)/user_i(1-10)/shuffle")

parser.add_argument("-l", '--length', dest="seq_len", required=True, type=int,
                    help="经过处理后序列长度")

parser.add_argument("-n", '--normalize', dest="is_normalize", required=False, type=bool, default=True,
                    help="是否对数据进行预归一化处理")


def _init_config(datasource_path: os.path):
    # user_id: 1 - 10
    # action_id: 1 - 18
    # attempt_id: 0 - 9
    file_path_list = os.listdir(datasource_path)
    users = set()
    actions = set()
    attempts = set()
    for file_path in file_path_list:
        user_id, action_id, attempt_id = file_path.split('.')[0].split('_')
        users.add(user_id)
        actions.add(action_id)
        attempts.add(attempt_id)
    return sorted(users), sorted(actions), sorted(attempts), sorted(file_path_list)


def upsampling_method(origin_mat, factor, seq_len):
    acc = F.interpolate(torch.from_numpy(origin_mat['accData']), size=(factor * seq_len), mode='linear')
    gyr = F.interpolate(torch.from_numpy(origin_mat['gyrData']), size=(factor * seq_len), mode='linear')
    label = (origin_mat['label'][0][0] - 1) * torch.ones((1, factor * seq_len))
    return acc, gyr, label


def padding_method(origin_mat, factor, seq_len):
    length = origin_mat['label'].shape[1]
    assert length == origin_mat['accData'].shape[2]
    assert length == origin_mat['gyrData'].shape[2]
    acc = F.pad(torch.from_numpy(origin_mat['accData']), (0, factor * seq_len - length), "constant", 0)
    gyr = F.pad(torch.from_numpy(origin_mat['gyrData']), (0, factor * seq_len - length), "constant", 0)
    label = (origin_mat['label'][0][0] - 1) * torch.ones((1, factor * seq_len))
    return acc, gyr, label


def preprocess_with_certain_method(datasource_path: os.path,
                                   output_dir: os.path,
                                   method: str = 'upsampling',
                                   strategy: str = 'normal_0',
                                   ratio: list = [0.8, 0.2],
                                   seq_len: int = 224,
                                   is_nomalize: bool = True):
    """
    在序列轴上采用进行相应方法到固定长度，依据不同策略切分数据成训练/测试集
    :param datasource_path: 数据源路径
    :param output_dir: 输出路径
    :param method: 预处理方法
        upsampling: 上采用数据到相应seq_len长度的倍数
        padding: 填充数据到相应seq_len长度的倍数
    :param strategy: 制作数据策略: normal/user_independent/shuffle/
        normal_i: test第i*2 ~ i*2 + 1次尝试，其余尝试数据进行训练，i：0~4
        user_i: test第i人，其余人数据进行train，i：1~10
        shuffle_i: 第i次随机shuffle，上面两种的折衷
    :param ratio: train/test比例
    :param seq_len: 固定长度
    :return: 在output_dir下生成train.mat/test.mat
    """
    # method, strategy, seq_len, normalize/none
    dst_dir = os.path.join(output_dir, '%s-%s-%d' %
                           (method, strategy, seq_len))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    logger.info("初试化源数据参数")

    users, actions, attempts, file_path_list = _init_config(datasource_path)

    logger.info("开始对数据进行%s处理到固定长度: %d" % (method, seq_len))

    merge_by_user_id = [list() for _ in users]
    merge_by_attempt_id = [list() for _ in attempts]
    for file_path in file_path_list:
        user_id, action_id, attempt_id = file_path.split('.')[0].split('_')
        origin_mat = scio.loadmat(os.path.join(datasource_path, file_path))
        length = origin_mat['label'].shape[1]
        # 由于原数据最短长度为40，所以只有超过40的末尾才做上采用或者padding
        factor = length // seq_len + (1 if length % seq_len >= 40 else 0)
        assert origin_mat['label'][0][0] == int(action_id)

        if method == "upsampling":
            acc, gyr, label = upsampling_method(origin_mat, factor, seq_len)
        elif method == "padding":
            acc, gyr, label = padding_method(origin_mat, factor, seq_len)

        for i in range(factor):
            sensor_data = SensorData(user_id=int(user_id),
                                     action_id=int(action_id),
                                     attempt_id=int(attempt_id),
                                     accData=acc[:, :, i * seq_len:(i + 1) * seq_len],
                                     gyrData=gyr[:, :, i * seq_len:(i + 1) * seq_len],
                                     label=label[:, i * seq_len:(i + 1) * seq_len])
            merge_by_user_id[sensor_data.user_id - 1].append(sensor_data)
            merge_by_attempt_id[sensor_data.attempt_id].append(sensor_data)

    logger.info("处理完成，开始基于策略对数据进行划分: %s" % strategy)

    train_data = {
        'accData': list(),
        'gyrData': list(),
        'label': list(),
        'user_id': list(),
    }
    test_data = {
        'accData': list(),
        'gyrData': list(),
        'label': list(),
        'user_id': list(),
    }
    if strategy.startswith("normal_"):
        _, target_attempt = strategy.split('_')
        target_attempt = int(target_attempt)
        for attempt_id, data in enumerate(merge_by_attempt_id):
            if attempt_id // 2 == target_attempt:
                # 加入测试集
                for instance in data:
                    assert instance.attempt_id == attempt_id
                    test_data['accData'].append(instance.accData)
                    test_data['gyrData'].append(instance.gyrData)
                    test_data['label'].append(instance.label)
                    test_data['user_id'].append(torch.from_numpy(np.array([[instance.user_id]])))
            else:
                # 加入训练集
                for instance in data:
                    assert instance.attempt_id == attempt_id
                    train_data['accData'].append(instance.accData)
                    train_data['gyrData'].append(instance.gyrData)
                    train_data['label'].append(instance.label)
                    train_data['user_id'].append(torch.from_numpy(np.array([[instance.user_id]])))
    elif strategy.startswith("shuffle_"):
        assert len(ratio) == 2
        for attempt_id, data in enumerate(merge_by_attempt_id):
            # 先shuffle
            random.seed(time.time())
            random.shuffle(data)
            # 前ratio[0]加入训练集
            for instance in data[:int(len(data) * ratio[0])]:
                assert instance.attempt_id == attempt_id
                train_data['accData'].append(instance.accData)
                train_data['gyrData'].append(instance.gyrData)
                train_data['label'].append(instance.label)
                train_data['user_id'].append(torch.from_numpy(np.array([[instance.user_id]])))
            # 后ratio[1]加入测试集
            for instance in data[int(len(data) * ratio[0]):]:
                assert instance.attempt_id == attempt_id
                test_data['accData'].append(instance.accData)
                test_data['gyrData'].append(instance.gyrData)
                test_data['label'].append(instance.label)
                test_data['user_id'].append(torch.from_numpy(np.array([[instance.user_id]])))
    else:
        _, target_user = strategy.split('_')
        target_user = int(target_user)
        for user_id, data in enumerate(merge_by_user_id):
            if user_id + 1 == target_user:
                # 加入测试集
                for instance in data:
                    assert instance.user_id == user_id + 1
                    test_data['accData'].append(instance.accData)
                    test_data['gyrData'].append(instance.gyrData)
                    test_data['label'].append(instance.label)
                    test_data['user_id'].append(torch.from_numpy(np.array([[instance.user_id]])))
            else:
                # 加入训练集
                for instance in data:
                    assert instance.user_id == user_id + 1
                    train_data['accData'].append(instance.accData)
                    train_data['gyrData'].append(instance.gyrData)
                    train_data['label'].append(instance.label)
                    train_data['user_id'].append(torch.from_numpy(np.array([[instance.user_id]])))
    for key, value in train_data.items():
        train_data[key] = torch.vstack(value)
    for key, value in test_data.items():
        test_data[key] = torch.vstack(value)

    if is_nomalize:
        logger.info("对数据进行归一化")
        _normalize(train_data, test_data)

    for key, value in train_data.items():
        train_data[key] = numpy.array(value)
    for key, value in test_data.items():
        test_data[key] = numpy.array(value)

    logger.info("数据集生成完成，保存")
    scio.savemat(os.path.join(dst_dir, 'train.mat'), train_data)
    scio.savemat(os.path.join(dst_dir, 'test.mat'), test_data)


def _normalize(train_data, test_data):
    def get_mean_std(data):
        axis = data.size(1)

        mean = torch.mean(data.permute(1, 0, 2).reshape(axis, -1), dim=-1).unsqueeze(0).unsqueeze(2)
        std = torch.std(data.permute(1, 0, 2).reshape(axis, -1), dim=-1).unsqueeze(0).unsqueeze(2)
        return mean, std

    def normalize(data, mean, std):
        return (data - mean) / std

    # 所有数据以训练集的均值和标准差做归一化
    acc_mean, acc_std = get_mean_std(train_data['accData'])
    train_data['accData'] = normalize(train_data['accData'], acc_mean, acc_std)
    test_data['accData'] = normalize(test_data['accData'], acc_mean, acc_std)

    gyr_mean, gyr_std = get_mean_std(train_data['gyrData'])
    train_data['gyrData'] = normalize(train_data['gyrData'], gyr_mean, gyr_std)
    test_data['gyrData'] = normalize(test_data['gyrData'], gyr_mean, gyr_std)


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess_with_certain_method(datasource_path=args.input_path,
                                   output_dir=args.output_path,
                                   method=args.method,
                                   strategy=args.strategy,
                                   ratio=[0.8, 0.2],
                                   seq_len=args.seq_len)
