import os
import argparse
import logging
import torch

from basic_config import BasicConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def init_configs() -> BasicConfig:
    parser = argparse.ArgumentParser(description="模型训练")

    """数据集及其处理相关参数"""
    parser.add_argument('--datasource_path', dest="datasource_path", required=False, type=str,
                        help="原数据路径，未经过切分")
    parser.add_argument('--dataset_path', dest="dataset_path", required=True, type=str,
                        help="包含train.mat/test.mat数据集路径")
    parser.add_argument('--preprocess_method', dest="preprocess_method", required=True, type=str,
                        help="预处理数据集方法策略，上采用方法：upsampling，填充零方法：padding")
    parser.add_argument('--preprocess_strategy', dest="preprocess_strategy", required=True, type=str,
                        help="预处理数据集切分策略，基于此加载相应的数据集：normal_i(0-4)/user_i(1-10)/shuffle_i(0-9)")
    parser.add_argument('--seq_len', dest="seq_len", required=True, type=int, default=224,
                        help="数据集经过处理后序列长度")
    parser.add_argument('--is_normalize', dest="is_normalize", required=False, type=bool, default=True,
                        help="是否对整个数据集做归一化处理")

    """训练超参"""
    parser.add_argument('--train_batch_size', dest="train_batch_size", required=False, type=int, default=64,
                        help="训练使用batch_size")
    parser.add_argument('--eval_batch_size', dest="eval_batch_size", required=False, type=int, default=64,
                        help="验证使用batch_size")
    parser.add_argument('--num_epoch', dest="num_epoch", required=False, type=int, default=1000,
                        help="训练epoch")
    parser.add_argument('--opt_method', dest="opt_method", required=False, type=str, default="adam",
                        help="训练模型使用优化器")
    parser.add_argument('--lr_rate', dest="lr_rate", required=False, type=float, default=1e-4,
                        help="训练学习率")
    parser.add_argument('--lr_rate_adjust_epoch', dest="lr_rate_adjust_epoch", required=False, type=int, default=20,
                        help="每训练一定epoch后根据调整因子调整学习率")
    parser.add_argument('--lr_rate_adjust_factor', dest="lr_rate_adjust_factor", required=False, type=float,
                        default=0.5,
                        help="每训练一定epoch后乘以学习率")
    parser.add_argument('--weight_decay', dest="weight_decay", required=False, type=float, default=1e-4,
                        help="训练正则化系数")
    parser.add_argument('--save_epoch', dest="save_epoch", required=False, type=int, default=50,
                        help="训练中途每隔一定epoch数后对模型进行保存")
    parser.add_argument('--eval_epoch', dest="eval_epoch", required=False, type=int, default=1,
                        help="训练中途每隔一定epoch数后使用模型在验证集上验证")
    parser.add_argument('--patience', dest="patience", required=False, type=int, default=10,
                        help="Early Stop机制，超过一定轮数eval loss未下降则停止训练")

    parser.add_argument('--check_point_path', dest="check_point_path", required=True, type=str,
                        help="训练中途临时保存根路径，后面经过处理根据不同模型和数据切分策略做分类")

    """GPU相关参数"""
    parser.add_argument('--use_gpu', dest="use_gpu", required=True, type=bool, default=torch.cuda.is_available(),
                        help="训练是否使用GPU")
    parser.add_argument('--gpu_device', dest="gpu_device", required=True, type=str, default="0",
                        help="训练使用的GPU编号：0-3")

    """模型选择相关参数"""
    parser.add_argument('--model_name', dest="model_name", required=True, type=str, default="resnet101",
                        help="训练使用的模型名")
    parser.add_argument('--head_name', dest="head_name", required=True, type=str, default="span_cls",
                        help="训练使用的预测头")
    parser.add_argument('--strategy_name', dest="strategy_name", required=True, type=str, default="span_cls",
                        help="使用的训练策略")

    """测试相关参数"""
    parser.add_argument('--test_batch_size', dest="test_batch_size", required=False, type=int, default=64,
                        help="测试使用batch_size")
    parser.add_argument('--n_classes', dest="n_classes", required=False, type=int, default=18,
                        help="分类个数")

    configs = BasicConfig()
    args = parser.parse_args()
    configs.model_name = args.model_name
    configs.head_name = args.head_name
    configs.strategy_name = args.strategy_name

    configs.datasource_path = args.datasource_path
    configs.dataset_path = args.dataset_path
    configs.preprocess_method = args.preprocess_method
    configs.preprocess_strategy = args.preprocess_strategy
    configs.seq_len = args.seq_len
    configs.is_normalize = args.is_normalize

    configs.train_batch_size = args.train_batch_size
    configs.eval_batch_size = args.eval_batch_size
    configs.num_epoch = args.num_epoch
    configs.opt_method = args.opt_method
    configs.lr_rate = args.lr_rate
    configs.lr_rate_adjust_epoch = args.lr_rate_adjust_epoch
    configs.lr_rate_adjust_factor = args.lr_rate_adjust_factor
    configs.weight_decay = args.weight_decay
    configs.save_epoch = args.save_epoch
    configs.eval_epoch = args.eval_epoch
    configs.patience = args.patience
    configs.check_point_path = os.path.join(args.check_point_path,
                                            '%s-%s-%d-%s' %
                                            (configs.preprocess_method,
                                             configs.preprocess_strategy,
                                             configs.seq_len,
                                             configs.model_name))
    configs.use_gpu = args.use_gpu
    configs.gpu_device = args.gpu_device

    configs.test_batch_size = args.test_batch_size
    configs.model_path = os.path.join(configs.check_point_path,
                                      '%s-%s-final' % (configs.model_mapping(configs.model_name),
                                                       configs.head_mapping(configs.head_name)))
    configs.n_classes = args.n_classes

    if not os.path.exists(configs.check_point_path):
        os.makedirs(configs.check_point_path)
    return configs


def init_model(model_name):
    logger.info('初始化模型：%s' % model_name)
    if model_name.startswith('resnet'):
        from model import resnet, ResNetConfig
        return resnet(model_name, ResNetConfig())
    elif model_name.startswith('mixer'):
        from model import mlp_mixer, MLPMixerConfig
        return mlp_mixer(model_name, MLPMixerConfig())
    elif model_name.startswith('vit'):
        from model import vit, TransformerConfig
        return vit(model_name, TransformerConfig())
    elif model_name.startswith('lstm'):
        from model import lstm, LSTMConfig
        return lstm(model_name, LSTMConfig())


def init_head(head_name, hidden_dim, n_classes):
    logger.info('初试化预测头：%s' % head_name)
    if head_name == 'span_cls':
        from model import SpanClassifier
        return SpanClassifier(hidden_dim, n_classes)


def init_strategy(config: BasicConfig):
    logger.info('初试化训练策略：%s' % config.strategy_name)
    model = init_model(config.model_name)
    head = init_head(config.head_name, model.get_output_size(), config.n_classes)
    if config.strategy_name == 'span_cls':
        from strategy import SpanCLSStrategy
        return SpanCLSStrategy(model, head)
