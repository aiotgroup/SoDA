import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Tester(object):
    def __init__(self,
                 strategy: nn.Module,
                 eval_data_loader: DataLoader,
                 n_classes: int,
                 output_path: os.path,
                 use_gpu=True
                 ):
        super(Tester, self).__init__()

        self.strategy = strategy

        self.eval_data_loader = eval_data_loader

        self.use_gpu = use_gpu

        self.n_classes = n_classes
        self.confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
        self.output_path = output_path

    def _to_var(self, data: tuple):
        if self.use_gpu:
            return [Variable(x.cuda()) for x in data]
        else:
            return [Variable(x) for x in data]

    def _calc_accuracy(self):
        correct = 0
        for i in range(self.n_classes):
            correct += self.confusion[i][i]
        return correct / np.sum(self.confusion)

    def _calc_precision_recall_f1(self):
        precision = [0 for _ in range(self.n_classes)]
        recall = [0 for _ in range(self.n_classes)]
        f1 = [0 for _ in range(self.n_classes)]

        for i in range(self.n_classes):
            precision[i] = self.confusion[i][i] / np.sum(self.confusion[i, :])
            recall[i] = self.confusion[i][i] / np.sum(self.confusion[:, i])

        for i in range(self.n_classes):
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        return precision, recall, f1

    def testing(self):
        if self.use_gpu:
            self.strategy = self.strategy.cuda()
        self.strategy.eval()
        with torch.no_grad():
            for data in tqdm(self.eval_data_loader):
                data = self._to_var(data)
                prob = self.strategy.predict(data[0], data[1])
                prediction = torch.max(prob, dim=1)[1]
                label = data[2][:, 0]
                for pred, gt in zip(prediction, label):
                    self.confusion[pred][gt] += 1

        print('Confusion Matrix: ')
        print(self.confusion)
        accuracy = self._calc_accuracy()
        precision, recall, f1 = self._calc_precision_recall_f1()
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('F1: ' + str(f1))
        print('mAccuracy: ' + str(accuracy))
        print('mPricision: ' + str(np.mean(precision)))
        print('mRecall: ' + str(np.mean(recall)))
        print('mF1: ' + str(np.mean(f1)))

        result = pd.DataFrame(self.confusion)
        result.to_csv(os.path.join(self.output_path, 'confusion_matrix.csv'), index=False, header=False)
