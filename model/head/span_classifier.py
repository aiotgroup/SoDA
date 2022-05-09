import torch.nn as nn


class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super(SpanClassifier, self).__init__()

        self.head = nn.Linear(hidden_dim, n_classes, bias=False)

        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, features):
        return self.head(features)
