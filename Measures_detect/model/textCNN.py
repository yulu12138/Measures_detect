# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class textCNN(nn.Module):
    def __init__(self, embedding, classfiy=3, num_filters=256, embedding_size=100, filter_sizes=(2, 3, 4), dropout=0.5):
        super(textCNN, self).__init__()
        self.classfiy = classfiy
        self.num_filters = num_filters
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embedding_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), classfiy)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        print(out.shape)
        out = out.unsqueeze(1)
        print(out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
