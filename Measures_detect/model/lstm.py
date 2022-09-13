import math

import torch
import torch.nn.functional as F
from torch import nn


class LSTM(nn.Module):
    def __init__(self, vocab=None, embedding_size=100, output_size=3, hidden_size=10, num_layers=5,
                 batch_first=True, dropout=0.001):
        super(LSTM, self).__init__()
        self.input_size = embedding_size
        self.out_size = output_size
        self.num_layers = num_layers
        self.dropout_size = dropout
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(vocab)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True,
                            batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, input, contents_len):
        input1 = []
        for i in input:
            input1.append(self.embedding(i))
        x_pad = nn.utils.rnn.pad_sequence(input1, batch_first=True, padding_value=0)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_pad, contents_len, batch_first=True)
        # inputs = self.embedding(input)

        '''self.hidden = (torch.rand(self.num_layers * 2, input.size(0), self.hidden_size),
                       torch.rand(self.num_layers * 2, input.size(0), self.hidden_size))'''
        out, _ = self.lstm(x_pack)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        query = self.dropout(out)
        attn_output, alpha_n = self.attention_net(out, query)

        out = self.linear(attn_output)
        return out


if __name__ == '__main__':
    lstm = LSTM(embedding_size=3, output_size=2, batch_first=True)
    list1 = [[[2.0, 3.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 1.0], [2.0, 3.0, 5.0]],
             [[2.0, 3.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 1.0], [2.0, 3.0, 5.0]]]
    tt1 = torch.tensor(list1)
    # print(tt1)
    tt2 = torch.reshape(tt1, (2, 4, 3))
    out = lstm(tt2)
    print(out)
    print(torch.tensor([-1]).shape)
    loss_fn = nn.CrossEntropyLoss()
    print(loss_fn(out, torch.tensor([1, 0])))
