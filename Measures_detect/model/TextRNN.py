import torch
from torch import nn


class TextRNN(nn.Module):
    def __init__(self, embedding=None, embedding_size=100, output_size=3, hidden_size=10, num_layers=5,
                 batch_first=True, dropout=0.001):
        super(TextRNN, self).__init__()
        self.input_size = embedding_size
        self.out_size = output_size
        self.num_layers = num_layers
        self.dropout_size = dropout
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True,
                            batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.5)

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
        out = out.permute(1, 0, 2)[-1]
        out = self.linear(out)
        return out
