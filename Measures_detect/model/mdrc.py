import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config.dataset_loader import dataset


class MDRC(nn.Module):
    def __init__(self, vocab=None, embedding_size=100, output_size=3, hidden_size=50, num_layers=5,
                 batch_first=True, dropout=0.001):
        super(MDRC, self).__init__()
        self.input_size = embedding_size
        self.out_size = output_size
        self.num_layers = num_layers
        self.dropout_size = dropout
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(vocab,freeze=False)
        self.lstm1 = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True,
                             batch_first=batch_first, dropout=dropout)
        self.lstm2 = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True,
                             batch_first=batch_first, dropout=dropout)
        self.conv1 = nn.Conv2d(1, 1, 2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(hidden_size, output_size)
        self.cat_linear = nn.Linear(hidden_size*4, output_size)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n


    def forward(self, file, input, contents_len):
        input1 = []
        for i in input:
            input1.append(self.embedding(i))
        x_pad = nn.utils.rnn.pad_sequence(input1, batch_first=True, padding_value=0)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_pad, contents_len, batch_first=True)
        
        out1, _ = self.lstm1(x_pack)
        out1, _ = torch.nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        query = self.dropout(out1)
        attn_output1, alpha_n = self.attention_net(out1, query)

        file = self.embedding(file)
        out2, _ = self.lstm2(file)
        batch_size = attn_output1.shape[0]
        out2 = out2[:, -1, :]
        out2_modify = out2.repeat(batch_size, 1).unsqueeze(1)
        out1_modify = attn_output1.unsqueeze(1)

        out = torch.cat([out1_modify, out2_modify], dim=1)
        out = out.unsqueeze(1)
        print(out.shape)
        out = self.conv1(out)
        out = self.pool1(out)
        out = out.squeeze(1).squeeze(1)
        out = self.linear(out)
        return out

'''
    # 消融实验，去掉注意力机制
    def forward(self, file, input, contents_len):
        input1 = []
        for i in input:
            input1.append(self.embedding(i))
        x_pad = nn.utils.rnn.pad_sequence(input1, batch_first=True, padding_value=0)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_pad, contents_len, batch_first=True)
        out1, _ = self.lstm1(x_pack)
        out1, _ = torch.nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)

        out1 = out1[:, -1, :]
        out1 = self.dropout(out1)
        out1 = out1.unsqueeze(1)
        file = self.embedding(file)
        out2, _ = self.lstm2(file)
        batch_size = out1.shape[0]

        out2 = out2[:, -1, :]
        out2_modify = out2.repeat(batch_size, 1).unsqueeze(1)
        out = torch.cat([out1, out2_modify], dim=1)
        out = out.unsqueeze(1)
        out = self.conv1(out)
        out = self.pool1(out)
        out = out.squeeze(1).squeeze(1)

        out = self.linear(out)
        return out
'''
''' # 去掉卷积神经网络
    def forward(self, file, input, contents_len):
        input1 = []
        for i in input:
            input1.append(self.embedding(i))
        x_pad = nn.utils.rnn.pad_sequence(input1, batch_first=True, padding_value=0)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_pad, contents_len, batch_first=True)

        out1, _ = self.lstm1(x_pack)
        out1, _ = torch.nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        query = self.dropout(out1)
        attn_output1, alpha_n = self.attention_net(out1, query)

        file = self.embedding(file)
        out2, _ = self.lstm2(file)
        batch_size = attn_output1.shape[0]
        out2 = out2[:, -1, :]
        out2_modify = out2.repeat(batch_size, 1)

        out = torch.cat((attn_output1, out2_modify), 1)
        out = self.cat_linear(out)
        return out
'''


if __name__ == '__main__':
    writer = SummaryWriter("../logs/logs/model")
    data_dir = '../data/dataset1/dataset_augment_synonym/'
    max_length = 100
    word_or_char = 'word'
    dt = dataset(data_dir, max_length=max_length, word_or_char=word_or_char,
                 fasttext_path='fasttext/FastText_Vector_100')
    embedding = torch.FloatTensor(dt.word_vector)
    mdrc = MDRC(vocab=embedding)
    writer.add_graph(mdrc, input_to_model=None, verbose=False)
