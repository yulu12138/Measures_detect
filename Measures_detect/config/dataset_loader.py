import os

import torch
from LAC import LAC
from gensim.models import FastText
from torch.utils.data import Dataset
from tqdm import tqdm


class dataset(Dataset):
    def __init__(self, data_dir, word_or_char='word', wordcut_dictionary_dir='', vocab_size=5000,
                 fasttext_path='model/fasttext/FastText_Vector_100', max_length=50, classfiy=3):
        '''
        初始化dataset,对数据进行处理，获得train_data、test_data和dev_data，并且获得其词汇表，并且给词汇表一个对应的序号。
        数据的格式应当为[句子序号，句子内容，标签，语句依附（该句子依附于哪个句子，没有依附即为-1）]
        :param data_dir:数据集的地址，其中应当包括train文件夹，test文件夹和dev文件夹
        :param word_or_char:语句是否需要分词
        :param wordcut_dictionary_dir:如果需要分词，是否需要加载自己定义的词典，如果需要加载，输入为词典的地址
        '''
        self.data_dir = data_dir
        self.wordcut_dictionary_dir = wordcut_dictionary_dir
        self.word_or_char = word_or_char
        self.fasttext_path = fasttext_path
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.classfiy = classfiy
        self.train_data, self.train_att_list, self.train_file_count, self.train_sentence_count = self.get_data('train/')
        self.test_data, self.test_att_list, self.test_file_count, self.test_sentence_count = self.get_data('test/')
        self.dev_data, self.dev_att_list, self.dev_file_count, self.dev_sentence_count = self.get_data('dev/')
        self.data = self.train_data + self.test_data + self.dev_data
        self.get_maxlength()
        self.wordlist, self.vocab = self.getvocab(self.data)
        self.word_vector = self.get_vocab_vector()

    def get_data(self, data_path='train/'):
        '''
        获得数据并进行导入
        :param data_path: 选择是导入指定文件夹下的内容
        :return: 返回两个列表，一个为数据的列表，其中每一个元素为一个tuple，tuple中有文件的名字，和一个list两个元素，里面的list是一个句子和
        对应标签的元组。第二个列表是一个依附的列表，列表中每一个元素是一个元组，为文件名，和该文件中所有句子的依附列表(表示这个句子依附于哪一个句子)
        sample：[(文件名1,[(句子1,标签1),(句子2,标签1),(句子3,标签2)]),(文件名2,[(句子1,标签3),(句子2,标签2)])],
                [(文件名1,[-1,0,0]),(文件名2,[-1,0])]（没有依附的话就为-1）
        '''
        filelist = os.listdir(self.data_dir + data_path)
        file_count = len(filelist)
        sentence_count = 0
        result = []
        att_list = []
        if self.word_or_char == 'char':
            for file in filelist:
                file_name = []
                for word in file.replace('.txt', ''):
                    file_name.append(word)
                with open(self.data_dir + data_path + file, 'r', encoding='utf-8') as infile:
                    target_temp = []
                    sentence_temp = []
                    temp_sequence = []
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        content, target, sequence = set[1], set[2], set[3]
                        temp = []
                        sentence_count += 1
                        for i in content:
                            temp.append(i)
                        try:
                            if int(target) > (self.classfiy - 1):
                                print('错误2：数据集中target超过类别大小，已自动忽略，具体文件为\t{}'.format(self.data_dir + data_path + file))
                            else:
                                target_temp.append(int(target))
                                sentence_temp.append(temp)
                                temp_sequence.append(sequence)
                        except:
                            print('错误1:数据集中target存在问题，已自动忽略，具体文件为\t{}'.format(self.data_dir + data_path + file))
                    result.append((file_name, [sentence_temp, target_temp]))
                    att_list.append((file_name, temp_sequence))
        elif self.word_or_char == 'word':
            lac = LAC(mode='seg')
            for file in filelist:
                file_name = []
                for word in lac.run(file.replace('.txt', '')):
                    file_name.append(word)
                with open(self.data_dir + data_path + file, 'r', encoding='utf-8') as infile:
                    target_temp = []
                    sentence_temp = []
                    temp_sequence = []
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        content, target, sequence = set[1], set[2], set[3]
                        temp = []
                        sentence_count += 1
                        for word in lac.run(content):
                            temp.append(word)
                        try:
                            if int(target) > (self.classfiy - 1):
                                print('错误2：数据集中target超过类别大小，已自动忽略，具体文件为\t{}'.format(self.data_dir + data_path + file))
                            else:
                                target_temp.append(int(target))
                                sentence_temp.append(temp)
                                temp_sequence.append(sequence)
                        except:
                            print('错误1:数据集中target存在问题，已自动忽略，具体文件为\t{}'.format(self.data_dir + data_path + file))
                    result.append((file_name, [sentence_temp, target_temp]))
                    att_list.append((file_name, temp_sequence))
        return result, att_list, file_count, sentence_count

    def get_maxlength(self):
        max_length = 0
        for dt in self.data:
            file, [sentences, targets] = dt
            max_length = max(max_length, len(file))
            for sentence in sentences:
                max_length = max(max_length, len(sentence))
        # 如果数据集中的最大句子长度小于设定的最大句子长度，那么修正max_length
        if max_length < self.max_length:
            print('修正最大长度(max_length)为:{}'.format(max_length))
            self.max_length = max_length

    def to_sequence(self, data, number='True'):
        result = []
        for dt in data:
            file, [sentences, targets] = dt
            file_se = []
            sequences = []
            for sentence in sentences:
                sequence = []
                length = len(sentence)
                if length > self.max_length:
                    sentence = sentence[:self.max_length]
                for j in sentence:
                    if number:
                        sequence.append(self.vocab[j])
                    else:
                        sequence.append(j)
                # print(sequence)
                sequences.append(sequence)
            for j in file:
                if number:
                    file_se.append(self.vocab[j])
                else:
                    file_se.append(j)
            result.append((file_se, [sequences, targets]))
        return result

    def getvocab(self, data):
        '''
        得到词典，将传入的数据进行处理，根据self.word_or_char选择对数据进行word还是char处理，给每一个词/字分配特定序号。
        :param data: 格式为get_data函数中返回的result（第一个返回）的数据格式
        :return: 一个列表一个字典，第一个列表为词的列表,第二个字典为每个字/词对应的序号
        '''
        worddict = {}
        for i in data:
            file, [sentences, targets] = i
            for word in file:
                if word in worddict:
                    worddict[word] += 1
                else:
                    worddict[word] = 1
            for sentence in sentences:
                for i in sentence:
                    if i in worddict:
                        worddict[i] += 1
                    else:
                        worddict[i] = 1

        worddict_sort = sorted(worddict.items(), key=lambda x: x[1], reverse=True)
        wordlist = ['<pad>']
        for key, value in worddict_sort:
            wordlist.append(key)
        word2id = {}
        count = 0
        for i in wordlist:
            word2id[i] = count
            count += 1
        '''# 使用gensim构建词汇表
        id2word = gensim.corpora.Dictionary([wordlist])
        # 每个词分配一个独特的ID
        word2id = id2word.token2id'''
        print(len(wordlist))
        print(word2id)
        return wordlist, word2id

    def get_vocab_vector(self):
        # 将词语转换成词向量，保存至self.word_vector
        word_vector = []
        fasttext = FastText.load(self.fasttext_path)
        for word in self.wordlist:
            word_vector.append(fasttext.wv.__getitem__(word))
        return word_vector

    def add_word(self, word):
        # 在word增加词汇
        fasttext = FastText.load()
        self.wordlist.append(word)
        self.vocab[word] = len(self.wordlist)
        self.word_vector.append(fasttext[word])

    def get_sequence(self, sequence, split_or_not=False, seg=' '):
        '''
        将给定的序列转化为向量的形式进行返回
        :param sequence:一个中文文本的序列,string.
        :param split_or_not:传入的数据是否分割,如果没有分割并且self.word_or_char为word，那么就先进行分割
        :param seg:以什么作为分隔符，默认空格为分隔符
        :return:返回一个向量，对应于给定的序列
        '''
        sequence_vector = []
        if split_or_not == False and self.word_or_char == 'word':
            lac = LAC(mode='seg')
            words = lac.run(sequence)
        else:
            words = sequence.split(seg)
        for word in words:
            if word not in self.wordlist:
                self.add_word(word)
            sequence_vector.append(self.word_vector[self.vocab[word]])
        return sequence_vector

    def to_pack_sequence(self, data):
        data = self.to_sequence(data)

        def sort1(contents, targets):
            map1 = []
            for i, j in zip(contents, targets):
                map1.append((i, j))
            r1 = sorted(map1, key=lambda x: len(x[0]), reverse=True)
            contents_new = []
            targets_new = []
            lengths = []
            for i in r1:
                content, target = i
                contents_new.append(torch.tensor(content))
                targets_new.append(target)
                lengths.append(len(content))
            targets_new = torch.tensor(targets_new)
            lengths = torch.tensor(lengths)
            return contents_new, targets_new, lengths

        sequence = []
        for sample in data:
            file, (contents, targets) = sample
            contents, targets, contents_len = sort1(contents, targets)
            # sample = nn.utils.rnn.pad_sequence(contents,batch_first=True,padding_value=0)
            # x_packed = nn.utils.rnn.pack_padded_sequence(sample,contents_len,batch_first=True)
            file = torch.tensor(file)
            sequence.append([file, [contents, targets], contents_len])
        return sequence

    def to_pack_sequence_string(self, data):
        data = self.to_sequence(data,number=False)

        def sort1(contents, targets):
            map1 = []
            for i, j in zip(contents, targets):
                map1.append((i, j))
            r1 = sorted(map1, key=lambda x: len(''.join(x[0])), reverse=True)

            contents, targets, lengths = [],[],[]
            for i in r1:
                content, target = i
                content = ''.join(content)
                contents.append(content)
                targets.append(target)
                lengths.append(len(content))
            lengths = torch.tensor(lengths)
            return contents, targets, lengths

        sequence = []

        from transformers import BertTokenizer, BertModel
        # 装载bert模型
        tokenizer = BertTokenizer.from_pretrained('model/bert')  # 包含上面三个文件的文件夹目录
        # model = BertModel.from_pretrained('model/bert')

        for sample in tqdm(data):
            file, (contents, targets) = sample
            contents, targets, contents_len = sort1(contents, targets)
            new_contents = []
            # new_file = torch.tensor(model(torch.tensor(tokenizer.encode(file)).unsqueeze(0))[0])
            new_file = torch.tensor(tokenizer.encode(file)).unsqueeze(0)

            for content in contents:
                tok = torch.tensor(tokenizer.encode(content)).unsqueeze(0)
                # ten = model(torch.tensor(tok))[0]
                # new_contents.append(torch.tensor(ten))
                new_contents.append(torch.tensor(tok)[0])
            # sample = nn.utils.rnn.pad_sequence(contents,batch_first=True,padding_value=0)
            # x_packed = nn.utils.rnn.pack_padded_sequence(sample,contents_len,batch_first=True)
            sequence.append([new_file, [new_contents, targets], contents_len])
        return sequence

if __name__ == '__main__':
    dt = dataset('../data/dataset/dataset/', fasttext_path='../model/fasttext/FastText_Vector_100')
    train_data = dt.to_sequence(dt.train_data)

    # print(train_data)
    # embed = nn.Embedding.from_pretrained(torch.FloatTensor(dt.word_vector))
    '''print(embed)
    print(embed(torch.LongTensor([0])))
    fasttext = FastText.load('../model/fasttext/FastText_Vector_100')
    print(fasttext['<pad>'])'''
    # print(train_data)

    data = dt.to_pack_sequence_string(dt.train_data)
    for i in data:
        print(i)
        break
