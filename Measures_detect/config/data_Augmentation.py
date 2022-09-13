import ast
import json
import os
import random
import time
from hashlib import md5

import requests
from LAC import LAC
from gensim.models import FastText
from tqdm import tqdm


class data_Augmentation():
    def __init__(self, triple_file, relation_aggregation_file='data/old_data/relation_aggregation1.txt',
                 fasttext_path='model/fasttext/FastText_Vector_100', train_data='data/dataset/train/',
                 customization_list=['data/new_data/NER/dictionary.txt', 'data/new_data/NER/EIdictionary.txt']):
        self.customization_list = customization_list
        self.train_data = self.get_train_data(train_data)
        self.entity, self.relation, self.triple = self.getEI(triple_file)
        self.entity2id, self.relation2id = self.EI_to_id(relation_aggregation_file)
        self.fasttext_path = fasttext_path

    def getEI(self, triple_file):
        '''
        从三元组文件中得到所有的实体和关系
        :param triple_file: 文件地址
        :return: 返回实体列表、关系列表、三元组列表
        '''
        entity = []
        relation = []
        triple = []
        with open(triple_file, 'r', encoding='utf-8') as infile:
            for line in infile.readlines():
                temp = ast.literal_eval(line.strip())
                triple.append(temp)
                entity.append(temp[0])
                entity.append(temp[2])
                relation.append(temp[1])
        entity = list(set(entity))
        relation = list(set(relation))
        return entity, relation, triple

    def get_train_data(self, train_file):
        filelist = os.listdir(train_file)
        result = {}
        for file in filelist:
            temp = []
            with open(train_file + file, 'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    set = line.strip().split('\t')
                    content, target, sequence = set[1], set[2], set[3]
                    try:
                        target = int(target)
                        sequence = int(sequence)
                        temp.append([content, target, sequence])
                    except:
                        print('{}文件{}数据存在问题，该行已经忽略'.format(file, content))
            result[file.replace('.txt', '')] = temp
        return result

    def EI_to_id(self, relation_aggregation_file):
        '''
        将实体和关系序列化，给每个实体和关系一个对应的标签
        :param relation_aggregation_file: 关系聚合的文件结果，关系聚合结果文件包含语义相似度较高的关系。
        :return: 返回序列化的实体关系,字典，如{'工程总承包企业资质':1,'质量监督机构':2}
        '''
        entity_dict = {}
        relation_dict = {}
        for key, value in enumerate(self.entity):
            entity_dict[value] = key
        with open(relation_aggregation_file, 'r', encoding='utf-8') as infile:
            for key, value in enumerate(infile.readlines()):
                temp = value.strip().split('/')
                for relation in temp:
                    relation_dict[relation] = key
        return entity_dict, relation_dict

    def get_synonym_entity(self, word_similarity=0.5, outpath=''):
        '''
        获得同义的文件，在知识图谱中通过一跳找到在同一个层级下的所有实体，根据语义相似度作为筛选结果。
        :param word_similarity:语义相似度的阈值
        :return:返回序列，每一个value是一个聚合的结果。
        如：[['具备检查条件特定场所', '特定场所'], ['本规定规定', '本规定第二款规定', '小规模食品生产经营者',
         '社会投资简易低风险工程建设项目','规定', '本规定', '北京市高级人民法院审判委员会', '最高人民法院', '下列规定']]
        '''
        fasttext = FastText.load(self.fasttext_path)

        def find_one_hop(triples, entity):
            entity_list = []
            head_list = []
            for triple in triples:
                if entity == triple[2]:
                    head_list.append(triple[0])
            head_list = list(set(head_list))
            for triple in triples:
                if triple[0] in head_list:
                    entity_list.append(triple[2])
            entity_list = list(set(entity_list))
            return entity_list

        synonym_entity = []
        entity_contain = []
        for i in tqdm(self.entity):
            if i not in entity_contain:
                entity_temp = []
                temp = find_one_hop(self.triple, entity=i)
                for word in temp:
                    if fasttext.wv.similarity(i, word) >= word_similarity:
                        entity_temp.append(word)
                synonym_entity += entity_temp
                if len(entity_temp) <= 1:
                    pass
                else:
                    entity_contain.append(entity_temp)

        if os.path.exists(outpath):
            with open(outpath + 'synonym.json', 'w', encoding='utf-8') as infile:
                json.dump(entity_contain, infile, ensure_ascii=False)
        else:
            print('无法找到输出位置outpath:{}'.format(outpath))

        return entity_contain

    def synonym_replace(self, data):
        lac = LAC(mode='seg')
        # 导入词典
        sentences = []
        for dictionary in self.customization_list:
            lac.load_customization(dictionary)
        for sentence in data:
            content, target, sequence = sentence
            temp = []
            word_temp = []
            candidata_temp = {}
            # content = lac.run(content)
            # print(content)
            for word in lac.run(content):
                for lexicon in self.synonym_entity:
                    if word in lexicon and len(lexicon) > 1:
                        word_temp.append(word)
                        te = lexicon
                        te.remove(word)
                        candidata_temp[word] = te
            word_temp = list(set(word_temp))
            if len(word_temp) == 0:
                sentences.append([sentence])
            else:
                number = int(random.random() * 1000) % len(word_temp)
                # for key in range(len(word_temp)):
                content_temp = [content]

                for i in range(number, len(word_temp)):
                    candidata = candidata_temp[word_temp[i]]
                    te = []
                    # number = int(random.random()*100) % len(candidata)
                    for j in content_temp:
                        for k in candidata:
                            temp2 = j
                            te.append(temp2.replace(word_temp[i], k))
                    content_temp = te
                for i in content_temp:
                    temp.append([i, target, sequence])
            sentences.append(temp)
        return sentences

    def augment_data(self, output_path='data/dataset_augment/train/', word_similarity=0.7):
        self.synonym_entity = self.get_synonym_entity(word_similarity=word_similarity)
        result = {}
        for file in tqdm(self.train_data):
            print(file)
            data = self.train_data[file]
            temp = self.synonym_replace(data)
            temp2 = []
            count = 0
            # 筛选每个标签的数量
            # 统计每个label的数量
            label = [0,0,0]
            for sentences in temp:
                for content,target,sequence in sentences:
                    label[int(target)] += 1
            minlabel = min(label)
            label = [minlabel] * 3

            numbers = 0
            for sentences in temp:
                random.shuffle(sentences)
                for sentence in sentences:
                    if numbers % 300 == 0 and numbers != 0:
                        result[file + str(count)] = temp2
                        count += 1
                        temp2 = []
                    else:
                        if label[int(sentence[1])] > 0:
                            label[int(sentence[1])] -= 1
                            temp2.append(sentence)
                            numbers += 1
            result[file + str(count)] = temp2

        for file in result:
            if result[file]:
                with open(output_path + file + '.txt', 'w', encoding='utf-8') as outfile:
                    count = 1
                    for sentence in result[file]:
                        outfile.write(
                            str(count) + '\t' + str(sentence[0]) + '\t' + str(sentence[1]) + '\t' + str(sentence[2]) + '\n')
                        count += 1
        return result

    def random_replace(self, output_path='data/dataset_augment_synonym/train/',
                       stopwords_path='data/file/stopwords.txt'):
        def get_stopwords(infile_path):
            stopwords = [line.strip() for line in open(infile_path, 'r', encoding='utf-8').readlines()]
            return stopwords

        if os.path.exists(stopwords_path):
            stopwords = get_stopwords(stopwords_path)
        else:
            stopwords = []
            print('无法加载停用词库，请重新确认地址\t{}'.format(stopwords_path))

        lac = LAC(mode='seg')
        print('加载lac成功')
        fasttext = FastText.load(self.fasttext_path)
        print('加载fasttext成功')

        for file in tqdm(self.train_data):
            data = self.train_data[file]
            for i in range(5):
                with open(output_path + file + str(i + 1) + '.txt', 'w', encoding='utf-8') as outfile:
                    for key, sentence in enumerate(data):
                        content, target, sequence = sentence
                        content_words = lac.run(content)

                        temp = int(random.random() * 1000) % len(content_words)
                        count = 0
                        while content_words[temp] in stopwords and count <= 10:
                            temp = int(random.random() * 1000) % len(content_words)
                            count += 1
                        else:
                            result_temp = ''
                            for key1, word in enumerate(content_words):
                                if key1 == temp:
                                    result_temp += fasttext.most_similar(content_words[temp], topn=5)[0][0]
                                else:
                                    result_temp += word
                        outfile.write(str(key) + '\t' + result_temp + '\t' + str(target) + '\t' + str(sequence) + '\n')


if __name__ == '__main__':
    data = data_Augmentation('../data/old_data/real_triple.txt', train_data='../data/dataset/dataset/train/',
                             relation_aggregation_file='../data/old_data/relation_aggregation1.txt',
                             fasttext_path='../model/fasttext/FastText_Vector_100',
                             customization_list=['../data/new_data/NER/dictionary.txt',
                                                 '../data/new_data/NER/EIdictionary.txt'])
    # data.augment_data(output_path='../data/dataset_augment2/train/')
    # data.back_translation(output_path='../data/dataset_back_translation/train/')
    # data.augment_data(output_path='../data/dataset/dataset/dataset_augment/train/')
    # data.random_replace(output_path='../data/dataset/dataset_augment_synonym1/train/',
    #                    stopwords_path='../data/file/stopwords.txt')
    data.augment_data(output_path='../data/dataset/dataset_augment_synonym2/train/')
