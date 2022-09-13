import json
import os
import random
import re
import time

import fasttext
import pandas as pd
from LAC import LAC
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer


class Config:
    def file_classify(self, directory_path='data/sentence_label/', file_path='sentence_label_1.txt'):
        infile_path = directory_path + file_path
        outfile_T_path = directory_path + 'class1.txt'
        outfile_F_path = directory_path + 'class0.txt'
        print(infile_path)
        infile = open(infile_path, 'r', encoding='utf-8')
        outfile_T = open(outfile_T_path, 'w', encoding='utf-8')
        outfile_F = open(outfile_F_path, 'w', encoding='utf-8')
        for line in infile.readlines():
            if line.isspace():
                pass
            else:
                list1 = line.replace('sentence', ' ').split('access')
                # print(list1)
                content = list1[0].strip()
                # print(list1[1])
                label = int(list1[1].strip())
                if label == 0:
                    outfile_F.write(content + '\n')
                else:
                    outfile_T.write(content + '\n')

        infile.close()
        outfile_T.close()
        outfile_F.close()

    def To_fasttext_format(self, directory_path='data/sentence_label/', ratio=0.3):
        infile1_path = directory_path + 'class0.txt'
        infile2_path = directory_path + 'class1.txt'
        if not os.path.exists(directory_path + 'fasttext/'):
            os.mkdir(directory_path + 'fasttext/')
        outfile1_path = directory_path + 'fasttext/fasttext_train.txt'
        outfile2_path = directory_path + 'fasttext/fasttext_test.txt'
        # load lac
        lac = LAC(mode='seg')

        infile1 = open(infile1_path, 'r', encoding='utf-8')
        infile2 = open(infile2_path, 'r', encoding='utf-8')
        access_dict = {}
        outfile1 = open(outfile1_path, 'w', encoding='utf-8')
        outfile2 = open(outfile2_path, 'w', encoding='utf-8')
        for line in infile1.readlines():
            words = lac.run(line.strip())
            temp = ''
            for word in words:
                temp += word + ' '
            access_dict[temp] = 0
        for line in infile2.readlines():
            words = lac.run(line.strip())
            temp = ''
            for word in words:
                temp += word + ' '
            access_dict[temp] = 1
        infile1.close()
        infile2.close()
        temp_key_ls = list(access_dict.keys())
        random.shuffle(temp_key_ls)
        new_dict = {}
        for key in temp_key_ls:
            new_dict[key] = access_dict[key]
        count = 0
        for key in new_dict:
            if count < len(new_dict) * ratio:
                outfile2.write("__label__" + str(new_dict[key]) + ' ,' + str(key) + '\n')
            else:
                outfile1.write("__label__" + str(new_dict[key]) + ' ,' + str(key) + '\n')
            count += 1
        outfile1.close()
        outfile2.close()

    def To_text_classfication_word(self, directory_path='data/sentence_label/', ratio=[0.7, 0.2, 0.1],
                                   dictinonary='file/Eidictinonary.txt'):
        if not os.path.exists(directory_path + 'text_classification_word/'):
            os.mkdir(directory_path + 'text_classification_word/')
        infile1_path = directory_path + 'class0.txt'
        infile2_path = directory_path + 'class1.txt'
        outfile1_path = directory_path + 'text_classification_word/'

        lac = LAC(mode='seg')
        lac.load_customization(dictinonary)
        infile1 = open(infile1_path, 'r', encoding='utf-8')
        infile2 = open(infile2_path, 'r', encoding='utf-8')
        access_dict = {}
        for line in infile1.readlines():
            words = lac.run(line.replace('，', '').replace('。').strip())
            temp = ''
            for word in words:
                temp += word + ' '
            access_dict[temp] = 0
        for line in infile2.readlines():
            words = lac.run(line.replace('，', '').replace('。').strip())
            temp = ''
            for word in words:
                temp += word + ' '
            access_dict[temp] = 1
        infile1.close()
        infile2.close()
        temp_key_ls = list(access_dict.keys())
        random.shuffle(temp_key_ls)
        new_dict = {}
        for key in temp_key_ls:
            new_dict[key] = access_dict[key]

        outfile1 = open(outfile1_path + 'class.txt', 'w', encoding='utf-8')
        outfile2 = open(outfile1_path + 'train.txt', 'w', encoding='utf-8')
        outfile3 = open(outfile1_path + 'test.txt', 'w', encoding='utf-8')
        outfile4 = open(outfile1_path + 'dev.txt', 'w', encoding='utf-8')
        outfile1.write('False\n')
        outfile1.write('True\n')
        outfile1.close()

        count = 0
        for key in new_dict:
            if count < len(new_dict) * ratio[0]:
                outfile2.write(str(key) + '\t' + str(new_dict[key]) + '\n')
            elif count < len(new_dict) * (ratio[0] + ratio[1]):
                outfile3.write(str(key) + '\t' + str(new_dict[key]) + '\n')
            else:
                outfile4.write(str(key) + '\t' + str(new_dict[key]) + '\n')
            count += 1
        outfile2.close()
        outfile3.close()
        outfile4.close()

    def To_text_classfication(self, directory_path='data/sentence_label/', ratio=[0.7, 0.2, 0.1]):
        if not os.path.exists(directory_path + 'text_classification/'):
            os.mkdir(directory_path + 'text_classification/')
        infile1_path = directory_path + 'class0.txt'
        infile2_path = directory_path + 'class1.txt'
        outfile1_path = directory_path + 'text_classification/'

        infile1 = open(infile1_path, 'r', encoding='utf-8')
        infile2 = open(infile2_path, 'r', encoding='utf-8')
        access_dict = {}
        for line in infile1.readlines():
            access_dict[line.strip()] = 0
        for line in infile2.readlines():
            access_dict[line.strip()] = 1
        infile1.close()
        infile2.close()
        temp_key_ls = list(access_dict.keys())
        random.shuffle(temp_key_ls)
        new_dict = {}
        for key in temp_key_ls:
            new_dict[key] = access_dict[key]

        outfile1 = open(outfile1_path + 'class.txt', 'w', encoding='utf-8')
        outfile2 = open(outfile1_path + 'train.txt', 'w', encoding='utf-8')
        outfile3 = open(outfile1_path + 'test.txt', 'w', encoding='utf-8')
        outfile4 = open(outfile1_path + 'dev.txt', 'w', encoding='utf-8')
        outfile1.write('False\n')
        outfile1.write('True\n')
        outfile1.close()

        count = 0
        for key in new_dict:
            if count < len(new_dict) * ratio[0]:
                outfile2.write(str(key) + '\t' + str(new_dict[key]) + '\n')
            elif count < len(new_dict) * (ratio[0] + ratio[1]):
                outfile3.write(str(key) + '\t' + str(new_dict[key]) + '\n')
            else:
                outfile4.write(str(key) + '\t' + str(new_dict[key]) + '\n')
            count += 1
        outfile2.close()
        outfile3.close()
        outfile4.close()

    def fasttext_train(self, directory_path='data/sentence_label/fasttext/', epoch_times=100, lr=0.01, dim=130,
                       minCount=1, t=0.01,
                       pretrainedVectors=''):
        train_path = directory_path + 'fasttext_train.txt'
        test_path = directory_path + 'fasttext_test.txt'
        model_fasttext = fasttext.train_supervised(train_path, epoch=epoch_times, dim=dim, lr=lr, minCount=minCount,
                                                   t=t, pretrainedVectors=pretrainedVectors)

        result = model_fasttext.test(test_path)
        print("Accuracy : " + str(result))
        model_fasttext.save_model(directory_path + 'fasttext')
        return result

    def check_download(self, directory_path='data/new_data/', file_name='test1.csv'):
        infile_path = directory_path + file_name
        outfile_path = directory_path + 'download_url.txt'
        data = pd.read_csv(infile_path)
        download_nead = data[data['正文'].isnull()]
        list1 = download_nead['标题链接1'].tolist()
        biaotinull = download_nead[download_nead['标题链接1'].isnull()]['标题链接'].tolist()
        list2 = list1 + biaotinull
        # write
        outfile = open(outfile_path, 'w')
        for i in list2:
            try:
                len(i)
                outfile.write(str(i) + '\n')
            except:
                pass
        outfile.close()

    def combine(self, directory_path='data/new_data/', file1_name='test1.csv', file2_name='supplement.csv'):
        infile1 = directory_path + file1_name
        infile2 = directory_path + file2_name
        outfile = directory_path + 'result.csv'
        combine1 = pd.read_csv(infile1, encoding='utf-8')
        combine2 = pd.read_csv(infile2, encoding='utf-8')
        temp1 = combine1.loc[:, ['标题1', '正文', '文本']]
        temp2 = combine2.loc[:, ['标题1', '正文', '文本']]
        temp1['标题1'] = temp1.apply(lambda x: str(x[0]).strip(), axis=1)
        temp2['标题1'] = temp2.apply(lambda x: str(x[0]).strip(), axis=1)
        result = pd.merge(temp1, temp2, how='outer')
        result.dropna(subset=['正文'], inplace=True)
        result.to_csv(outfile)

    def split_by_title(self, directory_path='data/new_data/'):
        def by_title(input, directory_path):
            path1 = directory_path + 'property/'
            path2 = directory_path + 'text/'
            if not os.path.exists(path1):
                os.mkdir(path1)
            if not os.path.exists(path2):
                os.mkdir(path2)
            x = input[1].replace('"', '“')
            infile1_name = path2 + x + '.txt'
            infile2_name = path1 + x + '.txt'
            infile = open(infile1_name, 'w', encoding='utf-8')
            infile.write(input[2])
            infile.close()
            infile = open(infile2_name, 'w', encoding='utf-8')
            infile.write(input[3])
            infile.close()

        result = directory_path + 'result.csv'
        data = pd.read_csv(result, encoding='utf-8')
        infile_path = directory_path + 'by_title/'
        if not os.path.exists(infile_path):
            os.mkdir(infile_path)
        data.apply(lambda x: by_title(x, infile_path), axis=1)
        print('end')

    def modify_by_title(self, directory_path='data/new_data/'):
        modify_path = directory_path + 'by_title_modify/'
        if not os.path.exists(modify_path):
            os.mkdir(modify_path)
        modify_path_text = modify_path + 'text/'
        modify_path_property = modify_path + 'property/'
        if not os.path.exists(modify_path_text):
            os.mkdir(modify_path_text)
        if not os.path.exists(modify_path_property):
            os.mkdir(modify_path_property)
        text_file_path = directory_path + 'by_title/text'
        namelist = os.listdir(text_file_path)
        for filename in namelist:
            infile_path1 = directory_path + 'by_title/text/' + filename
            infile_path2 = directory_path + 'by_title/property/' + filename
            outfile_path1 = modify_path + 'text/' + filename
            outfile_path2 = modify_path + 'property/' + filename
            infile1 = open(infile_path1, 'r', encoding='utf-8')
            outfile1 = open(outfile_path1, 'w', encoding='utf-8')
            for line in infile1.readlines():
                if not line.isspace():
                    linelists = re.split('[。：:；]', line.strip())
                    for i in linelists:
                        outfile1.write(i + '\n')
            outfile1.close()
            infile1.close()
            infile2 = open(infile_path2, 'r', encoding='utf-8')
            outfile2 = open(outfile_path2, 'w', encoding='utf-8')
            for line in infile2.readlines():
                if line.isspace():
                    pass
                else:
                    linelists = re.split(',', line.strip())
                    for i in linelists:
                        outfile2.write(i + '\n')
            outfile2.close()
            infile2.close()
        print('modify_by_title complete')

    def wordcut_process(self, directory_path='data/new_data/', second_level_directory_path='by_title_modify/',
                        dictinonary='file/Eidictinonary.txt'):
        infile_path = directory_path + second_level_directory_path
        infile_text_path = infile_path + 'text/'
        infile_property_path = infile_path + 'property/'
        outfile_path = directory_path + 'wordcut/'
        outfile_text_path = outfile_path + 'text/'
        outfile_property_path = outfile_path + '/property/'
        if not os.path.exists(outfile_path):
            os.mkdir(outfile_path)
        if not os.path.exists(outfile_text_path):
            os.mkdir(outfile_text_path)
        if not os.path.exists(outfile_property_path):
            os.mkdir(outfile_property_path)

        lac = LAC(mode='seg')
        lac.load_customization(dictinonary)
        namelist = os.listdir(infile_text_path)
        for name in namelist:
            infile_text = open(infile_text_path + name, 'r', encoding='utf-8')
            infile_property = open(infile_property_path + name, 'r', encoding='utf-8')
            outfile_text = open(outfile_text_path + name, 'w', encoding='utf-8')
            outfile_property = open(outfile_property_path + name, 'w', encoding='utf-8')
            for line in infile_text.readlines():
                if line.isspace():
                    continue
                words = lac.run(line.strip())
                temp = ''
                for word in words:
                    temp += word + ' '
                outfile_text.write(temp.strip() + '\n')
            for line in infile_property.readlines():
                outfile_property.write(line.strip() + '\n')
            infile_text.close()
            infile_property.close()
            outfile_text.close()
            outfile_property.close()

    def to_json(self, directory_path='data/new_data/by_title_modify/',
                json_file_path='data/new_data/data.json'):
        namelist = os.listdir(directory_path + '/text')
        dict1 = {}
        for name in namelist:
            infile1 = open(directory_path + 'text/' + name, 'r', encoding='utf-8')
            infile2 = open(directory_path + 'property/' + name, 'r', encoding='utf-8')
            temp = {}
            content = []
            description = []
            for line in infile1.readlines():
                if not line.isspace():
                    content.append(line.strip())
            temp['content'] = content
            for line in infile2.readlines():
                if not line.isspace():
                    description.append(line.strip())
            temp['description'] = description
            infile1.close()
            infile2.close()
            dict1[name.replace('.txt', '')] = temp
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(dict1, f, ensure_ascii=False)
        return

    def access(self, directory_path='data/sentence_label/', data_path='data/new_data/wordcut/text/',
               fasttext_path='data/sentence_label/fasttext/fasttext', possibility=0.8):
        namelist = os.listdir(data_path)
        access_path = directory_path + 'access/'
        if not os.path.exists(access_path + 'class0/'):
            os.makedirs(access_path + 'class0/')
        if not os.path.exists(access_path + 'class1'):
            os.makedirs(access_path + 'class1/')
        model = fasttext.load_model(fasttext_path)
        for name in namelist:
            infile1 = open(data_path + name, 'r', encoding='utf-8')
            outfile1 = open(access_path + 'class0/' + name, 'w', encoding='utf-8')
            outfile2 = open(access_path + 'class1/' + name, 'w', encoding='utf-8')
            for line in infile1.readlines():
                pred = model.predict(line.strip())
                label = pred[0][0]
                if label == '__label__1' and pred[1][0] > possibility:
                    outfile2.write(line)
                else:
                    outfile1.write(line)
            infile1.close()
            outfile1.close()
            outfile2.close()

    def key_word_generate(self, source_file_path='data/new_data/data.json', key_word_path='data/new_data/key_word.json',
                          topk=10):
        with open(source_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        lac = LAC(mode='seg')
        print('正在读取文件……')
        tic = time.time()
        filelist = []
        corpus = []
        for file in data:
            filelist.append(file)
            content = data[file]['content']
            temp = ''
            for sentence in content:
                for word in lac.run(sentence):
                    temp += word + ' '
            corpus.append(temp.strip())
        print('读取文件成功，共用时：\t{}s'.format(time.time() - tic))
        transformer = TfidfVectorizer()
        tfidf = transformer.fit_transform(corpus)
        word = transformer.get_feature_names()
        vsm = tfidf.toarray()
        category_keywords_li = []
        for i in range(vsm.shape[0]):
            sorted_keyword = sorted(zip(transformer.get_feature_names(), vsm[i]), key=lambda x: x[1], reverse=True)
            category_keywords = [w[0] for w in sorted_keyword[:topk]]
            category_keywords_li.append(category_keywords)

        key_word = {}
        for key, value in enumerate(category_keywords_li):
            key_word[filelist[key]] = value
        with open(key_word_path, 'w', encoding='utf-8') as outfile:
            json.dump(key_word, outfile, ensure_ascii=False)

    def get_sentence_weight(self, file_path='data/dataset/', key_word_path='data/new_data/key_word.json',
                            fasttext_path='model/fasttext/FastText_Vector_100',
                            sentence_weight_path='data/file/weight.json'):
        with open(key_word_path, 'r', encoding='utf-8') as infile:
            key_word_dict = json.load(infile)

        lac = LAC(mode='seg')
        result_weight = {}
        fasttext = FastText.load(fasttext_path)
        file_type = os.listdir(file_path)
        for type in file_type:
            print('开始处理{}里的文件'.format(type))
            filelist = os.listdir(file_path + type)
            for file in filelist:
                file_name = file.replace('.txt', '')
                file_weight = {}
                with open(file_path + type + '/' + file, 'r', encoding='utf-8') as infile:
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        content, target = set[1], set[2]
                        words = lac.run(content)
                        sum = 0.0
                        for word in words:
                            temp = 0
                            for i in key_word_dict[file_name]:
                                temp = max(temp, fasttext.similarity(word, i))
                            sum += temp
                        file_weight[content] = [target, sum]
                result_weight[file_name] = file_weight
            print('处理完成')

        try:
            with open(sentence_weight_path, 'w', encoding='utf-8') as outfile:
                json.dump(result_weight, outfile, ensure_ascii=False)
            print('写入完成，已经写入到{}中'.format(sentence_weight_path))
        except:
            print('权重数据没有保存成功，请重新确认文件地址是否有误')

    def weight_refine(self, weight_path='data/file/weight.json', file_path='data/dataset/dataset/',
                      new_weight_path='data/file/new_weight.json'):
        with open(weight_path, 'r', encoding='utf-8') as outfile:
            weight = json.load(outfile)
        file_type = os.listdir(file_path)
        for type in file_type:
            print('开始处理{}里的文件......'.format(type))
            filelist = os.listdir(file_path + type)
            for file in filelist:
                file_name = file.replace('.txt', '')
                file_weight = weight[file_name]
                new_file_weight = {}
                content_temp = []
                target_temp = []
                attach_tree = {}
                with open(file_path + type + '/' + file, 'r', encoding='utf-8') as infile:
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        content, target, attach = set[1], int(set[2]), int(set[3])
                        content_temp.append(content)
                        target_temp.append(target)
                        if target == 2 and attach != -1 and target_temp[attach - 1] == 1:
                            if content_temp[attach - 1] not in attach_tree:
                                attach_tree[content_temp[attach - 1]] = [content]
                            else:
                                temp = attach_tree[content_temp[attach - 1]]
                                temp.append(content)
                                attach_tree[content_temp[attach - 1]] = temp
                    # 重新计算权重
                    for content in attach_tree:
                        content_weight = 0
                        for node in attach_tree[content]:
                            try:
                                content_weight += file_weight[node][1]
                            except:
                                print(node)
                        weight[file_name][content][1] = content_weight
            print('{}处理完成'.format(type))

        with open(new_weight_path, 'w', encoding='utf-8') as outfile:
            json.dump(weight, outfile, ensure_ascii=False)

    def weight_refine_up_to_down(self, weight_path='data/file/weight.json', file_path='data/dataset/',
                                 new_weight_path='data/file/new_weight_up_to_down.json'):
        with open(weight_path, 'r', encoding='utf-8') as outfile:
            weight = json.load(outfile)
        file_type = os.listdir(file_path)
        for type in file_type:
            print('开始处理{}里的文件......'.format(type))
            filelist = os.listdir(file_path + type)
            for file in filelist:
                file_name = file.replace('.txt', '')
                file_weight = weight[file_name]
                new_file_weight = {}
                content_temp = []
                target_temp = []
                attach_tree = {}
                with open(file_path + type + '/' + file, 'r', encoding='utf-8') as infile:
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        content, target, attach = set[1], int(set[2]), int(set[3])
                        content_temp.append(content)
                        target_temp.append(target)
                        if target == 2 and attach != -1 and target_temp[attach - 1] == 1:
                            if content_temp[attach - 1] not in attach_tree:
                                attach_tree[content_temp[attach - 1]] = [content]
                            else:
                                temp = attach_tree[content_temp[attach - 1]]
                                temp.append(content)
                                attach_tree[content_temp[attach - 1]] = temp
                    # 重新计算权重
                    node_weight = {}
                    for content in attach_tree:
                        content_weight = weight[file_name][content][1]
                        sum_weight = 0
                        for node in attach_tree[content]:
                            sum_weight += file_weight[node][1]
                        for node in attach_tree[content]:
                            weight[file_name][node][1] = (file_weight[node][1] * content_weight) / sum_weight
            print('{}处理完成'.format(type))

        with open(new_weight_path, 'w', encoding='utf-8') as outfile:
            json.dump(weight, outfile, ensure_ascii=False)
        print('写入完成，已写入到{}'.format(new_weight_path))

    def weight_Normalization(self, weight_path='data/file/weight.json', contain_target=['1', '2'], norm_type='combine',
                             Normalization_path='data/file/weight_Normalization.json'):
        with open(weight_path, 'r', encoding='utf-8') as infile:
            result_weight = json.load(infile)

        def MaxMinNormalization(list1):
            # 去除了偏移量Min,最低不是0
            Max = max(list1)
            Min = min(list1)
            result = []
            for x in list1:
                result.append(((x - Min) / (Max - Min)) + Min)
            return result

        result = {}
        for file in result_weight:
            temp_file = {}
            sentences = result_weight[file]
            temp_content, temp_target, temp_weight = [], [], []
            for i in sentences:
                if sentences[i][0] in contain_target:
                    temp_content.append(i)
                    temp_target.append(sentences[i][0])
                    temp_weight.append(sentences[i][1])
            if len(temp_weight) <= 1:
                print('{}文件中，您划归的标签\t{}数量过少，不需要进行处理'.format(file, contain_target))
            else:
                temp_weight = MaxMinNormalization(temp_weight)
            for i in sentences:
                if i not in temp_content:
                    if norm_type == 'combine':
                        temp_file[i] = sentences[i]
                else:
                    index = temp_content.index(i)
                    temp_file[i] = [temp_target[index], temp_weight[index]]
            result[file] = temp_file

        with open(Normalization_path, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, ensure_ascii=False)
        print('已经修改完成，并保存至{}地址下，除了您指定的标签进行了归一化外，其余权重并没有进行任何修改'.format(Normalization_path))

    def weight_Normalization_by_allfile(self, weight_path='data/file/weight.json', contain_target=['1', '2'],
                                        norm_type='combine',
                                        Normalization_path='data/file/weight_Normalization.json'):
        with open(weight_path, 'r', encoding='utf-8') as infile:
            result_weight = json.load(infile)

        def MaxMinNormalization(list1):
            # 去除了偏移量Min,最低不是0
            Max = max(list1)
            Min = min(list1)
            result = []
            for x in list1:
                result.append(((x - Min) / (Max - Min)) + Min)
            return result

        result = {}
        temp_content, temp_target, temp_weight = [], [], []
        for file in result_weight:
            sentences = result_weight[file]
            for i in sentences:
                if sentences[i][0] in contain_target:
                    temp_content.append(i)
                    temp_target.append(sentences[i][0])
                    temp_weight.append(sentences[i][1])
        temp_weight = MaxMinNormalization(temp_weight)
        for file in result_weight:
            temp_file = {}
            sentences = result_weight[file]
            for i in sentences:
                if i not in temp_content:
                    if norm_type == 'combine':
                        temp_file[i] = sentences[i]
                else:
                    index = temp_content.index(i)
                    temp_file[i] = [temp_target[index], temp_weight[index]]
            result[file] = temp_file
        with open(Normalization_path, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, ensure_ascii=False)
        print('已经修改完成，并保存至{}地址下，除了您指定的标签进行了归一化外，其余权重并没有进行任何修改'.format(Normalization_path))

    def flatten(self, input_file='data/file/weight_Normalization.json',
                output_file='data/file/weight_Normalization_flatten.json'):
        with open(input_file, 'r', encoding='utf-8') as infile:
            weight = json.load(infile)
        result = {}
        for file in weight:
            for content in weight[file]:
                result[content] = weight[file][content]
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, ensure_ascii=False)


if __name__ == '__main__':
    config = Config()
    # config.file_classify('data/sentence_label/', 'sentence_label_1.txt')
    # config.To_fasttext_format('data/sentence_label/')
    # config.fasttext_train('data/sentence_label/fasttext/')
    # config.To_text_classfication('data/sentence_label/')
    # config.check_download('data/new_data/','test1.csv')
    # config.combine('data/new_data/','test1.csv','supplement.csv')
    # config.split_by_title('data/new_data/')
    # config.modify_by_title('data/new_data/')
    # config.wordcut_process('data/new_data/')
    # config.access()
    # config.to_json(directory_path='../data/new_data/by_title_modify/',
    #            json_file_path='../data/new_data/data.json')
    config.key_word_generate(sorce_file_path='../data/new_data/data.json',
                             key_word_path='../data/new_data/key_word.json', topk=10)
