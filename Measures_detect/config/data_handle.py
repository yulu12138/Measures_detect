import ast
import csv
import json
import os
import random
import re

import pandas as pd
from LAC import LAC


class data_handle():
    def get_dataset_statistical(self, dataset_path='data/dataset/'):
        count = {}
        print('开始获得数据统计……')
        file_type = os.listdir(dataset_path)
        for type in file_type:
            filelist = os.listdir(dataset_path + type)
            type_count = [0] * 3
            for file in filelist:
                file_name = file.replace('.txt', '')
                with open(dataset_path + type + '/' + file, 'r', encoding='utf-8') as infile:
                    for line in infile.readlines():
                        target = int(line.strip().split('\t')[2])
                        type_count[target] += 1
            count[type] = type_count
        # 总和

        count_sum = [0] * 3
        for type in count:
            for key, value in enumerate(count[type]):
                count_sum[key] += value
        print('综合内容：')
        for key, value in enumerate(count_sum):
            print('\tclass:\t{} , number:\t{}'.format(key, value))
        print('分类内容：')
        for type in count:
            print('type:\t{}'.format(type))
            for key, value in enumerate(count[type]):
                print('\tclass:\t{} , number:\t{}'.format(key, value))

    def generate_dictionary(self, inputfile_dir='data/new_data/NER/',
                            output_path='data/new_data/NER/dictionary.txt'):
        '''
        生成字典
        :param inputfile_dir: 输入文本的地址，标记文本的文件夹
        :param output_path:输出字典的地址
        :return: 返回一个list
        '''

        def get_entity(str1):
            '''
            得到实体序列，将包含在[]中的所有内容进行输出，包括嵌套
            :param str1: string，输入的是一句话
            :return: 回给一个list，包含在[]里面所有的内容
            '''
            stack = 0
            address = []
            result = []
            for key, value in enumerate(str1):
                if value == '[':
                    stack += 1
                    address.append(key + 1)
                elif value == ']' and stack > 0:
                    stack -= 1
                    result.append(str1[address[-1]:key])
                    address.remove(address[-1])
            return result

        def clean(list1):
            '''
            对list进行去重，只保留较少的内容，例如[中国人生[你好]美丽],[你好],只会return一个[你好]
            :param list1:
            :return:
            '''
            list1.sort(reverse=True)
            result = []
            for i in list1:
                flag = 0
                for j in result:
                    if j in i:
                        flag = 1
                        break
                if flag == 0:
                    result.append(i)
            return result

        def get_word(list1):
            '''
            得到标注的词语,例如@中国平安#企业,返回中国平安，企业。
            对得到的结果先进行一个set再转成list返回。
            :param list1:一个list,里面内容均为上诉格式
            :return:一个去重后的list
            '''
            result = []
            for i in list1:
                temp = i.replace('@', '').replace('$', '').replace('*', '')
                templist = re.split('#', temp)
                if '日期' not in templist and '时间' not in templist and '政策文件' not in templist:
                    result += templist
            return list(set(result))

        print('开始生成字典……')
        result = []
        # 找到路径下的所有文件，并逐一打开，调用上述函数对词语进行提取
        filelist = os.listdir(inputfile_dir)
        for file in filelist:
            with open(inputfile_dir + file, 'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    result += get_entity(line.strip())
        result = clean(result)
        result = get_word(result)
        # 写入提取出的词语，将长度大于20的实体扔掉
        try:
            with open(output_path, 'w', encoding='utf-8') as infile:
                for i in result:
                    if len(i) >= 1 and len(i) <= 20:
                        infile.write(i.strip() + '\n')
                print('字典生成成功，文件已保存至:\t{}'.format(output_path))
        except:
            print('路径不存在，请再次确认路径后重试')
        return result

    def to_common_dataset(self, input_file='data/dataset/', output_file='data/text_classification/',
                          ratio=[0.7, 0.9, 1], wordcut=True):
        if wordcut:
            lac = LAC(mode='seg')
            print('本次分词需要加载lac模型进行分词，加载lac模型完成')

        def get_content(input_file_path, wordcut):
            filelist = os.listdir(input_file_path)
            contents = []
            targets = []
            for file in filelist:
                with open(input_file_path + file, 'r', encoding='utf-8') as infile:
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        try:
                            if int(set[2]) <= 2:
                                if wordcut:
                                    temp = ''
                                    for content in lac.run(set[1]):
                                        temp += content + ' '
                                    contents.append(temp.strip())
                                else:
                                    contents.append(set[1])
                                targets.append(set[2])
                        except:
                            pass
            return contents, targets

        print('开始导入数据……')
        contents1, targets1 = get_content(input_file + 'train/', wordcut=wordcut)
        contents2, targets2 = get_content(input_file + 'test/', wordcut=wordcut)
        contents3, targets3 = get_content(input_file + 'dev/', wordcut=wordcut)
        print('导入数据完成')
        contents = contents1 + contents2 + contents3
        targets = targets1 + targets2 + targets3

        con_dict = {}
        for i, j in zip(contents, targets):
            con_dict[i] = j
        print('打乱数据，并对数据进行写入……')
        random.shuffle(contents)

        outfile1 = open(output_file + 'train.txt', 'w', encoding='utf-8')
        outfile2 = open(output_file + 'test.txt', 'w', encoding='utf-8')
        outfile3 = open(output_file + 'dev.txt', 'w', encoding='utf-8')

        length = len(contents)
        for key, value in enumerate(contents):
            if key <= ratio[0] * length:
                outfile1.write(value + '\t' + con_dict[value] + '\n')
            elif key <= ratio[1] * length:
                outfile2.write(value + '\t' + con_dict[value] + '\n')
            else:
                outfile3.write(value + '\t' + con_dict[value] + '\n')
        outfile1.close()
        outfile2.close()
        outfile3.close()
        print('写入完成，已经将文件写入到{}\t、{}\t、{}中'.format(output_file + 'train.txt', output_file + 'test.txt',
                                                  output_file + 'dev.txt'))

    def get_file_describe(self, input='data/new_data/data.json', output_path='data/new_data/',
                          output_file_name='describe.json'):
        '''
        从json文件中导入信息，将key为description中的内容进行格式化处理，转换成字典进行存储，存储至output中。例如：
        "中国人民银行 银保监会 财政部 发展改革委 工业和信息化部关于进一步对中小微企业贷款实施阶段性延期还本付息的通知"：{“content:......
        "description": ["[主题分类] 财政、金融、审计/财政", "[发文机构] 中国人民银行", "[联合发文单位] 中国银行保险监督管理委员会;财政部;
        国家发展和改革委员会;工业和信息化部", "[实施日期] ----", "[成文日期] 2020-06-01", "[发文字号] 银发〔2020〕122号",
        "[废止日期] ----", "[发布日期] 2020-06-01", "[有效性] 有效", "[文件来源] 政府公报 年 第期(总第期)"]}
        转换成：
        "中国人民银行 银保监会 财政部 发展改革委 工业和信息化部关于进一步对中小微企业贷款实施阶段性延期还本付息的通知"：{
        "主题分类": "财政、金融、审计/财政" ,"发文机构":"中国人民银行","联合发文单位":"中国银行保险监督管理委员会;
        财政部,国家发展和改革委员会;工业和信息化部"......}

        :param input:data地址
        :param output_path:输出文件地址
        :param output_file_name:输出文件名
        '''
        print('开始进行格式转换……')
        with open(input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for key in data:
            temp = {}
            str1 = data[key]['description']
            for line in str1:
                target, content = line.strip().split(']')
                target = target.replace('[', '').strip()
                temp[target] = content.strip()
            result[key] = temp
        with open(output_path + output_file_name, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=2, ensure_ascii=False)

        print("转换完毕，转换后文件已保存至：\t{}".format(output_path + output_file_name))

    def describe_to_triple(self, input='data/new_data/describe.json',
                           outfile_path='data/new_data/real_triple/',
                           type_list_path='data/new_data/real_triple/type_list.json'):
        if os.path.exists(type_list_path):
            with open(type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
        else:
            type_dict = {}
        print('开始处理描述文件，将描述文件转化为三元组格式并进行保存……')
        with open(input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = []
        synergy_result = []
        release_result = []
        for file in data:
            type_dict[file] = '文件'
            for key in data[file]:
                if key == '发文机构' and data[file][key].strip() != '----':
                    post = data[file][key].strip()
                    type_dict[data[file][key]] = '机构'
                    release_result.append("('" + post + "','发布文件','" + file + "')\n")
                elif key == '联合发文单位' and data[file][key].strip() != '----':
                    synergy = data[file][key].strip()
                else:
                    result.append([file, key, data[file][key]])
                    type_dict[data[file][key]] = key
            temp = synergy.split(';')
            for i in temp:
                synergy_result.append(post + '联合发文' + i)
                type_dict[i] = '机构'
        synergy_result = list(set(synergy_result))
        # 写入文件描述文件
        with open(outfile_path + 'describe.txt', 'w', encoding='utf-8') as outfile:
            for i in result:
                outfile.write("('" + i[0] + "','" + i[1] + "','" + i[2] + "')\n")
            print('描述文件写入成功，文件地址为：\t{}'.format(outfile_path + 'describe.txt'))
        # 写入实体类型文件
        with open(type_list_path, 'w', encoding='utf-8') as outfile:
            json.dump(type_dict, outfile, indent=2, ensure_ascii=False)
            print('实体类型文件写入成功，文件地址为：\t{}'.format(type_list_path))
        # 写入联合发文三元组文件
        with open(outfile_path + 'synergy.txt', 'w', encoding='utf-8') as outfile:
            for i in synergy_result:
                t1, t2 = i.split('联合发文')
                outfile.write("('" + t1 + "','联合发文','" + t2 + "')\n")
            print('联合发文三元组文件写入成功，文件地址为：\t{}'.format(outfile_path + 'synergy.txt'))
        # 写入发布三元组文件
        with open(outfile_path + 'release.txt', 'w', encoding='utf-8') as outfile:
            for i in release_result:
                outfile.write(i)
            print('发布三元组文件写入成功，文件地址为：\t{}'.format(outfile_path + 'release.txt'))
        return

    def content_to_triple(self, input='data/dataset/dataset/', output_path='data/new_data/real_triple/result.txt',
                          type_list_path='data/new_data/real_triple/type_list.json'):
        print('开始处理内容文件，将内容文件转化为三元组格式并进行保存……')
        filelist = os.listdir(input)
        result = []
        if os.path.exists(type_list_path):
            with open(type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
        else:
            type_dict = {}
        for type in filelist:
            list1 = os.listdir(input + type)
            for file in list1:
                file_name = file.strip().replace('.txt', '')
                type_dict[file_name] = '文件'
                file_content, file_target = [], []
                with open(input + type + '/' + file, 'r', encoding='utf-8') as infile:
                    for line in infile.readlines():
                        set = line.strip().split('\t')
                        content, target, sequence = set[1].strip(), set[2].strip(), set[3].strip()
                        # print(file,target,sequence)
                        file_content.append(content)
                        file_target.append(target)
                        if target == '1':
                            result.append([file_name, '概括性措施', content])
                            type_dict[content] = '概括性措施'
                        elif target == '2':
                            if sequence == '-1':
                                result.append([file_name, '概括性措施', content])
                                type_dict[content] = '概括性措施'
                            else:
                                if file_target[int(sequence) - 1] != '1':
                                    result.append([file_name, '描述性措施', content])
                                else:
                                    result.append([file_content[int(sequence) - 1], '描述性措施', content])
                                type_dict[content] = '描述性措施'
        with open(type_list_path, 'w', encoding='utf-8') as outfile:
            json.dump(type_dict, outfile, indent=2, ensure_ascii=False)
            print('类型文件保存成功，地址为：\t{}'.format(type_list_path))
        # print(result)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i in result:
                outfile.write("('" + i[0] + "','" + i[1] + "','" + i[2] + "')\n")
            print('内容转化三元组文件保存成功，地址为：\t{}'.format(output_path))

    def to_neo4j(self, input=['data/new_data/real_triple/result.txt'],
                 type_list_path='data/new_data/real_triple/type_list.json', entity_type_default='entity',
                 output_path='data/new_data/neo4j_file/'):
        print('开始将地址:\t{}\t数据进行处理，转化为neo4j格式文件'.format(input))
        triple = []
        for file in input:
            with open(file, 'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    triple.append(ast.literal_eval(line.strip()))
        # 建立实体关系字典
        entity_dict = {}
        entity_list = []
        for i in triple:
            entity_list.append(i[0])
            entity_list.append(i[2])
        entity_list = list(set(entity_list))
        for key, entity in enumerate(entity_list):
            entity_dict[entity] = key
        # 导入实体类型
        if os.path.exists(type_list_path):
            with open(type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
        else:
            type_dict = {}
            print('不存在类型文件，将自动为实体类型定义为\t{}'.format(entity_type_default))

        # 写入实体关系文件
        with open(output_path + 'entity.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(("entity:ID", "name", ":LABEL"))
            for i in entity_dict:
                if type_dict.get(i) == None:
                    type1 = entity_type_default
                else:
                    type1 = type_dict[i]
                csv_writer.writerow((str(entity_dict[i]), i, type1))
            print('实体文件写入成功，文件地址:\t{}'.format(output_path + 'entity.csv'))
        with open(output_path + 'relation.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow((':START_ID', ':END_ID', ':TYPE'))
            for i in triple:
                csv_writer.writerow((str(entity_dict[i[0]]), str(entity_dict[i[2]]), i[1]))
            print('关系文件写入成功，文件地址:\t{}'.format(output_path + 'relation.csv'))
        return

    def to_neo4j_with_weight(self, input=['data/new_data/real_triple/result.txt'],
                             property_file_list=['data/new_data/real_triple/describe.txt'],
                             type_list_path='data/new_data/real_triple/type_list.json', entity_type_default='entity',
                             output_path='data/new_data/neo4j_file_1/',
                             weight_list=['data/file/weight_Normalization_flatten.json']):
        print('开始将地址:\t{}\t数据进行处理，转化为neo4j格式文件'.format(input))
        triple, property_triple = [], []
        weight, property = {}, {}
        for weight_path in weight_list:
            with open(weight_path, 'r', encoding='utf-8') as outfile:
                weight.update(json.load(outfile))
        for property_file in property_file_list:
            with open(property_file, 'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    dt = ast.literal_eval(line.strip())
                    property_triple.append(dt)
        for i in property_triple:
            property[i[0]] = {}
        for i in property_triple:
            property[i[0]][i[1]] = i[2]
        print(property)

        for file in input:
            with open(file, 'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    triple.append(ast.literal_eval(line.strip()))
        # 建立实体关系字典
        entity_dict = {}
        entity_list = []
        for i in triple:
            entity_list.append(i[0])
            entity_list.append(i[2])
        entity_list = list(set(entity_list))
        for key, entity in enumerate(entity_list):
            entity_dict[entity] = key
        # 导入实体类型
        if os.path.exists(type_list_path):
            with open(type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
            print('导入实体类型文件成功')
        else:
            type_dict = {}
            print('不存在类型文件，将自动为实体类型定义为\t{}'.format(entity_type_default))

        # 写入实体关系文件
        with open(output_path + 'entity.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(("entity:ID", "name", ":LABEL", "weight"))
            for i in entity_dict:
                if type_dict.get(i) == None:
                    type1 = entity_type_default
                else:
                    type1 = type_dict[i]
                temp = None
                if type1 != "文件":
                    if i in weight:
                        temp = weight[i][1]
                    csv_writer.writerow((str(entity_dict[i]), i, type1, temp))
            print('实体文件写入成功，文件地址:\t{}'.format(output_path + 'entity.csv'))

        with open(output_path + 'file_entity.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(
                ("entity:ID", "name", ":LABEL", "主题分类", "实施日期", "成文日期", "发文字号", "废止日期", "发布日期", "有效性", "文件来源"))
            for i in entity_dict:
                type1 = None
                if type_dict.get(i) == None:
                    type1 = entity_type_default
                else:
                    type1 = type_dict[i]
                temp = None
                if type1 == "文件":
                    property_list = [None] * 8
                    if i in property:
                        if "主题分类" in property[i]:
                            property_list[0] = property[i]["主题分类"]
                        if "实施日期" in property[i]:
                            property_list[1] = property[i]["实施日期"]
                        if "成文日期" in property[i]:
                            property_list[2] = property[i]["成文日期"]
                        if "发文字号" in property[i]:
                            property_list[3] = property[i]["发文字号"]
                        if "废止日期" in property[i]:
                            property_list[4] = property[i]["废止日期"]
                        if "发布日期" in property[i]:
                            property_list[5] = property[i]["发布日期"]
                        if "有效性" in property[i]:
                            property_list[6] = property[i]["有效性"]
                        if "文件来源" in property[i]:
                            property_list[7] = property[i]["文件来源"]
                    csv_writer.writerow(
                        (str(entity_dict[i]), i, type1, property_list[0], property_list[1], property_list[2],
                         property_list[3], property_list[4], property_list[5], property_list[6], property_list[7]))
            print('文件实体文件写入成功，文件地址:\t{}'.format(output_path + 'file_entity.csv'))

        with open(output_path + 'relation.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow((':START_ID', ':END_ID', ':TYPE', 'weight'))
            for i in triple:
                temp = None
                if i[2] in weight:
                    temp = weight[i[2]][1]
                csv_writer.writerow((str(entity_dict[i[0]]), str(entity_dict[i[2]]), i[1], temp))
            print('关系文件写入成功，文件地址:\t{}'.format(output_path + 'relation.csv'))
        return

    def EI_to_dictionary(self, input_file='data/new_data/real_triple/real_triple.txt',
                         output_file='data/new_data/NER/EIdictionary.txt'):
        print('三元组文件转化为实体关系字典……')
        entity = []
        relation = []
        triple = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile.readlines():
                temp = ast.literal_eval(line.strip())
                triple.append(temp)
                entity.append(temp[0])
                entity.append(temp[2])
                relation.append(temp[1])
        entity = list(set(entity))
        relation = list(set(relation))
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i in entity:
                outfile.write(i + '\n')
            for i in relation:
                outfile.write(i + '\n')
            print('实体关系字典生成成功，文件已保存至：\t{}'.format(output_file))
        return

    def information_screening(self, export='data/new_data/handle/export.csv', filepath='data/new_data/neo4j_file_1/',
                              entitylist=['entity.csv', 'file_entity.csv'], outpath='data/new_data/handle/demo1/',
                              relation='relation.csv', label_refine=True):
        print('生成用例demo，demo的输入文件地址为：\t{}'.format(export))
        dataframe = pd.read_csv(export)
        namelist = []
        for i in dataframe.iterrows():
            for j in i[1]:
                te = j
                te = te.replace('{', '').replace('}', '').split(',')
                for k in te:
                    if 'entity' in k:
                        namelist.append(int(k.replace('"entity":', '')))
        namelist = list(set(namelist))
        print(namelist)
        # 实体
        for file in entitylist:
            data = pd.read_csv(filepath + file)
            new_data = pd.DataFrame()
            for entity in namelist:
                d1 = data[data['entity:ID'] == int(entity)]
                new_data = pd.concat([new_data, d1])
            new_data.index = range(len(new_data))
            new_data.insert(2, 'name_brief', new_data['name'])
            new_data['name_brief'] = new_data['name_brief'].apply(lambda x: x[:5] + '...')
            new_data = new_data.rename(columns={'entity:ID': 'ID', ':LABEL': 'LABEL'})
            if 'weight' in new_data.columns:
                new_data['weight'] = new_data['weight'].apply(lambda x: format(x, '.4f'))
            new_data.to_csv(outpath + file, index=False)
            print('{}生成成功'.format(file))

        # 关系
        re = pd.read_csv(filepath + relation)
        new = re.loc[re[':START_ID'].isin(namelist) & re[':END_ID'].isin(namelist)]
        new.index = range(len(new))

        new = new.rename(columns={':START_ID': 'Source', ':END_ID': 'Target', ':TYPE': 'Label'})
        new['weight'] = new['weight'].apply(lambda x: format(x, '.4f'))
        if label_refine:
            new['Label'] = new['Label'] + '-' + new['weight'].astype(str)
        new.to_csv(outpath + relation, index=False)
        print('demo生成成功，地址为:\t{}'.format(outpath))


if __name__ == '__main__':
    dh = data_handle()
    # dh.generate_dictionary()
    # dh.to_common_dataset(input_file='../data/dataset/', output_file='../data/text_classification/')
    # dh.get_file_describe(input='../data/new_data/data.json',output_path='../data/new_data/')
    # dh.describe_to_triple(input='../data/new_data/describe.json',
    #                      outfile_path='../data/new_data/real_triple/',
    #                      type_list_path='../data/new_data/real_triple/type_list.json')
    '''dh.content_to_triple(input='../data/dataset/', output_path='../data/new_data/real_triple/result.txt',
                         type_list_path='../data/new_data/real_triple/type_list.json')'''
    # dh.EI_to_dictionary(input_file='../data/new_data/real_triple/real_triple.txt',
    #                     output_file='../data/new_data/NER/EIdictionary.txt')
    dh.to_neo4j(input=['../data/new_data/real_triple/synergy.txt', '../data/new_data/real_triple/result.txt',
                       '../data/new_data/real_triple/release.txt'],
                output_path='../data/new_data/neo4j_file/',
                type_list_path='../data/new_data/real_triple/type_list.json')
