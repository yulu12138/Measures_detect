import csv
import json
import os
import error_report as er

def myargs():
    args = ''
    return args


class my_data():
    def __init__(self, type_title_path='../data/second/real_file.json', dataset_path='../data/dataset/dataset/',
                 type_list_path='../data/new_data/real_triple/type_list.json',
                 describe_file_path='data/new_data/describe.json',
                 weight_file_path='data/file/weight_Normalization_flatten.json',
                 institute_file_path='data/second/new_triple/part/institution_att.txt'):
        self.type_title_path, self.dataset_path, self.type_list_path = type_title_path, dataset_path, type_list_path
        self.describe_file_path, self.weight_file_path = describe_file_path, weight_file_path
        self.institute_file_path = institute_file_path
        # 设置在type list中没有存在的实体的类型
        self.entity_type_default = 'entity'

        # 加载数据
        self.type_title = self.load_data(self.type_title_path)

        # 导入训练集、测试集、验证集的数据
        self.train_data, self.test_data, self.dev_data = self.load_dataset('train'), self.load_dataset(
            'test'), self.load_dataset('dev')

        # 获得描述、发布、联合发文的三元组，包括事件
        self.des_triples = self.describe_generate_triple()
        # 获得机构属性的三元组
        self.institute_triple = self.load_att(self.institute_file_path)
        # 获得联合发文的三元组
        self.synergy_triple = self.load_att(self.synergy_file_path)

        # 获得文章内容的三元组
        self.train_con_triple = self.content_generate_triple(self.train_data, 'train')
        self.test_con_triple = self.content_generate_triple(self.test_data, 'test')
        self.dev_con_triple = self.content_generate_triple(self.dev_data, 'dev')
        # 融合三个结果到新的结果
        self.con_triples = {}
        self.con_triples.update(self.train_con_triple)
        self.con_triples.update(self.test_con_triple)
        self.con_triples.update(self.dev_con_triple)

    def load_data(self,file_path):
        '''
        导入json数据
        :param file_path: json文件的地址
        :return: 导入文件的结果
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                temp_result = json.load(infile)
            return temp_result
        except:
            print('load typy_title failed,please check your path :\t{}'.format(self.type_title_path))

    def load_dataset(self, type='train'):
        '''
        读取数据集中的数据
        :param type: 数据集的类型，分别为train、test、dev
        :return: 返回读取到的数据
        '''
        try:
            file_temp = {}
            for file in os.listdir(self.dataset_path + type + '/'):
                temp = []
                with open(self.dataset_path + type + '/' + file) as infile:
                    for line in infile.readlines():
                        temp.append(line.strip())
                file_temp[file] = temp
            return file_temp
        except:
            print('load dataset {} failed,please check your path :\t{}'.format(type, self.type_title_path))
            return {}

    def describe_generate_triple(self):
        '''
        获得描述数据的三元组数据
        :return: 返回一个字典，key包括release（机构发文）、describe（政策的描述信息三元组）、synergy（机构之间的联合发文）、institute_att（机构的描述三元组）
        '''
        if os.path.exists(self.type_list_path):
            with open(self.type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
        else:
            type_dict = {}
        print('开始处理描述文件，将描述文件转化为三元组格式并进行保存……')
        temp_result = {}

        with open(self.describe_file_path, 'r', encoding='utf-8') as f:
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
                    release_result.append((post, '发布文件', file))
                elif key == '联合发文单位' and data[file][key].strip() != '----':
                    synergy = data[file][key].strip()
                else:
                    result.append((file, key, data[file][key]))
                    type_dict[data[file][key]] = key
            temp = synergy.split(';')
            for i in temp:
                synergy_result.append((post, '联合发文', i))
                type_dict[i] = '机构'
        synergy_result = list(set(synergy_result))

        if er.existJudgment(self.institute_file_path):
            institute_att = []
            with open(self.institute_file_path, 'r', encoding='utf-8') as infile:
                for line in infile.readlines():
                    sets = line.strip().split('\t')
                    institute_att.append((sets[0], sets[1], sets[2]))
            temp_result['institute_att'] = institute_att

        # 写入文件描述文件
        temp_result['release'] = release_result
        temp_result['describe'] = result
        temp_result['synergy'] = synergy_result
        return temp_result

    def content_generate_triple(self, dict1, name='train'):
        print('开始处理内容文件，将内容文件转化为三元组格式并进行保存……')
        result = []
        if os.path.exists(self.type_list_path):
            with open(self.type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
        else:
            type_dict = {}
        for file in dict1:
            file_name = file.strip().replace('.txt', '')
            type_dict[file_name] = '文件'
            file_content, file_target = [], []
            for line in dict1[file]:
                set = line.strip().split('\t')
                content, target, sequence = set[1].strip(), set[2].strip(), set[3].strip()
                # print(file,target,sequence)
                file_content.append(content)
                file_target.append(target)
                if target == '1':
                    result.append((file_name, '概括性措施', content))
                    type_dict[content] = '概括性措施'
                elif target == '2':
                    if sequence == '-1':
                        result.append((file_name, '概括性措施', content))
                        type_dict[content] = '概括性措施'
                    else:
                        if file_target[int(sequence) - 1] != '1':
                            result.append((file_name, '描述性措施', content))
                        else:
                            result.append((file_content[int(sequence) - 1], '描述性措施', content))
                        type_dict[content] = '描述性措施'
        with open(self.type_list_path, 'w', encoding='utf-8') as outfile:
            json.dump(type_dict, outfile, indent=2, ensure_ascii=False)
            print('类型文件保存成功，地址为：\t{}'.format(self.type_list_path))
        # print(result)
        return {name: result}

    def triple_to_neo4j(self, dict1, with_weight=True, output_path='neo4j/'):
        print('开始将数据进行处理，转化为neo4j格式文件')
        triple, property_triple = [], []
        weight, property = {}, {}
        if with_weight and os.path.exists(self.weight_file_path):
            with open(self.weight_file_path, 'r', encoding='utf-8') as outfile:
                weight.update(json.load(outfile))

        entity_dict, entity_list = {}, []
        for type in dict1:
            for i in dict1[type]:
                entity_list.append(i[0])
                entity_list.append(i[2])
        entity_list = list(set(entity_list))
        for key, entity in enumerate(entity_list):
            entity_dict[entity] = key
        # 导入实体类型
        if os.path.exists(self.type_list_path):
            with open(self.type_list_path, 'r', encoding='utf-8') as infile:
                type_dict = json.load(infile)
            print('导入实体类型文件成功')
        else:
            type_dict = {}
            print('不存在类型文件，将自动为实体类型定义为\t{}'.format(self.entity_type_default))

        # 写入实体关系文件
        with open(output_path + 'entity.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(("entity:ID", "name", ":LABEL", "weight"))
            for i in entity_dict:
                if type_dict.get(i) == None:
                    type1 = self.entity_type_default
                else:
                    type1 = type_dict[i]
                temp = None
                if type1 != "文件":
                    if i in weight and with_weight:
                        temp = weight[i][1]
                    csv_writer.writerow((str(entity_dict[i]), i, type1, temp))
            print('实体文件写入成功，文件地址:\t{}'.format(output_path + 'entity.csv'))

        with open(output_path + 'relation.csv', 'w', encoding='utf-8-sig', newline="") as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow((':START_ID', ':END_ID', ':TYPE', 'weight'))
            for i in triple:
                temp = None
                if i[2] in weight and with_weight:
                    temp = weight[i[2]][1]
                csv_writer.writerow((str(entity_dict[i[0]]), str(entity_dict[i[2]]), i[1], temp))
            print('关系文件写入成功，文件地址:\t{}'.format(output_path + 'relation.csv'))
        return

    def load_att(self, file_path):
        '''
        获得所有的属性的文件，输入的file每行是一个三元组，中间用\t分隔，格式为：实体|关系|实体
        :param file_path: 文件地址
        :return: 一个list
        '''
        temp_triple = []
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile.readlines():
                sets = line.strip().split('\t')
                temp_triple.append(sets)
        return temp_triple
