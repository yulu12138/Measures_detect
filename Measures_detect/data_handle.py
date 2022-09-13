from config.data_handle import data_handle

dh = data_handle()

# 生成字典
dh.generate_dictionary(inputfile_dir='abandon/process_file/NER/', output_path='abandon/process_file/NER/dictionary.txt')
'''
#生成数据集
dh.to_common_dataset(input_file='data/dataset/', output_file='data/text_classification/')
'''
# 得到文件的描述
dh.get_file_describe(input='data/new_data/data.json', output_path='data/new_data/')

# 描述转化为三元组
dh.describe_to_triple()

# 内容转化为三元组
dh.content_to_triple()

# 转换文件格式，转化为neo4j格式的文件
dh.to_neo4j()

# EI转化为dictionary
dh.EI_to_dictionary()

# 转化为neo4j格式的文件，带有权重
dh.to_neo4j_with_weight(input=['data/new_data/real_triple/result.txt', 'data/new_data/real_triple/release.txt',
                               'data/new_data/real_triple/synergy.txt'], output_path='data/new_data/neo4j_file_1/',
                        weight_list=['data/file/weight_refine_content1_Normalization_flatten.json'])

# 将用于展示的数据转化为gephi的数据，保存至demo中
dh.information_screening(export='data/new_data/handle/export.csv', outpath='data/new_data/handle/demo1/')
