from config.config import Config
from config.data_handle import data_handle

config = Config()
config.weight_refine()
config.weight_Normalization_by_allfile(weight_path='data/file/new_weight.json', contain_target=['1', '2'],
                                       norm_type='split',
                                       Normalization_path='data/file/weight_refine_content1_Normalization.json')
# config.weight_Normalization(weight_path='data/file/new_weight.json', contain_target=['2'], norm_type='split',
#                            Normalization_path='data/file/weight_refine_content2_Normalization.json')
config.flatten(input_file='data/file/weight_refine_content1_Normalization.json',
               output_file='data/file/weight_refine_content1_Normalization_flatten.json')
# config.flatten(input_file='data/file/weight_refine_content2_Normalization.json',
#               output_file='data/file/weight_refine_content2_Normalization_flatten.json')
dh = data_handle()
dh.to_neo4j_with_weight(input=['data/new_data/real_triple/result.txt', 'data/new_data/real_triple/release.txt',
                               'data/new_data/real_triple/synergy.txt'], output_path='data/new_data/neo4j_file_1/',
                        weight_list=['data/file/weight_refine_content1_Normalization_flatten.json'])

dh = data_handle()
dh.information_screening(export='data/new_data/handle/export.csv', outpath='data/new_data/handle/demo1/')

'''
dh = data_handle()
dh.get_dataset_statistical()
'''
'''
aug = data_Augmentation(triple_file='data/old_data/real_triple.txt')
aug.augment_data()
'''

'''
config = Config()
config.weight_refine_up_to_down()
config.weight_Normalization(weight_path='data/file/new_weight_up_to_down.json', contain_target=['1'], norm_type='split',
                            Normalization_path='data/file/weight_refine1_content1_Normalization.json')
config.weight_Normalization(weight_path='data/file/new_weight_up_to_down.json', contain_target=['2'], norm_type='split',
                            Normalization_path='data/file/weight_refine1_content2_Normalization.json')
config.flatten(input_file='data/file/weight_refine1_content1_Normalization.json',
               output_file='data/file/weight_refine1_content1_Normalization_flatten.json')
config.flatten(input_file='data/file/weight_refine1_content2_Normalization.json',
               output_file='data/file/weight_refine1_content2_Normalization_flatten.json')
dh = data_handle()
dh.to_neo4j_with_weight(input=['data/new_data/real_triple/result.txt', 'data/new_data/real_triple/release.txt',
                               'data/new_data/real_triple/synergy.txt'], output_path='data/new_data/neo4j_file_2/',
                        weight_list=['data/file/weight_refine1_content1_Normalization_flatten.json',
                                     'data/file/weight_refine1_content2_Normalization_flatten.json'])
'''
