import os

def existJudgment(file):
    if os.path.exists(file):
        return True
    else:
        print('file_path:{} not exist, please check it'.format(file))