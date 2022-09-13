import random

import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config.dataset_loader import dataset
from model.mdrc_1 import MDRC1

# 设置一些参数
data_dir = 'data/dataset/dataset_augment_synonym/'
# data_dir = 'data/dataset/dataset_augment/'
# data_dir = 'data/dataset/dataset/'

cuda = True
optimizer = 'Adam'  # 优化器
learning_rate = 1e-2  # 学习率
# 设置训练网络的一些参数
epoch = 200  # 训练的轮数
model_save_dir = 'logs/'  # 存储的位置
max_length = 100  # 最长句子的长度
each_step_print = 40  # 每隔多少个文章打印一次
word_or_char = 'char'  # 以词为输入还是以句子为输入
batch_improve_size = 1700  # 多少个batch不更新loss即退出
dropout = 0.1

seed = 1001  # pytorch种子

writer = SummaryWriter("logs/logs/model")

torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('开始装载数据')
# 加载数据
dt = dataset(data_dir, max_length=max_length, word_or_char=word_or_char)
train_data = dt.to_pack_sequence_string(dt.train_data)
test_data = dt.to_pack_sequence_string(dt.test_data)
dev_data = dt.to_pack_sequence_string(dt.dev_data)
print('装载数据完成')
print('File count')
print('train:\t{}'.format(dt.train_file_count))
print('test:\t{}'.format(dt.test_file_count))
print('dev:\t{}'.format(dt.dev_file_count))
print('Sentence count')
print('train:\t{}'.format(dt.train_sentence_count))
print('test:\t{}'.format(dt.test_sentence_count))
print('dev:\t{}'.format(dt.dev_sentence_count))


# 加载模型
mdrc = MDRC1(dropout=dropout)
# 判断是否有cuda,是否使用cuda
if torch.cuda.is_available() and cuda:
    mdrc = mdrc.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available() and cuda:
    loss_fn = loss_fn.cuda()

# 优化器
if optimizer == 'SGD' or optimizer == 'sgd':
    optim = torch.optim.SGD(mdrc.parameters(), lr=learning_rate)
elif optimizer == 'Adam' or optimizer == 'adam':
    optim = torch.optim.Adam(mdrc.parameters(), lr=learning_rate)
elif optimizer == 'Asgd' or optimizer == 'ASGD':
    optim = torch.optim.ASGD(mdrc.parameters(), lr=learning_rate)
elif optimizer == 'Adadelta':
    optim = torch.optim.Adadelta(mdrc.parameters(), lr=learning_rate)
elif optimizer == 'Adagrad':
    optim = torch.optim.Adagrad(mdrc.parameters(), lr=learning_rate)
else:
    optim = torch.optim.SGD(mdrc.parameters(), lr=learning_rate)
# 装载tensorboard
writer = SummaryWriter('/logs/mdrc')

from transformers import BertTokenizer,BertModel
# 装载bert模型
tokenizer = BertTokenizer.from_pretrained('model/bert')  # 包含上面三个文件的文件夹目录
model = BertModel.from_pretrained('model/bert')

total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数

# random.seed(shuffle_seed)

print('开始进行模型训练')
for i in range(epoch):
    print('Epoch:\t{}'.format(i))
    # 训练模型
    mdrc.train()
    accuracy = 0.0
    loss_step = 0.0
    sentence_step = 0
    random.shuffle(train_data)
    for data in train_data:
        file, [contents, targets], contents_len = data
        for key, content in enumerate(contents):
            outputs = mdrc(file, content)
            temp_target = torch.tensor([targets[key]])

            loss = loss_fn(outputs, temp_target)
            loss_step += loss
            accuracy += (outputs.argmax(1) == temp_target).sum()
            loss.backward()
            optim.step()
            total_train_step += 1
        sentence_step += len(targets)

        '''# sentences = torch.tensor(contents)
        # targets = torch.tensor(targets)
        # file = torch.tensor(file)
        if torch.cuda.is_available() and cuda:
            contents = contents.cuda()
            targets = targets.cuda()
            file = file.cuda()
        outputs = mdrc(file, contents, contents_len)
        loss = loss_fn(outputs, targets)
        loss_step += loss
        accuracy += (outputs.argmax(1) == targets).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1'''

        # 打印accuracy和loss，并写入summarywriter中
        if total_train_step % each_step_print == 0:
            print('Train_step:{},loss:{},accuracy:{}'.format(total_train_step, loss_step, accuracy / sentence_step))
            writer.add_scalar('Train_loss', loss_step, total_train_step)
            writer.add_scalar('Train_accuracy', accuracy / sentence_step, total_train_step)
            accuracy = 0.0
            loss_step = 0.0
            sentence_step = 0.0

    total_test_loss, total_test_accuracy = 0, 0
    mdrc.eval()
    pred, real = [], []
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    total_batch = 0
    flag = False  # 记录是否很久没有效果提升
    with torch.no_grad():
        for data in test_data:
            file, [contents, targets], contents_len = data
            real += targets
            total_batch += len(targets)
            # sentences = torch.tensor(contents)
            # targets = torch.tensor(targets)
            # file = torch.tensor(file)
            if torch.cuda.is_available() and cuda:
                contents = contents.cuda()
                targets = targets.cuda()
                file = file.cuda()

            for key, content in enumerate(contents):
                outputs = mdrc(file, content)
                temp_target = torch.tensor([targets[key]])
                total_test_loss += loss_fn(outputs, temp_target)
                total_test_accuracy += (outputs.argmax(1) == temp_target).sum()


            '''outputs = mdrc(file, contents, contents_len)
            test_loss = loss_fn(outputs, targets)
            if test_loss < dev_best_loss:
                dev_best_loss = test_loss
                last_improve = total_batch
            total_test_loss += test_loss
            total_test_accuracy += (outputs.argmax(1) == targets).sum()
            pred += outputs.argmax(1).numpy().tolist()'''
        print('Total test loss : {}  ,accuracy : {}'.format(total_test_loss,
                                                            total_test_accuracy / dt.test_sentence_count))
        writer.add_scalar('Test loss', total_test_loss, i)
        writer.add_scalar('Test_accuracy', total_test_accuracy / dt.test_sentence_count, i)
        print('confusion_matrix')
        print(confusion_matrix(y_pred=pred, y_true=real))
        print('classification_report')
        print(classification_report(y_true=real, y_pred=pred, digits=4))

        if total_batch - last_improve > batch_improve_size:
            # 验证集loss超过batch_improve_size没下降，结束训练
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break

    if flag:
        break

torch.save(mdrc, model_save_dir + 'mdrc.pth')
print('模型训练完成，已保存至{}'.format(model_save_dir + 'mdrc.pth'))