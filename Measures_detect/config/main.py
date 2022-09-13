import random

import torch
from sklearn.metrics import confusion_matrix, classification_report
from tensorboardX import SummaryWriter
from torch import nn

from config.dataset_loader import dataset
from model.lstm import LSTM

# 设置一些参数
data_dir = 'data/dataset1/dataset/'
cuda = True
optimizer = 'Adam'
learning_rate = 1e-3
# 设置训练网络的一些参数
epoch = 200  # 训练的轮数y
model_save_dir = '../logs/'
max_length = 100
each_step_print = 20
word_or_char = 'word'
'''seed = 1000
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True'''

print('开始装载数据')
# 加载数据
dt = dataset('data/dataset1/dataset/', max_length=max_length, word_or_char=word_or_char)
train_data = dt.to_pack_sequence(dt.train_data)
test_data = dt.to_pack_sequence(dt.test_data)
dev_data = dt.to_pack_sequence(dt.dev_data)
print('装载数据完成')
print('File count')
print('train:\t{}'.format(dt.train_file_count))
print('test:\t{}'.format(dt.test_file_count))
print('dev:\t{}'.format(dt.dev_file_count))
print('Sentence count')
print('train:\t{}'.format(dt.train_sentence_count))
print('test:\t{}'.format(dt.test_sentence_count))
print('dev:\t{}'.format(dt.dev_sentence_count))

# 加载词嵌入
embedding = torch.FloatTensor(dt.word_vector)

# 加载模型
lstm = LSTM(vocab=embedding)
# 判断是否有cuda,是否使用cuda
if torch.cuda.is_available() and cuda:
    lstm = lstm.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available() and cuda:
    loss_fn = loss_fn.cuda()

# 优化器
if optimizer == 'SGD' or optimizer == 'sgd':
    optim = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
elif optimizer == 'Adam' or optimizer == 'adam':
    optim = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
elif optimizer == 'Asgd' or optimizer == 'ASGD':
    optim = torch.optim.ASGD(lstm.parameters(), lr=learning_rate)
elif optimizer == 'Adadelta':
    optim = torch.optim.Adadelta(lstm.parameters(), lr=learning_rate)
elif optimizer == 'Adagrad':
    optim = torch.optim.Adagrad(lstm.parameters(), lr=learning_rate)

# 装载tensorboard
writer = SummaryWriter('/logs/lstm')

total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数

print('开始进行模型训练')
for i in range(epoch):
    print('Epoch:\t{}'.format(i))
    # 训练模型
    lstm.train()
    accuracy = 0.0
    loss_step = 0.0
    sentence_step = 0
    random.shuffle(train_data)
    for data in train_data:
        file, [contents, targets], contents_len = data
        sentence_step += len(targets)
        '''for s1,t1 in zip(contents,targets):
            s1_tensor = torch.tensor([s1])
            t1_tensor = torch.tensor([t1])
            output = lstm(s1_tensor)
            loss = loss_fn(output,t1_tensor)
            optim.zero_grad()
            loss.backward()
            optim.step()'''
        # sentences = torch.tensor(contents)
        # targets = torch.tensor(targets)
        # file = torch.tensor(file)
        if torch.cuda.is_available() and cuda:
            contents = contents.cuda()
            targets = targets.cuda()
            file = file.cuda()
        outputs = lstm(contents, contents_len)
        loss = loss_fn(outputs, targets)
        loss_step += loss
        accuracy += (outputs.argmax(1) == targets).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1
        # 打印accuracy和loss，并写入summarywriter中
        if total_train_step % each_step_print == 0:
            print('Train_step:{},loss:{},accuracy:{}'.format(total_train_step, loss_step, accuracy / sentence_step))
            writer.add_scalar('Train_loss', loss_step, total_train_step)
            writer.add_scalar('Train_accuracy', accuracy / sentence_step, total_train_step)
            accuracy = 0.0
            loss_step = 0.0
            sentence_step = 0.0

    total_test_loss = 0
    total_test_accuracy = 0
    lstm.eval()
    pred = []
    real = []
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    total_batch = 0
    flag = False  # 记录是否很久没有效果提升
    with torch.no_grad():
        random.shuffle(test_data)
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
            outputs = lstm(contents, contents_len)
            test_loss = loss_fn(outputs, targets)
            total_test_loss += test_loss
            if test_loss <= dev_best_loss:
                dev_best_loss = test_loss
                last_improve = total_batch
            total_test_accuracy += (outputs.argmax(1) == targets).sum()
            pred += outputs.argmax(1).numpy().tolist()
        print('Total test loss : {}  ,accuracy : {}'.format(total_test_loss,
                                                            total_test_accuracy / dt.test_sentence_count))
        writer.add_scalar('Test loss', total_test_loss, i)
        writer.add_scalar('Test_accuracy', total_test_accuracy / dt.test_sentence_count, i)
        print('confusion_matrix')
        print(confusion_matrix(y_pred=pred, y_true=real))
        print('classification_report')
        print(classification_report(y_true=real, y_pred=pred))

        if total_batch - last_improve > 1500:
            # 验证集loss超过1000batch没下降，结束训练
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break

    if flag:
        break

torch.save(lstm, model_save_dir + 'lstm.pth')
print('模型训练完成，已保存至{}'.format(model_save_dir + 'lstm.pth'))
