import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from nltk import FreqDist
from pymongo import MongoClient
from tqdm import tqdm

torch.manual_seed(1)

class Mongo():
    '''MongoDB Database Management'''

    def __init__(self):
        self.db_client = MongoClient(os.environ['COMTRIS_MONGODB_URI'])
        self.db_cursor = self.db_client['COMTRIS']

    def client(self):
        '''DB client cursor 반환'''
        return self.db_client
    
    def cursor(self):
        '''RAAS cursor 반환'''
        return self.db_cursor

    def __del__(self):
        self.db_client.close()


# DB Manager 호출
db = Mongo()

# 학습 데이터 로드 ###############################################################
train_list = list(db.cursor()['danawa'].find({}, {"_id": 0, "performance": 0}))

total_data_cnt = len(train_list)
learning_data_cnt = int(len(train_list) * 0.7)
test_data_cnt = int(len(train_list) * 0.3)

print("Total data count: {:d}".format(total_data_cnt))
print("Learning data count: {:d}".format(learning_data_cnt))
print("Test data count: {:d}\n".format(test_data_cnt))
##############################################################################


# 데이터 전처리 #################################################################
train_x_table = pd.DataFrame(train_list, columns=["VGA", "M/B", "RAM", "SSD", "POWER"])
train_y_table = pd.DataFrame(train_list, columns=["CPU"])

# 학습 데이터 전처리
train_x_set = FreqDist(np.hstack(train_x_table.values[:learning_data_cnt]))
train_x_index = {x : idx for idx, x in enumerate(train_x_set)}
x_train = []
for tensor in train_x_table.values[:learning_data_cnt]:
    x_train.append(list(map(train_x_index.get, tensor)))

train_y_set = FreqDist(np.hstack(train_y_table.values[:learning_data_cnt]))
train_y_index = {y : idx for idx, y in enumerate(train_y_set)}
y_train = []
for tensor in np.hstack(train_y_table.values[:learning_data_cnt]):
    y_train.append(train_y_index[tensor])

# 최종 학습 데이터 텐서 생성
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# 데이터 확인
print(train_x_table)
print(train_y_table)
print()
print(x_train)
print(y_train)
print()
print(train_x_index)
print(train_y_index)
##############################################################################


# 학습 ########################################################################
# 모델 초기화
model = nn.Linear(5, len(train_y_index))

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 1
for epoch in tqdm(range(epochs)):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 1000 == 0:
        print('Cost: {:.6f}'.format(cost.item()))
##############################################################################


# 결과 확인 #####################################################################
print("\nResult checking ...")
x_test = []
y_test = []
for tensor in train_x_table.values[learning_data_cnt:]:
    x_test.append(list(map(train_x_index.get, tensor)))
for tensor in np.hstack(train_y_table.values[learning_data_cnt:]):
    y_test.append(train_y_index[tensor])

x_test = torch.FloatTensor(x_test)

answer = 0
fails = []
for i in tqdm(range(test_data_cnt)):
    fail = {}
    test = list(model(x_test[i]))
    prediction = test.index(max(test))
    if y_test[i] == prediction:
        answer += 1
    else:
        fail['answer'] = y_test[i]
        fail['prediction'] = prediction
        fail['model'] = model(x_test[i])
        fails.append(fail)

print("accuracy: {}% !".format((answer/test_data_cnt) * 100))
for fail in fails:
    print("#" * 100)
    print("answer: {:d}".format(fail['answer']))
    print("prediction: {:d}".format(fail['prediction']))
    print(fail['model'])
###############################################################################