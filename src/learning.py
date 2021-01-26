import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk import FreqDist
from pymongo import MongoClient
from tqdm import tqdm
from datetime import datetime

torch.manual_seed(1)

class Mongo():
    '''MongoDB Database Management'''

    def __init__(self):
        self.db_client = MongoClient(os.environ['COMTRIS_SERVER_MONGODB_URI'])
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



# 데이터 전처리  #################################################################
train_data = list(db.cursor()['gallery'].find({"pass": 1}, {"_id": 0, "performance": 0})) \
           + list(db.cursor()['pc_quote'].find({"pass": 1}, {"_id": 0, "performance": 0})) \
           + list(db.cursor()['review'].find({"pass": 1}, {"_id": 0, "performance": 0}))

train_data = list(db.cursor()['temporary_data'].find())
train_data = list(db.cursor()['pc_quote'].find({"pass": 1, "shop_date": {'$gt': datetime.strptime('2021-01-01', '%Y-%m-%d')}}, {"_id": 0, "performance": 0}))

x_column = ["M/B", "VGA", "SSD", "RAM", "POWER"]
y_column = ["CPU"]

# index 생성기
x_index = {}
for col in x_column:
    part = pd.DataFrame(train_data, columns=[col])
    temp = {}
    index = 1.0
    for part in set(np.hstack(part.values)):
        temp[part] = index
        index += 1
    x_index.update(temp)

y_index = {}
part = pd.DataFrame(train_data, columns=y_column)
index = 0
for name in set(np.hstack(part.values)):
    y_index[name] = index
    index += 1

# 데이터 숫자 매핑
x_table = pd.DataFrame(train_data, columns=x_column)
x_data = []
for tensor in x_table.values:
    x_data.append(list(map(x_index.get, tensor)))

y_table = pd.DataFrame(train_data, columns=y_column)
y_data = []
for tensor in np.hstack(y_table.values):
    y_data.append(y_index[tensor])

# 최종 학습 데이터
x_data = torch.FloatTensor(x_data)
y_data = torch.LongTensor(y_data)

class Data(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(np.array(x_data))
        self.y=torch.from_numpy(np.array(y_data))
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

data_set = Data()
trainloader = DataLoader(dataset=data_set, batch_size=32)
#################################################################################



# 학습 ########################################################################
# 모델 초기화
class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net,self).__init__()
        self.layer_1 = nn.Linear(D_in, D_out*2)
        self.layer_out = nn.Linear(D_out*2, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_out(x) 
        return x

input_dim = len(x_column)         # how many Variables are in the dataset
output_dim= len(y_index)          # number of classes
learning_rate=0.01

model = Net(input_dim, output_dim)
# model = nn.Linear(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print('W:',list(model.parameters())[0].size())
print('b',list(model.parameters())[1].size())

epochs = 10
loss_list=[]
for epoch in tqdm(range(epochs)):
    for x, y in trainloader:
        #clear gradient 
        optimizer.zero_grad()
        prediction = model(x)
        loss=criterion(prediction, y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        
        loss_list.append(loss.data)

    print('epoch {}, loss {}'.format(epoch, loss.item()))
##############################################################################


# 결과 확인 #####################################################################
print("\nResult checking ...")
x_test = torch.FloatTensor(x_data)
y_test = torch.LongTensor(y_data)

answer = 0
fails = []
for i in tqdm(range(2000)):
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

# for fail in fails:
#     print("#" * 100)
#     print("answer: {:d}".format(fail['answer']))
#     print("prediction: {:d}".format(fail['prediction']))
#     print(fail['model'])
print("accuracy: {}% !".format((answer/2000) * 100))
###############################################################################