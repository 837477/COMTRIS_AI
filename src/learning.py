import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime


torch.manual_seed(1)


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x=torch.from_numpy(np.array(x_data))
        self.y=torch.from_numpy(np.array(y_data))
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


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


# 데이터 전처리 ##################################################################
x_column = ["VGA", "M/B", "RAM", "SSD", "POWER"]
y_column = ["CPU"]

db = MongoClient(os.environ['COMTRIS_MONGODB_URI'])['COMTRIS']

date = datetime.strptime('2020-11-01', '%Y-%m-%d')
train_data = list(db['gallery'].find({"pass": 1, "shop_date": {'$gt': date}}))\
           + list(db['pc_qoute'].find({"pass": 1, "shop_date": {'$gt': date}}))\
           + list(db['review'].find({"pass": 1, "shop_date": {'$gt': date}}))
print("학습 데이터 개수: {:d}".format(len(train_data)))


# index 생성기
index_dict = {}
for col in ["CPU", "VGA", "M/B", "RAM", "SSD", "POWER"]:
    part = pd.DataFrame(train_data, columns=[col])
    part_list = list(set(np.hstack(part.values)))
    part_list.sort()

    part_to_index = {}
    index_to_part = {}
    index = 0
    for part_name in part_list:
        part_to_index[part_name] = str(index)
        index_to_part[str(index)] = part_name
        index += 1

    value = {
        "value": {
            "part_to_index": part_to_index,
            "index_to_part": index_to_part
        }
    }
    index_dict[col] = value['value']
    db['master_config'].update_one({"key": col + "_dict"}, {"$set": value}, upsert=True)
index = {}
for part in index_dict:
    for p_i in index_dict[part]["part_to_index"]:
        index_dict[part]["part_to_index"][p_i] = int(index_dict[part]["part_to_index"][p_i])
    index.update(index_dict[part]["part_to_index"])


# 데이터 숫자 매핑
x_table = pd.DataFrame(train_data, columns=x_column)
x_data = []
for tensor in x_table.values:
    x_data.append(list(map(index.get, tensor)))

y_table = pd.DataFrame(train_data, columns=y_column)
y_data = []
for tensor in np.hstack(y_table.values):
    y_data.append(index[tensor])

# 최종 학습 데이터 준비
x_data = torch.FloatTensor(x_data)
y_data = torch.LongTensor(y_data)
data_set = Data(x_data, y_data)
trainloader = DataLoader(dataset=data_set, batch_size=32)


print("학습 데이터 개수: {:d}".format(len(train_data)))
print(x_data)
print(y_data)


# 학습 ########################################################################
# Hyper params
input_dim = len(x_column)
part = pd.DataFrame(train_data, columns=y_column)
output_dim = len(set(np.hstack(part.values)))

learning_rate=0.02

model = Net(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epochs = 2000
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

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

# needs 저장
needs = db['master_config'].find_one({"key": "needs"})
if not needs:
    db['master_config'].insert_one({"key": "needs", "value": {}})
    needs = {}
if y_column[0] not in needs:
    needs[y_column[0]] = x_column
db['master_config'].update_one({"key": "needs"}, {"$set": {"value": needs}})

# 모델 저장
if y_column[0] == "M/B":
    y_column[0] = "MB"
PATH = "./model/" + y_column[0]
torch.save(model, PATH)
