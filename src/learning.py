import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk import FreqDist
from tqdm import tqdm
from datetime import datetime
from db_connection import Mongo

torch.manual_seed(1)

# DB Manager 호출
db = Mongo()

# 데이터 전처리  #################################################################
db_result = list(db.cursor()['pc_quote'].find({"pass": 1, "shop_date": {'$gt': datetime.strptime('2021-01-01', '%Y-%m-%d')}}))
# train_data = list(db.cursor()['gallery'].find({"pass": 1, "shop_date": {'$gt': datetime.strptime('2021-01-01', '%Y-%m-%d')}}))


train_data = []
for data in db_result:
    for check in ["AMD"]:
        if check in data['CPU']:
            train_data.append(data)
print(len(train_data))

x_column = ["M/B", "VGA", "SSD", "RAM", "POWER"]
y_column = ["CPU"]

# index 생성기
x_index = {}
x_convert = {}
for col in x_column:
    part = pd.DataFrame(train_data, columns=[col])
    temp1 = {}
    temp2 = {}
    index = 1
    for name in set(np.hstack(part.values)):
        temp1[name] = index
        temp2[index] = name
        index += 1
    x_index.update(temp1)
    x_convert[col] = temp2

y_index = {}
y_convert = {}
part = pd.DataFrame(train_data, columns=y_column)
index = 0
for name in set(np.hstack(part.values)):
    y_index[name] = index
    y_convert[index] = name
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
trainloader = DataLoader(dataset=data_set, batch_size=16)
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


epochs = 10000
for epoch in range(epochs):
    for x, y in trainloader:
        #clear gradient 
        optimizer.zero_grad()
        prediction = model(x)
        loss=criterion(prediction, y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()

    if loss.item() < 0.1:
        break

    if epoch % 1000 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
##############################################################################


# 결과 확인 #####################################################################
print("\nResult checking ...")
x_test = torch.FloatTensor(x_data)
y_test = torch.LongTensor(y_data)

answer = 0
for i in tqdm(range(100)):
    test = list(model(x_test[i]))
    prediction = test.index(max(test))
    if y_test[i].item() == prediction:
        answer += 1
print("accuracy: {}% !".format((answer/100) * 100))


for i in range(5):
    test = list(model(x_test[i]))
    prediction = test.index(max(test))

    print("Input: ")
    for idx, name in enumerate(x_column):
        print(x_convert[name][int(x_test[i][idx].item())])
    print("-"* 30)
    print("Answer: {}".format(y_convert[y_test[i].item()]))
    print("Prediction: {}".format(y_convert[prediction]))
    print("#" * 100)

###############################################################################