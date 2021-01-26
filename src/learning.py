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

x_column = ["VGA", "M/B", "RAM", "POWER"]
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
trainloader = DataLoader(dataset=data_set, batch_size=64)
#################################################################################



# 학습 ########################################################################
# 모델 초기화
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)

    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))  
        x = self.linear2(x)
        return x

input_dim = len(x_column)         # how many Variables are in the dataset
hidden_dim = len(y_index)*2       # hidden layers
output_dim= len(y_index)          # number of classes
learning_rate=0.1

model = Net(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print('W:',list(model.parameters())[0].size())
print('b',list(model.parameters())[1].size())

epochs = 2000
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
    if epoch % 1000 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
##############################################################################


# 결과 확인 #####################################################################
print("\nResult checking ...")
x_test = torch.FloatTensor(x_data)
y_test = torch.LongTensor(y_data)

answer = 0
fails = []
for i in tqdm(range(20)):
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

for fail in fails:
    print("#" * 100)
    print("answer: {:d}".format(fail['answer']))
    print("prediction: {:d}".format(fail['prediction']))
    print(fail['model'])
print("accuracy: {}% !".format((answer/20) * 100))
###############################################################################