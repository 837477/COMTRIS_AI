import os
import torch
import torch.nn as nn
import numpy as np
from pymongo import MongoClient


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


class Comtris():
    def __init__(self):
        self.db = MongoClient(os.environ['COMTRIS_MONGODB_URI'])['COMTRIS']
        self.model = {
            "CPU": torch.load("./model/CPU"),
            "VGA": torch.load("./model/VGA"),
            "M/B": torch.load("./model/MB"),
            "RAM": torch.load("./model/RAM"),
            "SSD": torch.load("./model/SSD"),
            "POWER": torch.load("./model/POWER"),
        }
        for part in self.model:
            self.model[part].eval() 
        self.index_dict = {
            "CPU": self.db['master_config'].find_one({"key": "CPU_dict"})['value'],
            "VGA": self.db['master_config'].find_one({"key": "VGA_dict"})['value'],
            "M/B": self.db['master_config'].find_one({"key": "M/B_dict"})['value'],
            "RAM": self.db['master_config'].find_one({"key": "RAM_dict"})['value'],
            "SSD": self.db['master_config'].find_one({"key": "SSD_dict"})['value'],
            "POWER": self.db['master_config'].find_one({"key": "POWER_dict"})['value']
        }
        self.part_needs = self.db['master_config'].find_one({"key": "needs"})['value']
        self.index = {}
        for part in self.index_dict:
            for p_i in self.index_dict[part]["part_to_index"]:
                self.index_dict[part]["part_to_index"][p_i] = int(self.index_dict[part]["part_to_index"][p_i])
            self.index.update(self.index_dict[part]["part_to_index"])
    
    def part(self):
        part = {
            "CPU": list(self.index_dict['CPU']['part_to_index'].keys()),
            "VGA": list(self.index_dict['VGA']['part_to_index'].keys()),
            "M/B": list(self.index_dict['M/B']['part_to_index'].keys()),
            "RAM": list(self.index_dict['RAM']['part_to_index'].keys()),
            "SSD": list(self.index_dict['SSD']['part_to_index'].keys()),
            "POWER": list(self.index_dict['POWER']['part_to_index'].keys())
        }
        return part
    
    def needs(self):
        return self.part_needs

    def prediction(self, parts, target):
        # 예측 데이터 개수 확인
        if len(parts) != len(self.part_needs[target]):
            return False
        
        if target not in {"CPU", "VGA", "M/B", "RAM", "SSD", "POWER"}:
            return False
        
        # 예측 데이터 가공
        x = []
        for part in parts:
            x.append(self.index[part])
        x = torch.FloatTensor(x)
        
        # 예측 값 추출
        y = list(self.model[target](x))
        y = y.index(max(y))
        result = self.index_dict[target]['index_to_part'][str(y)]
        
        return result


if __name__ == "__main__":
    CT = Comtris()

    # 순서 매우 중요!!
    # ["AMD 3100", "ASROCK A320M", "ASROCK RX570", "3200 8G", "500GB", "600W"]
    # [CPU, M/B, VGA, RAM, SSD, POWER]

    needs = CT.needs()
    part = CT.part()
    # CPU TEST
    '''
    for i in range(5):
        x = []
        for p in part:
            if p not in needs['CPU']:
                continue
            x.append(np.random.choice(part[p]))
        result = CT.prediction(x, "CPU")
        print(x)
        print(result)
        print("#" * 100)
    # VGA TEST
    for i in range(5):
        x = []
        for p in part:
            if p not in needs['VGA']:
                continue
            x.append(np.random.choice(part[p]))
        result = CT.prediction(x, "VGA")
        print(x)
        print(result)
        print("#" * 100)
    '''

    result = CT.prediction(["GTX1660SUPER ASUS", "A320 ASUS", "3200 16GB", "1TB", "600W"], "CPU")
    print(result)
