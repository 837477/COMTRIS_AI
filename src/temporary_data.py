#-*-coding: utf-8-*-
import os
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm


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


def create_temporary_data():
    
    model = Mongo()

    for i in tqdm(range(400000)):
        document = {}

        performance_list = ["low", "mid", "high"]
        performance = np.random.choice(performance_list)

        # CPU
        cpu_dict = {
            "low": ["i3", "r3"],
            "mid": ["i5", "r5"],
            "high": ["i7", "r7"]
        }
        cpu = np.random.choice(cpu_dict[performance])
        
        # VGA
        vga_dict = {
            "low": ["GTX 1050", "GTX 1050TI"],
            "mid": ["GTX 1060", "GTX 1070"],
            "high": ["GTX 1080", "GTX 1080TI"]
        }
        vga = np.random.choice(vga_dict[performance])
        
        # RAM
        ram_dict = {
            "low": ["4GB", "8GB"],
            "mid": ["8GB", "16GB"],
            "high": ["16GB", "32GB"]
        }
        ram = np.random.choice(ram_dict[performance])

        # M/B
        intel_dict = {
            "low": "H410",
            "mid": "B460",
            "high": "Z490"
        }
        amd_dict = {
            "low": "A520",
            "mid": "B550",
            "high": "X570",
        }
        if cpu[0] == 'i':
            mb = intel_dict[performance]
        else:
            mb = amd_dict[performance]

        # SSD
        ssd_dict = {
            "low": "256GB",
            "mid": "512GB",
            "high": "1TB"
        }
        ssd = ssd_dict[performance]

        # POWER
        power_dict = {
            "low": "500W",
            "mid": "600W",
            "high": "700W"
        }
        power = power_dict[performance]

        pc = {
            "CPU": cpu,
            "VGA": vga,
            "RAM": ram,
            "M/B": mb,
            "SSD": ssd,
            "POWER": power,
            "performance": performance
        }

        model.cursor()['temporary_data'].insert_one(pc)


if __name__ == "__main__":
    create_temporary_data()