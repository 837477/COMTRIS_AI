import os
from pymongo import MongoClient

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