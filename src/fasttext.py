import os
from gensim.models import Word2Vec, KeyedVectors, FastText
from db import Mongo

DB = Mongo()

# 데이터 전처리 #################################################
db_result = list(DB.cursor()['gallery'].find({"pass": 1}))
result = [post['join'] for post in db_result]


#### HyperParameter
vec_size = 6
windows = 6
min_count = 10
iteration = 100
workers = 4

model = FastText(sentences=result, size=vec_size, window=windows, min_count=min_count, iter=iteration, workers=workers)
model_result = model.wv.most_similar("AMD 5800X")
print(model_result)
