import time
import os
import tokens
import config
import numpy as np

# 'ai/data/data/chriso/gauged/test.py'

def fetch(pyfile):
    ret = ''
    
    with open(pyfile, 'r') as file:
        for i in file:
            ret += i
    
    return [ret]
 
def tokenize_keyword(data, keyword):
    ret = []
    l = len(keyword)
    
    for s in data:
        ss = ''
        lasti = 0
        for i in range(0, len(s)-l+1):
            if s[i:i+l] == keyword and s not in tokens.token_to_id.keys():
                ret.append(s[lasti:i])
                ret.append(s[i:i+l])
                lasti = i+l
        ret.append(s[lasti:])

    return ret

def tokenize(data):
    for i in tokens.token_to_id.keys():
        data = tokenize_keyword(data, i)
    data = list(filter(lambda x: x != '', data))
    
    return data

def tokens_to_index(data):
    if len(data) == 0:
        return None
    
    ret = []
    for i in range(len(data)):
        try:
            ret.append(tokens.token_to_id[data[i]])
    
        except KeyError:
            return None

    return ret

def index_to_tokens(data):
    ret = []
    
    for i in range(len(data)):
        ret.extend(tokens.id_to_token[data[i]])
    
    return ret

def dataset(data, seq_len):
    x, y = [], []

    for i in range(len(data)-seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    return x, y
    
def main():
    paths = list(open('ai/data/python50k_eval.txt', 'r'))
    x_data, y_data = [], []

    for i in range(len(paths)): 
        data = fetch(f'ai/data/{paths[i]}'[:-1])
        token_data = tokenize(data)
        index_data = tokens_to_index(token_data)
        
        if index_data:
            x, y = dataset(index_data, 16)
            print(len(x_data), len(y_data), i)
            x_data.extend(x)
            y_data.extend(y)
        
        if i == 400:
            np.save(f'{config.DATASET_PATH}x.npy', np.array(x_data, dtype=np.uint8))
            np.save(f'{config.DATASET_PATH}y.npy', np.array(y_data, dtype=np.uint8))
            break

if __name__ == '__main__':
    main()
