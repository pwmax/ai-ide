import time
import os
import io
import tokenize as tknz
import config
import numpy as np
from tokens import token_to_id, id_to_token

"""

150k Python Dataset  
https://www.sri.inf.ethz.ch/py150

"""

def fetch(pyfile: str) -> list[str]:
    ret = ''
    try:
        with open(pyfile, 'r') as file:
            for i in file:
                ret += i
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    return [ret]

def extract_keywords(pyfile: str) -> list[str]: 
    keywords = []
    keywords_info = []
    with tknz.open(pyfile) as f:
        all_tokens = tknz.generate_tokens(f.readline)
        for keyword in all_tokens:
            keywords_info.append(keyword)
        for keyword in keywords_info:
            keywords.append(keyword[1])
    return keywords, keywords_info

def tokenize_processing(tokens: list[str]) -> list[str]:
    ret_tokens = [] 
    # Remove comments and ''
    tokens = list(filter(lambda x: x != '', tokens))
    tokens = list(filter(lambda x: x[0] != '#', tokens))
    tokens = list(filter(lambda x: x[0:3] != "'''", tokens))
    tokens = list(filter(lambda x: x[0:3] != '"""', tokens))
    
    # If token not found return []
    for token in tokens:
        if token in token_to_id.keys():
            ret_tokens.append(token)
        else:
            l = [c for c in token if c in token_to_id.keys()]
            if len(l) != len(token):
                return []
            else:
                ret_tokens.extend(l)  
    return ret_tokens

# Tokenize .py file
def tokenize(pyfile: str) -> list[str]:
    data = fetch(pyfile)
    if not data:
        return []
    tokens = []
    keywords, _ = extract_keywords(pyfile)
    data = data[0]
    lasti = 0
    for keyword in keywords:
        i = data.find(keyword)
        s1 = data[:i]
        s2 = data[i:i+len(keyword)]
        tokens.append(s1)
        tokens.append(s2)
        data = data[i+len(keyword):]
    tokens = tokenize_processing(tokens)
    return tokens

# String tokens to integer
def tokens_to_id(tokens: list[str]) -> list[int]:
    id_tokens = []
    for token in tokens:
        id_tokens.append(token_to_id[token])
    return id_tokens

# Integer tokens to string
def id_to_tokens(id_tokens: list[int]) -> list[str]:
    ret_tokens = []
    for i in id_tokens:
        ret_tokens.append(id_to_token[i])
    return ret_tokens

# Split data on x, y (input, target) 
def data_to_xy(data: list[int], seq_len: int) -> [list[int], list[int]] :
    x, y = [], []
    for i in range(len(data)-seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return x, y

# Create numpy training dataset
def create_dataset(dataset_size: int = 1e6, seq_len: int = 32):
    paths = list(open('./ai-ide/data/python100k_train.txt', 'r'))
    x_data, y_data = [], []
    for i in range(len(paths)): 
        tokens = tokenize(f'./ai-ide/data/{paths[i]}'[:-1])
        if tokens:
            tokens = tokens_to_id(tokens)
            x, y = data_to_xy(tokens, seq_len)
            print(len(y_data), i)
            x_data.extend(x)
            y_data.extend(y)
        if len(y_data) >= dataset_size:
            np.save(f'{config.DATASET_PATH}x.npy', np.array(x_data, dtype=np.uint8))
            np.save(f'{config.DATASET_PATH}y.npy', np.array(y_data, dtype=np.uint8))
            break

if __name__ == '__main__':
    #tokens = tokenize('ai-ide/models/grumodel.py')
    #print(tokens)
    #tokens = tokens_to_id(tokens)
    #print(tokens)
    #tokens = id_to_tokens(tokens)
    #print(tokens)
    create_dataset(dataset_size=10000000, seq_len=32)