import config
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self):
        self.x = np.load(f'{config.DATASET_PATH}x.npy')[:-20000]
        self.y = np.load(f'{config.DATASET_PATH}y.npy')[:-20000]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return (self.x[i], self.y[i])

class TestDataset(Dataset):
    def __init__(self):
        self.x = np.load(f'{config.DATASET_PATH}x.npy')[-20000:-10000]
        self.y = np.load(f'{config.DATASET_PATH}y.npy')[-20000:-10000]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return (self.x[i], self.y[i])

class ValDataset(Dataset):
    def __init__(self):
        self.x = np.load(f'{config.DATASET_PATH}x.npy')[-10000:]
        self.y = np.load(f'{config.DATASET_PATH}y.npy')[-10000:]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return (self.x[i], self.y[i])

if __name__ == '__main__':
    trainset = TrainDataset()
    testset = TestDataset()
    valset = ValDataset()
    print(len(trainset))
    print(len(testset))
    print(len(valset))