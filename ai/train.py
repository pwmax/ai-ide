import torch
import torch.nn as nn
import keyboard
import config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from eztorch.trainer import Trainer
from dataset import TrainDataset, TestDataset
from models.transformer import Transformer
from models.grumodel import GruModel

def train():
    device = config.DEVICE
    loss_list = []

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=True)

    #model = Transformer(k=64, heads=8, depth=4, seq_length=16, 
    #                    num_tokens=137, num_classes=137)

    model = GruModel(32, 137, 137).to(device)
    trainer = Trainer(model)
    trainer.load_model(f'{config.MODEL_PATH}model.pth')

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoc in range(config.NUM_EPOCH):
        for x, y in dataloader:
            x = x.to(device).long()
            y = y.to(device).long()
            loss = trainer.update(x, y, criterion, optim)
            loss_list.append(loss.item())
            print(f'loss {epoc} loss {loss.item():.7f}')
            
            if keyboard.is_pressed('3'):
                plt.ylim(0, 5)
                plt.plot(loss_list)
                plt.show()
    
    trainer.save_model(f'{config.MODEL_PATH}model.pth')

def test():
    device = config.DEVICE
    loss = []
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=1, 
                            shuffle=False)

    #model = Transformer(k=64, heads=16, depth=8, seq_length=16, 
    #                    num_tokens=137, num_classes=137)
    
    model = GruModel(32, 137, 137).to(device)
    trainer = Trainer(model)
    trainer.load_model(f'{config.MODEL_PATH}model.pth')

    for x, y in dataloader:
        x = x.to(device).long()
        y = y.to(device).long()
        out = torch.argmax(model(x), dim=1).item()
        y = y.item()
        if out == y:
            loss.append(True)
        else:
            loss.append(False)
        print(loss.count(True)/len(loss))

if __name__ == '__main__':
    train()
    #test()