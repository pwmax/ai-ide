import time
import torch
import torch.nn as nn
import keyboard
import config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from eztorch.trainer import Trainer
from dataset import TrainDataset, TestDataset, ValDataset
from models.transformer import Transformer
from models.grumodel import GruModel

def train():
    val_dataset = ValDataset()
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=True)

    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=True)

    model = GruModel(16, 157, 157)
    trainer = Trainer(model)
    trainer.load_model(f'{config.MODEL_PATH}30mil-0-model.pth')

    def show_loss(train_loss, val_loss):
        if keyboard.is_pressed('3'):
            plt.ylim(0, 10)
            plt.plot(train_loss, "-r", label="train loss")
            plt.plot(val_loss, "-b", label="val loss")
            plt.legend(loc="upper right")
            plt.show()

    def input_transform(x, y):
        return (x.long(), y.long())
    
    def out_transform(out):
        return out
    
    trainer.train(
        loss='crossentropy', 
        device=config.DEVICE, 
        lr=config.LEARNING_RATE, 
        epoch=config.NUM_EPOCH, 
        save_path=config.MODEL_PATH, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_data_fn=show_loss,
        input_data_transform=input_transform,
        out_data_transform=out_transform
    ) 
    
def test():
    device = config.DEVICE
    loss = []
    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=1, 
                            shuffle=False)

    model = GruModel(16, 157, 157).to(device)
    trainer = Trainer(model)
    trainer.load_model(f'{config.MODEL_PATH}0-model.pth')

    for x, y in dataloader:
        x = x.to(device).long()
        y = y.item()
        with torch.no_grad():
            out = torch.argmax(model(x), dim=1).item()
        if out == y:
            loss.append(True)
        else:
            loss.append(False)
        print(loss.count(True)/len(loss))

if __name__ == '__main__':
    train()
    #test()