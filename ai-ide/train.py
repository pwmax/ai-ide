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
from torch.utils.tensorboard import SummaryWriter

def train():
    val_dataset = ValDataset()
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                                shuffle=True)

    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                  shuffle=True)

    model = GruModel(16, 157, 157)
    trainer = Trainer(model)
    #trainer.load_model(f'{config.MODEL_PATH}30mil-0-model.pth')
    writer = SummaryWriter()

    def show_loss(train_loss, val_loss, epoch):
        print(f'epoch {epoch} loss {train_loss[-1]:.9f}')
        writer.add_scalar('train_loss', train_loss[-1], len(train_loss))
        
        if len(val_loss) != 0:
            avrg = 0
            for i in val_loss:
                avrg += i
            avrg /= len(val_loss)
            writer.add_scalar('val_loss', avrg, epoch)
            val_loss.clear()
        
    def input_transform(x, y):
        return (x.long(), y.long())

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
    ) 
    
def test():
    device = config.DEVICE
    loss = []
    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=1, 
                            shuffle=False)

    model = GruModel(16, 157, 157).to(device)
    trainer = Trainer(model)
    trainer.load_model(f'{config.MODEL_PATH}4-model.pth')

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