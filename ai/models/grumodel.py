
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.l = nn.Linear(in_f, out_f, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_f)

    def forward(self, x):
        x = self.l(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)
        return x

class GruModel(nn.Module):
    def __init__(self, emb_dim, num_tokens, num_classes):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, emb_dim)
        self.gru = nn.GRU(128, 128, batch_first=True)
        
        self.block1 = nn.Sequential(
            Block(32, 64),
            Block(64, 128),
            Block(128, 128),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.block1(x)
        x, h = self.gru(x)
        x = x.reshape(x.size(0), -1)
        x = self.block2(x)
        return x

if __name__ == '__main__':
    model = GruModel(emb_dim=32, num_tokens=137, num_classes=137)
    data = torch.zeros(128, 32).long()
    out = model(data)
    print(out.shape)