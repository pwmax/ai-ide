import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        self.keys    = nn.Linear(k, k * heads, bias=False)
        self.queries = nn.Linear(k, k * heads, bias=False)
        self.values  = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(heads * k, k)
   
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.queries(x).view(b, t, h, k)
        keys    = self.keys(x)   .view(b, t, h, k)
        values  = self.values(x) .view(b, t, h, k)
       
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))
        
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)
        
        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.fc_block = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        out = self.fc_block(x)
        return self.norm2(out + x)

class Transformer(nn.Module):
    def __init__(
        self, 
        k, 
        heads, 
        depth, 
        seq_length, 
        num_tokens, 
        num_classes, 
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k).to(device)
        self.pos_emb = nn.Embedding(seq_length, k).to(device)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)
        self.out = nn.Linear(k, num_classes)
        self._to_device()

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        positions = torch.arange(t, device='cuda')
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        
        x = tokens + positions
        x = self.tblocks(x)
        x = self.out(x.mean(dim=1))
        return x

    def _to_device(self):
        for p in self.parameters():
            p.data = p.data.to(self.device)
    
if __name__ == '__main__':
    model = Transformer(k=16, heads=8, depth=3, seq_length=16,
                        num_tokens=88, num_classes=88, device='cuda')
    
    data = torch.zeros(32, 16).long().to('cuda')
    out = model(data)
    print(out.shape)
