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
    
class TransformerEncoder(nn.Module):
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
        emb_dim, 
        heads, 
        num_encoder_layers, 
        seq_length, 
        num_tokens, 
        num_classes, 
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, emb_dim).to(device)
        self.pos_emb = nn.Embedding(seq_length, emb_dim).to(device)

        tblocks = []
        for i in range(num_encoder_layers):
            tblocks.append(TransformerEncoder(k=emb_dim, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)
        self.out_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        self._to_device()

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        positions = torch.arange(t, device='cuda')
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        
        x = tokens + positions
        x = self.tblocks(x)
        x = x.view(x.size(0), -1)
        
        x = self.out_block(x)
        return x

    def _to_device(self):
        for p in self.parameters():
            p.data = p.data.to(self.device)
    
if __name__ == '__main__':
    model = Transformer(emb_dim=16, heads=8, num_encoder_layers=6, seq_length=32,
                        num_tokens=157, num_classes=157, device='cuda')
    
    data = torch.zeros(1, 32).long().to('cuda')
    out = model(data)
    print(out.shape)
