import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.set_printoptions(sci_mode=False, precision=4)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
# create a mapping from characters to integers
stoi = { ch:i + 1 for i,ch in enumerate(chars) }
stoi['<>'] = 0
itos = { i:ch for ch,i in stoi.items() }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

n_layer = 1
n_embd = 32
learning_rate = 1e-3
block_size = 8
batch_size = 64 # how many independent sequences will we process in parallel?

def get_batch(split, batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        v = self.value(x)
        q = self.value(x)

        wei = q @ v.transpose(-2, -1) * self.head_size ** -0.5 # B, T, C @ B, C, T -> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # B, T, T
        out = wei @ q # B, T, T @ B, T, C -> B, T, C
        return out

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, vocab_size)        
    
    def forward(self, x, targets=None):
        logits = self.token_emb(x)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), targets.view(-1))
            return logits, loss


class TwoGramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.fc = nn.Linear(2 * n_embd, vocab_size)
    
    def forward(self, x, targets=None):
        b, t = x.shape
        z = torch.zeros((b, 1)).long()
        sx = torch.cat((z, x[:, :-1]), 1)
        
        x = torch.cat((self.emb(x), self.emb(sx)), -1)
        logits = self.fc(x)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), targets.view(-1))
            return logits, loss


class TransFormer1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.sa_head = Head(n_embd, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        b, t = x.shape
        tok_emb =self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(0, t))

        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), targets.view(-1))
            return logits, loss

class TransBlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block1(n_embd) for _ in range(n_layer)])

        self.sa_head = Head(n_embd, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        b, t = x.shape
        tok_emb =self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(0, t))

        x = tok_emb + pos_emb
        # x = self.sa_head(x)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), targets.view(-1))
            return logits, loss

class TransFormerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, 4) for _ in range(n_layer)])

        self.sa_head = Head(n_embd, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        b, t = x.shape
        tok_emb =self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(0, t))

        x = tok_emb + pos_emb
        # x = self.sa_head(x)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), targets.view(-1))
            return logits, loss   

class MultiHead(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // num_heads) for _ in range(num_heads)])
        # self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # out = self.proj(out)
        return out

class Block1(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.sa = Head(n_embd, n_embd)
        self.ffw = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.sa = MultiHead(n_embd, num_heads)
        self.ffw = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


@torch.no_grad()
def eval_mode(model, split):
    model.eval()
    xb, yb = get_batch(split, 1024 * 2)
    logits, loss = model(xb, yb)
    model.train()
    return loss.item()

import time

def train_model(model):
    t = time.time()
    opti = torch.optim.AdamW(model.parameters(), learning_rate)
    for step in range(100001):
        xb, yb = get_batch('train', batch_size)
        logits, loss = model(xb, yb)
        opti.zero_grad(set_to_none=True)
        loss.backward()
        opti.step()
        if step % 10000 == 0:
            n = time.time()
            train_time = n - t
            print(f'{step:6d} {train_time:.2f}s loss: {loss.item():.5f}, train: {eval_mode(model, "train"):.5f}, val: {eval_mode(model, "val"):.5f}')
            t = n


if __name__ == '__main__':
    model = 'bigram'
    torch.manual_seed(13)

    if len(sys.argv) == 2:
        model = sys.argv[1]
    
    print('model: ', model)
    
    if model == 'two':
        # TwoGramModel 6402 parameters
        # 30000 loss: 2.31668, train: 2.26643, val: 2.30044
        model = TwoGramModel()
    elif model == 'tf1':
        # TransFormer1Model 7618 parameters
        #  30000 loss: 2.46153, train: 2.43245, val: 2.43977
        model = TransFormer1Model()
    elif model == 'tf':
        # TransFormerModel 13986 parameters
        #  30000 loss: 2.46153, train: 2.43245, val: 2.43977
        model = TransFormerModel()

    elif model == 'block':
        # TransBlockModel 12930 parameters (1 layer)
        # 90000 12.27s loss: 2.00185, train: 2.06866, val: 2.13217
        # TransBlockModel 28866 parameters (4 layer)
        # 100000 33.52s loss: 2.00742, train: 1.91113, val: 1.99278
        # TransBlockModel 50114 parameters (8 layer)
        # 100000 60.19s loss: 1.88215, train: 1.79920, val: 1.95630
        model = TransBlockModel()
    else:
        #  BigramModel 4356 parameters
        #  30000 loss: 2.43584, train: 2.45850, val: 2.48722
        model = BigramModel()
    print(model.__class__.__name__, sum(p.numel() for p in model.parameters()), 'parameters')
    print(model)
    train_model(model)

        
    # print('==============', sys.argv)