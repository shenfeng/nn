import torch
from model import *
import torch.nn as nn
from torch.nn import functional as F
import time
import random

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) + 3

# create a mapping from characters to integers
stoi = {ch: i + 3 for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

config = ModelConfig(vocab_size=vocab_size, batch_size=256, block_size=96, eval_interval=300, eval_iters=10,
                     n_embd=256, n_head=8, learning_rate=9e-4, silu=False)


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


def get_batch2(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    xs, ys = [], []
    for idx in ix:
        x, y = [0] * config.block_size, [0] * config.block_size
        for i, v in enumerate(data[idx:idx + config.block_size]):
            v = v.item()
            prob = random.random()
            x[i] = v
            y[i] = 0

            if prob < 0.15:
                y[i] = v
                prob /= 0.15
                if prob < 0.8:  # 80% randomly change token to mask token
                    x[i] = 1
                elif prob < 0.9:  # 10% randomly change token to random token
                    x[i] = random.randrange(3, vocab_size)

                else:  # 10% randomly change token to current token
                    pass
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, device=config.device), torch.tensor(ys, device=config.device)
    # x = torch.stack([data[i:i + config.block_size] for i in ix])


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def estimate_loss2(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch2(split)
            logits, _ = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_net():
    model = Transformer(config)
    model.to(config.device)
    logging.info(config)
    logging.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_steps):

        for g in optimizer.param_groups:
            if iter < 10 or iter > 500:
                g['lr'] = 0.0003
            else:
                g['lr'] = 0.001

        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_steps - 1:
            losses = estimate_loss(model)
            logging.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def train_bet():
    config.bert = True
    model = Transformer(config)
    model.to(config.device)
    logging.info(config)
    logging.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_steps):
        for g in optimizer.param_groups:
            if iter < 10 or iter > 500:
                g['lr'] = 0.0003
            else:
                g['lr'] = 0.001

        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_steps - 1:
            losses = estimate_loss2(model)
            logging.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        x, y = get_batch2('train')

        # evaluate the loss
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    train_net()
