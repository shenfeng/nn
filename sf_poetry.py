import logging
import time, sys
from collections import defaultdict
from random import uniform

import numpy

from model import *

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)


def prepare_ds(fn):
    with open(fn, encoding='utf8') as f:
        lines = f.readlines()
        chars = set()

        all = []
        for l in lines:
            l = l.strip()
            chars.update(list(l))
            all.append('^' + l.split('^')[1])
            all.append(l)

    stoi = {s: i for i, s in enumerate(sorted(chars))}
    itos = {i: s for s, i in stoi.items()}

    logging.info(f'no. of poetry {len(lines)}. vocab size {len(stoi)}')

    def encode(txt):
        return [stoi[c] for c in txt]

    def decode(ids):
        return ''.join([itos[i] for i in ids])

    byLen = defaultdict(list)
    for l in all:
        e = encode(l)
        byLen[len(e)].append(e)

    train_data, test_data = [], []

    def to_ds(items):
        X, Y = [], []
        for it in items:
            X.append(it[:-1])
            Y.append(it[1:])
        return torch.tensor(X), torch.tensor(Y)

    for l, items in byLen.items():
        l = len(items)
        if l > 100:
            s = max(int(l * 0.95), l - 16)
            train_data.append(to_ds(items[:s]))
            test_data.append(to_ds(items[s:]))
        elif l > 10:
            train_data.append(to_ds(items))

    nums = [l[0].size(0) for l in train_data]
    cumprob = numpy.cumsum(numpy.array(nums) / sum(nums))

    def get_batch(batch_size):
        prob = uniform(0, 1)
        chX, chY = train_data[numpy.searchsorted(cumprob, prob)]
        if chX.size(0) <= batch_size:
            return chX, chY
        idx = torch.randint(chX.size(0), (batch_size,))
        return chX[idx], chY[idx]

    return Dotdict(trains=train_data, tests=test_data,
                   get_batch=get_batch, stoi=stoi, itos=itos, nums=nums, encode=encode, decode=decode)


@torch.inference_mode()
def loss_in_test(model, tests, config):
    total_loss, n = 0, 0
    model.eval()
    for x, y in tests:
        x = x.to(config.device)
        y = y.to(config.device)
        logits, loss = model(x, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    return total_loss / n, n


def get_config():
    config = ModelConfig(block_size=82, vocab_size=10315, dropout=0.2, batch_size=192, max_steps=1950000)
    if torch.cuda.is_available():
        config.n_embd = 256
        config.batch_size = 512
        config.learning_rate = 6e-4
        config.n_head = 8
        config.n_layer = 12
        config.dropout = 0.10
        config.rotary = True
        config.eval_interval = 1000

    logging.info(config)
    return config


def train_net():
    step, losses = 1, []
    config = get_config()

    model = Transformer(config)
    ds = prepare_ds('poetry.all.txt')
    logging.info('load done, begin train')
    model.to(config.device)
    load_path(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    while True:
        for g in optimizer.param_groups:
            if step < 200:
                g['lr'] = config.learning_rate / 4
            elif step > 20000:
                g['lr'] = config.learning_rate / 2
            else:
                g['lr'] = config.learning_rate
        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = ds.get_batch(config.batch_size)
        batch = [t.to(config.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)
        losses.append(loss.item())
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if config.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % config.eval_interval == 0 or step == config.max_steps - 1 or step == 3:
            tmp = losses[-1000:]
            avg_loss = sum(tmp) / len(tmp)
            eval_loss, _ = loss_in_test(model, ds.tests, config)
            step_ms = (t1 - t0) * 1000
            logging.info(
                f"step {step}, loss {loss.item():.4f}, {step_ms:.2f}ms avg loss: {avg_loss:.4f}, test loss {eval_loss:.4}")

        if step % 10000 == 0:
            out_path = get_save_path()
            torch.save(model.state_dict(), out_path)
            logging.info(f'saving stats to {out_path}, loss {loss.item():.4f}')

        step += 1
        # termination conditions
        if config.max_steps >= 0 and step >= config.max_steps:
            break


if __name__ == '__main__':
    train_net()
