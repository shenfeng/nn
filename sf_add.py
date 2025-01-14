# train a transformer to add numbers.
# to see if transformer can actually generalize

import logging
import time, sys
from collections import defaultdict
from random import uniform

import numpy
import torch

try:
    from model import *
except Exception as e:
    pass

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)


def get_config():
    config = ModelConfig(block_size=40, vocab_size=16, dropout=0.2, batch_size=192, max_steps=1950000)
    if torch.cuda.is_available():
        config.n_embd = 256
        config.batch_size = 2048
        config.learning_rate = 9e-4
        config.n_head = 8
        config.n_layer = 4
        config.dropout = 0.1
        config.rotary = True
        config.eval_interval = 500

    logging.info(config)
    return config


import random, string

vocab = string.digits + '*-+=$'
stoi = {s: i + 1 for i, s in enumerate(sorted(vocab))}
itos = {i: s for s, i in stoi.items()}


def encode(txt):
    return [stoi[c] for c in txt]


def decode(ids):
    return ''.join([itos[i] for i in ids])


def test_get_batch():
    x, y = get_batch(10, '-')
    for a, b in zip(x.tolist(), y.tolist()):
        print(decode(a), decode([t for t in b if t > 0]))


def get_batch(batch_size, op, generated={}):
    nums = [9, 99, 999, 9999, 99999, 999999, 9999999]

    if op not in generated:
        by_size = []
        for i in range(40):
            by_size.append([])
        generated[op] = by_size
    by_size = generated[op]

    for i in range(100000):
        a = random.randint(0, random.choice(nums))
        b = random.randint(0, random.choice(nums))

        if a % 17 == 2 and b % 29 == 3: continue

        if op == '+':
            c = a + b
            s = f'{a}+{b}={c}$'
        elif op == '-':
            c = a - b
            s = f'{a}-{b}={c}$'
        else:
            c = a * b
            s = f'{a}*{b}={c}$'

        bucket = by_size[len(s)]
        bucket.append(s)
        if len(bucket) >= batch_size:
            by_size[len(s)] = []
            X, Y = [], []
            for s in bucket[:batch_size]:
                ids = encode(s)
                target = ids[s.index('=') + 1:]

                y = [0] * (len(ids) - len(target) - 1) + target

                X.append(ids[:-1])
                Y.append(y)
            return torch.tensor(X), torch.tensor(Y)


def get_test_batch(batch_size):
    pass


def train_net():
    step, losses = 1, []
    config = get_config()

    model = Transformer(config)
    logging.info('load done, begin train')
    model.to(config.device)
    load_path(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    while True:
        for g in optimizer.param_groups:
            if step < 200:
                g['lr'] = config.learning_rate / 3
            elif step > 30000:
                g['lr'] = config.learning_rate / 2
            else:
                g['lr'] = config.learning_rate
        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = get_batch(config.batch_size)
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
            # eval_loss, _ = loss_in_test(model, ds.tests, config)
            step_ms = (t1 - t0) * 1000
            logging.info(
                f"step {step}, loss {loss.item():.4f}, {step_ms:.2f}ms avg loss: {avg_loss:.4f}")

        if step % 10000 == 0:
            out_path = get_save_path()
            torch.save(model.state_dict(), out_path)
            logging.info(f'saving stats to {out_path}, loss {loss.item():.4f}')

        step += 1
        # termination conditions
        if config.max_steps >= 0 and step >= config.max_steps:
            break


def t():
    config = get_config()
    model = Transformer(config)
    logging.info('load done, begin train')
    model.to(config.device)
    load_path(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for g in optimizer.param_groups:
        g['lr'] = 0.003


import threading
import queue


def train_net2(config, optimizer, model):
    # q = queue.Queue(maxsize=32)

    # total_n = 600
    #
    # def prepare_work():
    #     for step in range(total_n):
    #         op = random.choice(['-', '+', '*'])
    #         batch = get_batch(config.batch_size, op)
    #         q.put((op, batch))
    #
    # threading.Thread(target=prepare_work, daemon=True).start()

    losses = []

    for step in range(600):
        t0 = time.time()
        op = random.choice(['-', '+', '*'])
        batch = get_batch(config.batch_size, op)
        q.task_done()
        # batch = get_batch(config.batch_size, op)
        batch = [t.to(config.device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)
        losses.append(loss.item())
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if config.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()
        if step % 100 == 3:
            tmp = losses[-100:]
            avg_loss = sum(tmp) / len(tmp)
            print(f'{step} step time {(t1 - t0) * 1000:.4f}ms, loss {loss.item()}, avg {avg_loss} {op}')


def run_test():
    config = get_config()

    model = Transformer(config)
    logging.info('load done, begin train')
    model.to(config.device)
    # load_path(model)
    model.load_state_dict(torch.load('/hy-tmp/models/00000.pt'))


def run_test0(model, config):
    ok = 0
    total = 50
    for i in range(total):
        a = random.randint(0, 9999999)
        b = random.randint(0, 9999999)
        s = f'{a}+{b}='
        result = a + b
        exp = torch.tensor([encode(s)]).to(config.device)
        ids = generate(model, exp, 15)
        tt = decode(ids.tolist()[0])
        c = int(tt[tt.index('=') + 1:tt.index('$')])

        print(tt, result, c == result)
        if c == result: ok += 1
    print(ok, ok / total)


if __name__ == '__main__':
    pass
    train_net()
