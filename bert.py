import logging
import random
from collections import defaultdict

import numpy
import sentencepiece as spm
import torch
from model import *

token_model = '/tmp/poetry.model'
poetry_file = '/tmp/poetry.all.txt'

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)


def get_batch(by_len, sp):
    items = list(by_len.values())
    nums = [len(l) for l in items]
    cumprob = numpy.cumsum(numpy.array(nums) / sum(nums))

    mask_index = sp.PieceToId('<mask>')
    cls_index = sp.PieceToId('<cls>')

    def get_batch0(batch_size):
        prob = random.uniform(0, 1)
        idx = numpy.searchsorted(cumprob, prob)
        bl = items[idx]
        if len(bl) > batch_size:
            bl = [bl[random.randrange(0, len(bl))] for _ in range(batch_size)]
        xs, ys = [], []
        for xt in bl:
            x, y = [0] * len(xt), [0] * len(xt)
            for i, v in enumerate(xt):
                prob = random.random()
                x[i] = v
                y[i] = 0
                if prob < 0.15:
                    y[i] = v
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        x[i] = mask_index
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        x[i] = random.randrange(5, sp.get_piece_size())
                    # 10% randomly change token to current token
                    else:
                        pass
            xs.append([cls_index] + x)
            ys.append([0] + y)
        return torch.tensor(xs), torch.tensor(ys)

    return get_batch0


def read_ds():
    sp = spm.SentencePieceProcessor(token_model)
    lines = [sp.EncodeAsIds(l.strip()) for l in open(poetry_file, encoding='utf-8')]

    n = int(len(lines) * 0.95)
    train, test = lines[:n], lines[n:]
    logging.info('train lines %d, test lines %d' % (len(train), len(test)))

    train_by_len, test_by_len = defaultdict(list), defaultdict(list)

    for t in train:
        train_by_len[len(t)].append(t)
    for t in test:
        test_by_len[len(t)].append(t)

    # print(len(train_by_len))
    # print({k: len(v) for k, v in train_by_len.items()})

    return Dotdict(train_batch=get_batch(train_by_len, sp), test_batch=get_batch(test_by_len, sp),
                   sp=sp, block_size=max((len(l) for l in lines)))


def ana_tokens():
    sp = spm.SentencePieceProcessor(token_model)
    for i in range(sp.vocab_size()):
        token = sp.IdToPiece(i)
        if i > 255 and len(token) > 1:
            print(id, token)


def build_vocab():
    spm.SentencePieceTrainer.Train(
        '--input=/tmp/poetry.all.txt --model_prefix=/tmp/poetry ' +
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=<pad> ' +
        '--split_by_whitespace=false --model_type=unigram ' +
        '--user_defined_symbols=<sep>,<cls>,<mask> --vocab_size=9000 --byte_fallback=True --character_coverage=0.9995')


def eval_model(model, config, ds):
    with torch.inference_mode():
        model.eval()
        x, y = ds.test_batch(4)
        x, y = x.to(config.device), y.to(config.device)
        logits, _ = model(x)
        model.train()
        print(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0))

    idx = 0
    for y0 in y:
        print(ds.sp.DecodeIds(x[idx].cpu().tolist()), ds.sp.DecodeIds(y[idx].cpu().tolist()), y[idx])
        for i, v in enumerate(y0):
            if v.item() > 0:
                a = logits[idx, i, :]
                valus, indics = torch.topk(a, 6)
                print(idx, i, v.item(), ds.sp.IdToPiece(v.item()), ds.sp.DecodeIds(indics.cpu().tolist()))
                print(valus.float().cpu().numpy(), indics.cpu().numpy())
                print()
        idx += 1


def train_net():
    ds = read_ds()
    config = ModelConfig(bert=True, vocab_size=ds.sp.vocab_size(), block_size=80, n_layer=12,
                         eval_interval=100, batch_size=512, n_embd=512, n_head=16, rotary=False)
    model = Transformer(config)
    logging.info(config)
    logging.info(model)
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for step in range(config.max_steps):
        x, y = ds.train_batch(config.batch_size)
        x, y = x.to(config.device), y.to(config.device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            logging.info(f"step {step}: train loss {loss.item():.4f}")


if __name__ == '__main__':
    # build_vocab()
    # read_ds()
    train_net()
