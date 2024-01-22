import logging
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@dataclass
class ModelConfig:
    block_size: int = 16  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    batch_size: int = 16  # how many independent sequences will we process in parallel?
    eval_interval: int = 100
    # dtype: torch.dtype = torch.float32
    dtype: torch.dtype = torch.bfloat16
    eval_iters: int = 5
    max_steps: int = 995000
    n_layer: int = 8
    n_embd: int = 64
    n_head: int = 4
    bias: bool = False
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate: float = 1e-3
    rotary: bool = True


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    # assert 0 <= 1 < ndim
    # print(ndim, freqs_cis.shape, x.shape)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LayerNorm0(torch.nn.Module):
    # def __init__(self, dim: int, eps: float = 1e-6):
    def __init__(self, ndim, bias, config):
        super().__init__()
        # self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim, dtype=torch.bfloat16))

    def _norm(self, x):
        return

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        # output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, dtype=config.dtype))
        self.bias = nn.Parameter(torch.zeros(ndim, dtype=config.dtype)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=config.dtype)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                           dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, dtype=config.dtype)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, config=config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, config=config)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd, dtype=config.dtype)

        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, dtype=config.dtype)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=config.dtype)

        if config.rotary:
            self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size).to(config.device)
            logging.info('using rotary position embedding')
        else:
            self.freqs_cis = None
            self.wpe = nn.Embedding(config.block_size, config.n_embd, dtype=config.dtype)

        # for n, p in self.named_parameters():
        #     print(n, p.shape, p.requires_grad)
        # print(self.freqs_cis.requires_grad)

        n_params = sum(p.numel() for p in self.parameters())
        device = torch.cuda.get_device_name() if torch.cuda.is_available() else ''
        logging.info("number of parameters: %.4fM %s" % (n_params / 1e6, device))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.freqs_cis is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
            pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
            _bsz, seqlen = idx.shape
            freqs_cis = self.freqs_cis[:seqlen]

        for block in self.h:
            x = block(x, freqs_cis)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


@torch.inference_mode()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, block_size=82):
    model.eval()
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx


save_path = 'models'


def get_save_path():
    try:
        os.mkdir(save_path)
    except Exception as e:
        pass

    for i in range(1000):
        out_path = 'models/%05d.pt' % i
        if not os.path.exists(out_path):
            return out_path


def load_path(model):
    if os.path.exists(save_path):
        p = sorted(os.listdir(save_path))[-1]
        model.load_state_dict(torch.load('%s/%s' % (save_path, p)))
        logging.info(f"loading parameters from {p}")
    else:
        logging.warning("error loading already trained")
