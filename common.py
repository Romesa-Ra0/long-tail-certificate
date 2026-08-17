"""
Shared pieces. Every step imports from here so you never repeat yourself.

The idea in one line: a vision-language model decides by comparing an image
embedding with fixed text embeddings and picking the closest one. We build that
same decision rule in miniature so it trains on a laptop in one minute.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN, D_EMB, N_CLASS = 128, 64, 12
TEMP = 0.07
SEED = 0



class Encoder(nn.Module):
    """This plays the role of the image encoder."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, D_EMB),
        )

    def forward(self, x):
        # normalising means every embedding sits on the unit sphere,
        # so "closeness" is just an angle
        return F.normalize(self.net(x), dim=1)


def build_world(seed=SEED):
    """Prototypes, data generator, and the long-tailed training set."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    # 12 fixed prototypes = the frozen text embeddings. Never trained.
    protos = F.normalize(torch.randn(N_CLASS, D_EMB, generator=g), dim=1)
    A = torch.randn(D_EMB, D_IN, generator=g) / np.sqrt(D_EMB)

    def sample(n_per_class, noise=0.20):
        xs, ys = [], []
        for c, n in enumerate(n_per_class):
            signal = protos[c].unsqueeze(0).repeat(n, 1) @ A
            xs.append(signal + noise * torch.randn(n, D_IN, generator=g))
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    # long tail: 2000 examples for class 0, 12 for class 11
    counts = [max(int(round(2000 * (0.62 ** c))), 12) for c in range(N_CLASS)]
    x_tr, y_tr = sample(counts)
    x_te, y_te = sample([300] * N_CLASS)      # test set is balanced on purpose
    return dict(protos=protos, counts=counts, g=g,
                x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)


def logits(model, x, protos):
    """Cosine similarity to every prototype, scaled by temperature."""
    return model(x) @ protos.t() / TEMP


def train(world, epochs=80, verbose=True):
    model = Encoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y, protos, g = world["x_tr"], world["y_tr"], world["protos"], world["g"]
    for ep in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = F.cross_entropy(logits(model, x[idx], protos), y[idx])
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d}  loss {loss.item():.4f}")
    return model


def get_model(world, seed=SEED, verbose=True):
    """Train once per seed, then reuse the saved weights in every later step."""
    cache = f"model_seed{seed}.pt"
    model = Encoder()
    if os.path.exists(cache):
        model.load_state_dict(torch.load(cache))
        if verbose:
            print(f"loaded trained model from {cache}")
        return model
    if verbose:
        print(f"no cached model for seed {seed}, training now (about one minute)")
    model = train(world, verbose=verbose)
    torch.save(model.state_dict(), cache)
    return model


def accuracy(model, x, y, protos):
    with torch.no_grad():
        pred = logits(model, x, protos).argmax(1)
    per = {c: (pred[y == c] == c).float().mean().item() for c in range(N_CLASS)}
    return (pred == y).float().mean().item(), per


def prune(model, sparsity):
    """Post-training global magnitude pruning: kill the smallest weights."""
    import copy
    m = copy.deepcopy(model)
    if sparsity <= 0:
        return m
    ws = [p for n, p in m.named_parameters() if n.endswith("weight")]
    thr = torch.quantile(torch.cat([w.detach().abs().flatten() for w in ws]), sparsity)
    with torch.no_grad():
        for w in ws:
            w.mul_((w.abs() > thr).float())
    return m
