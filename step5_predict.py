"""
STEP 5 - the claim that decides whether the proposal is alive.

Can the label-free margin, measured on the dense model, predict which classes
will survive compression? If yes, you can warn a clinic that a rare-disease
capability has gone without ever having labels for that disease.

This step runs three seeds. One seed is not evidence: with twelve classes a
single correlation moves around a lot, and reporting only the seed that looked
good is the most common quiet dishonesty in applied machine learning.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from scipy.stats import spearmanr
from common import build_world, get_model, accuracy, prune, N_CLASS

SPARSITY, SEEDS = 0.8, [0, 1, 2]
all_m, all_r, all_c, rhos = [], [], [], []

for seed in SEEDS:
    world = build_world(seed)
    model = get_model(world, seed, verbose=False)
    protos, x, y = world["protos"], world["x_te"], world["y_te"]

    with torch.no_grad():
        sims = model(x) @ protos.t(); pred = sims.argmax(1)
        t2 = sims.topk(2, dim=1).values; fps = t2[:, 0] - t2[:, 1]
    margin = [fps[pred == c].mean().item() for c in range(N_CLASS)]

    _, base = accuracy(model, x, y, protos)
    _, after = accuracy(prune(model, SPARSITY), x, y, protos)
    retained = [after[c] / max(base[c], 1e-9) for c in range(N_CLASS)]

    rho, p = spearmanr(margin, retained); rhos.append(rho)
    print(f"seed {seed}:  rho = {rho:.3f}   p = {p:.4f}")
    all_m += margin; all_r += retained; all_c += world["counts"]

rho_all, p_all = spearmanr(all_m, all_r)
print(f"\nmean rho over seeds: {np.mean(rhos):.3f}")
print(f"pooled over all {len(all_m)} class-seed pairs: rho = {rho_all:.3f}, p = {p_all:.5f}")
print("\nWHAT TO CHECK: individual seeds bounce around, the pooled correlation")
print("is the number worth reporting. If the pooled p were large, Objective 1")
print("would be in trouble, and you would want to know now, not in year three.")

plt.figure(figsize=(5.4, 3.6))
plt.scatter(all_m, all_r, c=np.log10(all_c), cmap="viridis", s=38)
plt.colorbar(label="log10 training examples")
plt.xlabel("label-free margin (dense model)")
plt.ylabel(f"fraction retained at {SPARSITY:.0%} sparsity")
plt.tight_layout(); plt.savefig("step5_predict.png", dpi=150)
print("saved step5_predict.png")
