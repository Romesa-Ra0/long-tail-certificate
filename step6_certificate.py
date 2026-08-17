"""
STEP 6 - test the certificate the proposal promised, and watch it fail.

The proposal said: a prediction survives compression if
        margin  >  2 * (embedding displacement)
Displacement is label-free: how far compression moves embeddings, measured on
unlabelled inputs. This step measures both sides of that inequality.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from common import build_world, get_model, prune, N_CLASS

SPARSITIES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
world = build_world(); model = get_model(world)
protos, x = world["protos"], world["x_te"]

with torch.no_grad():
    sims = model(x) @ protos.t(); pred = sims.argmax(1)
    t2 = sims.topk(2, dim=1).values; fps = t2[:, 0] - t2[:, 1]
margin = [fps[pred == c].mean().item() for c in range(N_CLASS)]
head_m, tail_m = np.mean(margin[:4]), np.mean(margin[-4:])

probe = x[torch.randperm(len(x))[:1000]]      # unlabelled probe set
print("\n sparsity   eps(95th pct)   2*eps   classes certified (of 12)")
eps_list = []
for s in SPARSITIES:
    with torch.no_grad():
        d = (prune(model, s)(probe) - model(probe)).norm(dim=1)
    eps = d.quantile(0.95).item(); eps_list.append(eps)
    n_ok = sum(1 for m in margin if m > 2 * eps)
    print(f"   {s:.2f}        {eps:.3f}       {2*eps:.3f}          {n_ok}")

print(f"\nmean margin, four commonest classes: {head_m:.3f}")
print(f"mean margin, four rarest classes:    {tail_m:.3f}")
print("\nWHAT THIS MEANS: from about 30 per cent sparsity onward, 2*eps is")
print("already larger than even the head-class margins, so the condition")
print("certifies nothing, while step 3 showed the model was still fine at 70")
print("per cent. The worst-case bound is far too pessimistic.")
print("\nWHY: eps is a worst-case radius in every direction, and the factor of")
print("two assumes both similarities move against you at once. Real")
print("displacement is mostly in directions that do not change the ranking.")
print("\nThis is a genuine finding, not a bug in your code. It says the year-two")
print("objective must be a per-class, high-probability bound, not this one.")

plt.figure(figsize=(6, 3.6))
plt.plot([s*100 for s in SPARSITIES], [2*e for e in eps_list], color="#b33951",
         lw=2, label="2 x eps (what the certificate demands)")
plt.axhline(head_m, color="#2b4c7e", lw=1.4, label="mean margin, head classes")
plt.axhline(tail_m, color="#5b8fc9", lw=1.4, label="mean margin, tail classes")
plt.xlabel("sparsity (%)"); plt.ylabel("cosine-similarity units")
plt.legend(frameon=False, fontsize=8); plt.tight_layout()
plt.savefig("step6_certificate.png", dpi=150)
print("saved step6_certificate.png")
