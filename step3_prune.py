"""
STEP 3 - compress it, and watch who pays.

What you are learning: this is the phenomenon your whole proposal rests on. The
aggregate number stays respectable while the rare classes are being destroyed.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import build_world, get_model, accuracy, prune, N_CLASS

SPARSITIES = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
world = build_world(); model = get_model(world)
head, tail = list(range(4)), list(range(N_CLASS - 4, N_CLASS))

rows = []
print("\n sparsity   overall    head     tail")
for s in SPARSITIES:
    ov, per = accuracy(prune(model, s), world["x_te"], world["y_te"], world["protos"])
    h = np.mean([per[c] for c in head]); t = np.mean([per[c] for c in tail])
    rows.append((s, ov, h, t))
    print(f"   {s:.2f}      {ov:.3f}    {h:.3f}    {t:.3f}")

base = rows[0]
worst = max([r for r in rows if base[1] - r[1] <= 0.05], key=lambda r: r[0])
print(f"\nAt {worst[0]:.0%} sparsity the overall number has fallen only "
      f"{100*(base[1]-worst[1]):.1f} points,")
print(f"the head has lost {100*(base[2]-worst[2]):.1f} points,")
print(f"but the tail has lost {100*(base[3]-worst[3]):.1f} points.")

sp = [r[0]*100 for r in rows]
plt.figure(figsize=(6, 3.6))
plt.plot(sp, [r[2] for r in rows], color="#2b4c7e", lw=2, label="head classes")
plt.plot(sp, [r[1] for r in rows], color="#707b8c", lw=1.6, ls="--", label="overall")
plt.plot(sp, [r[3] for r in rows], color="#b33951", lw=2, label="tail classes")
plt.xlabel("sparsity (%)"); plt.ylabel("accuracy"); plt.ylim(0, 1.05)
plt.legend(frameon=False); plt.tight_layout(); plt.savefig("step3_prune.png", dpi=150)
print("saved step3_prune.png")
