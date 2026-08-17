"""
STEP 4 - measure the margin, the quantity that explains step 3.

Margin = similarity to the chosen prototype minus similarity to the runner-up.
A big margin means the decision is far from flipping.

Two versions are computed. The "labelled" one groups by the true class, which
you cannot do at deployment. The "label-free" one groups by the model's own
prediction and only needs the gap between the top two similarities. If the
label-free one works, the whole proposal becomes possible.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from common import build_world, get_model, N_CLASS

world = build_world(); model = get_model(world)
protos, x, y = world["protos"], world["x_te"], world["y_te"]

with torch.no_grad():
    sims = model(x) @ protos.t()            # cosine similarities, no temperature
    pred = sims.argmax(1)
    top2 = sims.topk(2, dim=1).values
    free_per_sample = top2[:, 0] - top2[:, 1]      # needs no labels at all

lab, free = {}, {}
for c in range(N_CLASS):
    s = sims[y == c]
    other = s.clone(); other[:, c] = -2.0
    lab[c] = (s[:, c] - other.max(1).values).mean().item()
    free[c] = free_per_sample[pred == c].mean().item()

print("\n class   train examples   labelled margin   label-free margin")
for c in range(N_CLASS):
    print(f"   {c:2d}        {world['counts'][c]:5d}          {lab[c]:.3f}             {free[c]:.3f}")

plt.figure(figsize=(5.6, 3.4))
plt.semilogx(world["counts"], [free[c] for c in range(N_CLASS)], "o-", color="#2b4c7e")
plt.xlabel("training examples (log scale)"); plt.ylabel("label-free margin")
plt.tight_layout(); plt.savefig("step4_margin.png", dpi=150)

print("\nWHAT TO CHECK: the two margin columns should be close to each other,")
print("and both should fall as the class gets rarer. Rare classes sit near")
print("their competitor, which is why they break first.")
print("saved step4_margin.png")
