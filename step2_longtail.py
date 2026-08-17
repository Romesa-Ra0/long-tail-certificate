"""
STEP 2 - look at the tail before touching anything.

What you are learning: an overall accuracy number is an average weighted by how
common each class is. Before you can claim compression hurt the tail, you must
show the tail was fine to begin with.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import build_world, get_model, accuracy, N_CLASS

world = build_world()
model = get_model(world)
overall, per = accuracy(model, world["x_te"], world["y_te"], world["protos"])

print(f"\noverall accuracy {overall:.3f}\n")
print(" class   train examples   accuracy")
for c in range(N_CLASS):
    print(f"   {c:2d}        {world['counts'][c]:5d}         {per[c]:.3f}")

plt.figure(figsize=(6, 3.4))
plt.bar(range(N_CLASS), [per[c] for c in range(N_CLASS)], color="#2b4c7e")
plt.xlabel("class (0 = most common, 11 = rarest)"); plt.ylabel("dense accuracy")
plt.ylim(0, 1.05); plt.tight_layout(); plt.savefig("step2_per_class.png", dpi=150)

print("\nWHAT TO CHECK: even the rarest classes should be well above chance")
print("(chance is 1/12 = 0.083). The dense model has learned the tail.")
print("saved step2_per_class.png")
