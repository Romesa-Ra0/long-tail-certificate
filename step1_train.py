"""
STEP 1 - build the model and check it works.

What you are learning: a vision-language model does not have a normal output
layer. It embeds the input, and classifies by asking which fixed class vector
that embedding is closest to. Everything later depends on this geometry.
"""
from common import build_world, get_model, accuracy

world = build_world()
print("training-set size per class (long tailed):", world["counts"])

model = get_model(world)
overall, per = accuracy(model, world["x_te"], world["y_te"], world["protos"])

print(f"\noverall test accuracy: {overall:.3f}")
print("\nWHAT TO CHECK: this should be above 0.95. If the dense model is weak,")
print("nothing measured later means anything, because you cannot tell damage")
print("caused by compression from damage that was there all along.")
