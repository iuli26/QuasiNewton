import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open("optimizer_result_1.pkl", "rb") as f:
    results = pickle.load(f)

df = pd.DataFrame(results)
print(df)

# --- Combined Plots ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- Plot Iterations ---
ax = axes[0, 0]
for n in df["n"].unique():
    subset = df[df["n"] == n]
    labels = (subset["method"].astype(str) +
              f" (n={n})")
    ax.bar(labels, subset["iterations"], label=f"n={n}")
ax.set_ylabel("Iterations")
ax.set_title("Iterations per Optimizer & Line Search")
ax.tick_params(axis="x", rotation=60)

# --- Plot Function Value ---
ax = axes[0, 1]
for n in df["n"].unique():
    subset = df[df["n"] == n]
    labels = (subset["method"].astype(str) +
              f" (n={n})")
    ax.bar(labels, subset["f_min"], label=f"n={n}")
ax.set_ylim(0, 0.0035)
ax.set_ylabel("Final Function Value")
ax.set_title("Function Value per Optimizer & Line Search")
ax.tick_params(axis="x", rotation=60)

# --- Plot Error vs SciPy ---
ax = axes[1, 0]
for n in df["n"].unique():
    subset = df[df["n"] == n]
    labels = (subset["method"].astype(str) +
              f" (n={n})")
    ax.bar(labels, subset["diff_vs_scipy"], label=f"n={n}")
ax.set_ylabel("Difference vs SciPy (%)")
ax.set_title("Error vs SciPy Minimizer")
ax.tick_params(axis="x", rotation=60)

# --- Plot Runtime ---
ax = axes[1, 1]
for n in df["n"].unique():
    subset = df[df["n"] == n]
    labels = (subset["method"].astype(str) +
              f" (n={n})")
    ax.bar(labels, subset["time"], label=f"n={n}")
ax.set_ylabel("Runtime (s)")
ax.set_title("Runtime per Optimizer & Line Search")
ax.tick_params(axis="x", rotation=60)

# Layout
plt.tight_layout()
plt.show()
