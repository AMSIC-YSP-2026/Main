"""
Plot the cost function (MSE) over iteration count from a saved
cost_history.npy file produced by the gradient descent training script.

Just set COST_HISTORY_PATH below and run:
    python plot_cost.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# ==== EDIT THESE ====
COST_HISTORY_PATH = ".\\MachineLearning\\cost_history.npy"
OUTPUT_PATH = ".\\MachineLearning\\cost_plot.png"   # None = show plot in a window
LOG_SCALE = False    # set True to view cost on a log scale (useful if it drops fast)
# =====================


def main():
    try:
        cost_history = np.load(COST_HISTORY_PATH)
    except FileNotFoundError:
        print(f"Error: {COST_HISTORY_PATH} not found. Run the training script first.", file=sys.stderr)
        sys.exit(1)

    iterations = np.arange(len(cost_history))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, cost_history, color="blue", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function vs. Iteration Count")
    plt.grid(True, alpha=0.3)

    if LOG_SCALE:
        plt.yscale("log")

    if OUTPUT_PATH:
        plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {OUTPUT_PATH}")
    else:
        plt.show()

if __name__ == "__main__":
    main()